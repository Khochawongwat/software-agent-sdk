"""Own one hardened agent-server container per conversation."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.error import URLError
from urllib.request import urlopen
from uuid import UUID

import httpx

from openhands.agent_server.config import V1_SESSION_API_KEY_ENV, Config
from openhands.agent_server.conversation_registry import ConversationRegistry
from openhands.agent_server.docker_runtime.provisioning import RuntimeProvisioningStore
from openhands.agent_server.models import (
    ConversationRuntimeError,
    ConversationRuntimeInfo,
    ConversationRuntimeStatus,
)
from openhands.agent_server.persistence.store import _get_persistence_dir
from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands.sdk.logger import get_logger
from openhands.sdk.utils.cipher import Cipher
from openhands.sdk.utils.command import execute_command, sanitized_env


if TYPE_CHECKING:
    from fastapi import APIRouter

    from openhands.agent_server.conversation_service import ConversationService


logger = get_logger(__name__)

_CONVERSATIONS_DIR = "/var/openhands/conversations"
_PERSISTENCE_DIR = "/var/openhands/.openhands"
_WORKSPACE_DIR = "/workspace"
_OWNER_LABEL = "ai.openhands.runtime-owner"
_CONVERSATION_LABEL = "ai.openhands.conversation-id"
# How often running containers are checked, so one that died (out of memory,
# killed) is reported as an error instead of staying "available".
# ponytail: polling can't tell the exit code or OOM; `docker events` can.
_WATCH_INTERVAL_SECONDS = 10.0
_RESUME_AT_MOST_EVERY_SECONDS = 3600.0
_CONTAINER_EXITED = ConversationRuntimeError(
    code="container_exited",
    message="The conversation's container stopped unexpectedly.",
)
# Set by the runtime for each container; never taken from the server's env.
_RUNTIME_ENV = (
    "HOME",
    "OH_CONVERSATIONS_PATH",
    "OH_PERSISTENCE_DIR",
    "OH_CONVERSATION_RUNTIME",
    "OH_SECRET_KEY",
    V1_SESSION_API_KEY_ENV,
    "OH_RUNTIME_LAUNCHED_PROFILE",
)


@dataclass(slots=True)
class ConversationContainer:
    host: str
    api_key: str
    container_id: str

    def stop(self) -> None:
        result = execute_command(["docker", "stop", self.container_id])
        if result.returncode != 0 and "No such container" not in result.stderr:
            raise RuntimeError(
                f"Failed to stop conversation container: {result.stderr}"
            )

    def is_running(self) -> bool:
        result = execute_command(
            ["docker", "inspect", "-f", "{{.State.Running}}", self.container_id]
        )
        return result.returncode == 0 and result.stdout.strip() == "true"


class DockerConversationRegistry(ConversationRegistry):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        paths = (
            f"{config.conversations_path.resolve()}\0"
            f"{_get_persistence_dir(config).resolve()}"
        )
        self.owner = hashlib.sha256(paths.encode()).hexdigest()[:24]
        self.provisioning = RuntimeProvisioningStore(config)
        self._containers: dict[UUID, ConversationContainer] = {}
        self._starts: dict[UUID, asyncio.Task[ConversationContainer]] = {}
        self._deleting: set[UUID] = set()
        self._lock = asyncio.Lock()
        self._service: ConversationService | None = None
        self._last_access: dict[UUID, float] = {}
        self._sessions: dict[UUID, int] = {}
        self._eviction_task: asyncio.Task[None] | None = None
        self._watch_task: asyncio.Task[None] | None = None
        self._crashed: dict[UUID, ConversationRuntimeError] = {}
        self._resumed: dict[UUID, float] = {}
        # Conversations running a turn at the last check (resume on crash).
        self._turns: set[UUID] = set()
        self._resumes: set[asyncio.Task[None]] = set()

    def configure_service(self, service: ConversationService) -> None:
        self._service = service
        service.runtime_cipher_resolver = self.resolve_persisted_cipher

    def resolve_persisted_cipher(self, conversation_id: UUID) -> Cipher:
        """Resolve persisted state without weakening per-runtime isolation.

        Conversations created before Docker mode have no provisioning identity and
        were encrypted with the host key. A present but invalid identity still
        raises rather than falling back to that key.
        """
        identity = self.provisioning.load_optional(conversation_id)
        return identity.cipher if identity is not None else self.provisioning.cipher

    def runtime_info(self, conversation_id: UUID) -> ConversationRuntimeInfo:
        identity = self.provisioning.load_optional(conversation_id)
        if identity is None:
            return ConversationRuntimeInfo(
                runtime_status=ConversationRuntimeStatus.MISSING,
                can_resume=False,
            )
        if self.get(conversation_id):
            status = ConversationRuntimeStatus.AVAILABLE
        elif self.is_starting(conversation_id):
            status = ConversationRuntimeStatus.STARTING
        elif conversation_id in self._crashed:
            return ConversationRuntimeInfo(
                runtime_status=ConversationRuntimeStatus.ERROR,
                can_resume=True,
                runtime_error=self._crashed[conversation_id],
            )
        else:
            status = ConversationRuntimeStatus.MISSING
        return ConversationRuntimeInfo(runtime_status=status, can_resume=True)

    @property
    def serves_persisted_event_reads(self) -> bool:
        return True

    async def start(self) -> None:
        await asyncio.to_thread(self.cleanup_stale_containers)
        self._watch_task = asyncio.create_task(self._watch_containers_loop())
        for conversation_id in self._take_interrupted():
            self._resume_later(conversation_id)
        if self.config.conversation_idle_ttl_seconds:
            self._eviction_task = asyncio.create_task(self._evict_idle_runtimes_loop())

    def add_execution_routes(self, router: APIRouter) -> None:
        from openhands.agent_server.docker_runtime.routers import (
            docker_conversation_router,
        )
        from openhands.agent_server.event_router import event_read_router

        # Persisted event history is safe to read from the outer catalog for
        # both Docker-backed and historical host-local conversations. Writes
        # continue through the runtime proxy below.
        router.include_router(event_read_router)
        router.include_router(docker_conversation_router)

    @property
    def workspace_router(self) -> APIRouter:
        from openhands.agent_server.docker_runtime.routers import (
            docker_workspace_router,
        )

        return docker_workspace_router

    @property
    def conversation_sockets_router(self) -> APIRouter:
        from openhands.agent_server.docker_runtime.routers import docker_sockets_router

        return docker_sockets_router

    @property
    def session_sockets_router(self) -> APIRouter:
        from openhands.agent_server.docker_runtime.routers import (
            docker_session_sockets_router,
        )

        return docker_session_sockets_router

    def conversation_dir(self, conversation_id: UUID) -> Path:
        return self.provisioning.direct_child(
            self.config.conversations_path, conversation_id.hex
        )

    def workspace_dir(self, conversation_id: UUID) -> Path:
        workspace = self.provisioning.load(conversation_id).workspace_path
        if workspace.is_symlink():
            raise ValueError("Conversation workspace must not be a symlink")
        return workspace.resolve()

    def get(self, conversation_id: UUID) -> ConversationContainer | None:
        return self._containers.get(conversation_id)

    def is_starting(self, conversation_id: UUID) -> bool:
        return conversation_id in self._starts

    def attach_session(self, conversation_id: UUID) -> None:
        """Record an outer proxied session attached to this runtime.

        Non-zero counts suppress idle eviction, mirroring the inner
        ``EventService.has_external_subscribers()`` guard: a client holding a
        live events websocket or a long-lived proxied stream keeps the
        container alive even while the conversation itself looks idle.
        """
        self._sessions[conversation_id] = self._sessions.get(conversation_id, 0) + 1

    def detach_session(self, conversation_id: UUID) -> None:
        """Release a session recorded by :meth:`attach_session`."""
        remaining = self._sessions.get(conversation_id, 0) - 1
        if remaining > 0:
            self._sessions[conversation_id] = remaining
        else:
            self._sessions.pop(conversation_id, None)
        self._last_access[conversation_id] = time.monotonic()

    def has_attached_sessions(self, conversation_id: UUID) -> bool:
        """True if an outer proxied session is currently attached."""
        return self._sessions.get(conversation_id, 0) > 0

    def cleanup_stale_containers(self) -> None:
        result = execute_command(
            ["docker", "ps", "-aq", "--filter", f"label={_OWNER_LABEL}={self.owner}"]
        )
        if result.returncode != 0:
            logger.warning("Failed to list stale conversation containers")
            return
        ids = result.stdout.split()
        if ids:
            execute_command(["docker", "rm", "-f", *ids])

    async def get_or_create(self, conversation_id: UUID) -> ConversationContainer:
        async with self._lock:
            if conversation_id in self._deleting:
                raise RuntimeError("Conversation is being deleted")
            self._last_access[conversation_id] = time.monotonic()
            container = self._containers.get(conversation_id)

        if container is not None:
            if await asyncio.to_thread(container.is_running):
                return container
            async with self._lock:
                if self._containers.get(conversation_id) is container:
                    self._containers.pop(conversation_id)

        async with self._lock:
            if conversation_id in self._deleting:
                raise RuntimeError("Conversation is being deleted")
            task = self._starts.get(conversation_id)
            if task is None:
                task = asyncio.create_task(
                    asyncio.to_thread(self._build_container, conversation_id)
                )
                self._starts[conversation_id] = task

        try:
            container = await asyncio.shield(task)
        except BaseException:
            async with self._lock:
                if self._starts.get(conversation_id) is task:
                    self._starts.pop(conversation_id, None)
            raise

        async with self._lock:
            existing = self._containers.get(conversation_id)
            if existing is not None:
                if existing is not container:
                    await asyncio.to_thread(container.stop)
                self._last_access[conversation_id] = time.monotonic()
                return existing
            if self._starts.get(conversation_id) is not task:
                await asyncio.to_thread(container.stop)
                raise RuntimeError("Conversation container start was cancelled")
            self._starts.pop(conversation_id, None)
            self._containers[conversation_id] = container
            self._crashed.pop(conversation_id, None)
            self._last_access[conversation_id] = time.monotonic()
            return container

    async def begin_delete(self, conversation_id: UUID) -> bool:
        async with self._lock:
            if conversation_id in self._deleting:
                return False
            self._deleting.add(conversation_id)
            return True

    async def finish_delete(self, conversation_id: UUID) -> None:
        async with self._lock:
            self._deleting.discard(conversation_id)

    async def stop(self, conversation_id: UUID) -> None:
        async with self._lock:
            task = self._starts.pop(conversation_id, None)
            container = self._containers.pop(conversation_id, None)
            self._last_access.pop(conversation_id, None)
            self._sessions.pop(conversation_id, None)
        if task is not None:
            try:
                started = await task
            except Exception:
                started = None
            container = container or started
        if container is not None:
            await asyncio.to_thread(container.stop)

    async def shutdown(self) -> None:
        if self.config.conversation_resume_message:
            await self._remember_interrupted()
        for name in ("_eviction_task", "_watch_task"):
            task = getattr(self, name)
            if task is not None:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
                setattr(self, name, None)
        ids = set(self._containers) | set(self._starts)
        await asyncio.gather(*(self.stop(cid) for cid in ids), return_exceptions=True)

    def _running_container_ids(self) -> set[str]:
        result = execute_command(
            [
                "docker",
                "ps",
                "-q",
                "--no-trunc",
                "--filter",
                f"label={_OWNER_LABEL}={self.owner}",
            ]
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "docker ps failed")
        return set(result.stdout.split())

    async def _watch_containers_loop(self) -> None:
        while True:
            await asyncio.sleep(_WATCH_INTERVAL_SECONDS)
            try:
                await self.check_containers()
            except Exception:
                logger.exception("error_checking_docker_runtimes")

    async def check_containers(self) -> list[UUID]:
        """Drop containers that died on their own; return their conversations.

        A container the runtime stops is removed from the registry first, so
        only an unexpected exit (out of memory, killed) is reported.
        """
        running = await asyncio.to_thread(self._running_container_ids)
        async with self._lock:
            dead = [
                conversation_id
                for conversation_id, container in self._containers.items()
                if container.container_id not in running
            ]
            for conversation_id in dead:
                self._containers.pop(conversation_id)
                self._last_access.pop(conversation_id, None)
                self._sessions.pop(conversation_id, None)
                self._crashed[conversation_id] = _CONTAINER_EXITED
        for conversation_id in dead:
            logger.warning("Conversation container for %s exited", conversation_id)
            # A container saves its state at the end of a turn, so after a
            # crash only this check's last answer shows the turn was running.
            if conversation_id in self._turns:
                self._resume_later(conversation_id)
        if self.config.conversation_resume_message:
            await self._note_turns()
        return dead

    async def _note_turns(self) -> None:
        """Ask each live container whether it is running a turn."""
        containers = list(self._containers.items())
        answers = await asyncio.gather(
            *(self._turn_running(cid, container) for cid, container in containers),
            return_exceptions=True,
        )
        self._turns = {cid for (cid, _), on in zip(containers, answers) if on is True}

    # Resuming interrupted turns (``conversation_resume_message``).

    @property
    def _interrupted_path(self) -> Path:
        return self.provisioning.control_root / "interrupted.json"

    async def _turn_running(
        self, conversation_id: UUID, container: ConversationContainer
    ) -> bool:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(
                f"{container.host}/api/conversations/{conversation_id}",
                headers={"X-Session-API-Key": container.api_key},
            )
        return response.json().get("execution_status") == (
            ConversationExecutionStatus.RUNNING.value
        )

    async def _remember_interrupted(self) -> None:
        """Before stopping: note the conversations that are running a turn."""
        containers = list(self._containers.items())
        running = await asyncio.gather(
            *(self._turn_running(cid, container) for cid, container in containers),
            return_exceptions=True,
        )
        ids = [str(cid) for (cid, _), on in zip(containers, running) if on is True]
        await asyncio.to_thread(self._interrupted_path.write_text, json.dumps(ids))

    def _take_interrupted(self) -> list[UUID]:
        """The conversations noted at the last stop, once."""
        path = self._interrupted_path
        if not self.config.conversation_resume_message or not path.is_file():
            return []
        ids = [UUID(value) for value in json.loads(path.read_text())]
        path.unlink()
        return ids

    def _resume_later(self, conversation_id: UUID) -> None:
        now = time.monotonic()
        last = self._resumed.get(conversation_id)
        if last is not None and now - last < _RESUME_AT_MOST_EVERY_SECONDS:
            logger.warning("Not resuming %s again within the hour", conversation_id)
            return
        self._resumed[conversation_id] = now
        task = asyncio.create_task(self._resume(conversation_id))
        self._resumes.add(task)
        task.add_done_callback(self._resumes.discard)

    async def _resume(self, conversation_id: UUID) -> None:
        try:
            container = await self.get_or_create(conversation_id)
            await self._send_message(
                container, conversation_id, self.config.conversation_resume_message
            )
            logger.info("Resumed interrupted conversation %s", conversation_id)
        except Exception:
            logger.exception("Could not resume conversation %s", conversation_id)

    async def _send_message(
        self, container: ConversationContainer, conversation_id: UUID, text: str | None
    ) -> None:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{container.host}/api/conversations/{conversation_id}/events",
                headers={"X-Session-API-Key": container.api_key},
                json={
                    "role": "user",
                    "content": [{"type": "text", "text": text}],
                    "run": True,
                },
            )
        response.raise_for_status()

    async def _evict_idle_runtimes_loop(self) -> None:
        ttl = self.config.conversation_idle_ttl_seconds
        if not ttl:
            return
        interval = max(1.0, min(60.0, ttl / 2))
        while True:
            await asyncio.sleep(interval)
            try:
                await self._evict_idle_runtimes(ttl)
            except Exception:
                logger.exception("error_evicting_idle_docker_runtimes")

    async def _evict_idle_runtimes(self, ttl_seconds: float) -> None:
        service = self._service
        if service is None:
            return
        cutoff = time.monotonic() - ttl_seconds
        async with self._lock:
            candidates = [
                (conversation_id, container)
                for conversation_id, container in self._containers.items()
                if self._last_access.get(conversation_id, float("inf")) <= cutoff
                and not self.has_attached_sessions(conversation_id)
            ]

        for conversation_id, container in candidates:
            info = await service.get_conversation(conversation_id)
            if (
                info is None
                or info.execution_status == ConversationExecutionStatus.RUNNING
            ):
                continue
            async with self._lock:
                if self._containers.get(conversation_id) is not container:
                    continue
                if self._last_access.get(conversation_id, float("inf")) > cutoff:
                    continue
                if self.has_attached_sessions(conversation_id):
                    continue
                self._containers.pop(conversation_id)
                self._last_access.pop(conversation_id, None)
            try:
                await asyncio.to_thread(container.stop)
            except Exception:
                async with self._lock:
                    if conversation_id not in self._containers:
                        self._containers[conversation_id] = container
                        self._last_access[conversation_id] = time.monotonic()
                logger.warning(
                    "Failed to stop idle conversation runtime %s",
                    conversation_id,
                    exc_info=True,
                )
            else:
                logger.info(
                    "Stopped idle conversation runtime %s (idle >= %.0fs)",
                    conversation_id,
                    ttl_seconds,
                )

    def _build_container(self, conversation_id: UUID) -> ConversationContainer:
        identity = self.provisioning.load(conversation_id)
        runtime_dir = self.provisioning.runtime_dir(conversation_id)
        persistence_dir = self.provisioning.direct_child(runtime_dir, "persistence")
        conversation_dir = self.conversation_dir(conversation_id)
        workspace_dir = self.workspace_dir(conversation_id)
        for directory in (persistence_dir, conversation_dir, workspace_dir):
            directory.mkdir(parents=True, mode=0o700, exist_ok=True)

        env = sanitized_env()
        env.update(
            {
                "HOME": _PERSISTENCE_DIR,
                "OH_CONVERSATIONS_PATH": _CONVERSATIONS_DIR,
                "OH_PERSISTENCE_DIR": _PERSISTENCE_DIR,
                "OH_CONVERSATION_RUNTIME": "local",
                "OH_SECRET_KEY": identity.encryption_key.get_secret_value(),
                V1_SESSION_API_KEY_ENV: identity.api_key.get_secret_value(),
                "OH_RUNTIME_LAUNCHED_PROFILE": (
                    identity.launched_agent_profile.model_dump_json()
                    if identity.launched_agent_profile
                    else ""
                ),
            }
        )
        if "DEBUG" in os.environ:
            env["DEBUG"] = os.environ["DEBUG"]
        # Operator-chosen variables, never the ones that isolate this runtime.
        passed = [
            name
            for name in self.config.conversation_container_env
            if name in os.environ and name not in (*_RUNTIME_ENV, "DEBUG")
        ]
        for name in passed:
            env[name] = os.environ[name]

        flags: list[str] = []
        for name in (*_RUNTIME_ENV, "DEBUG", *passed):
            if name in env:
                flags.extend(("-e", name))
        for host, target in (
            (conversation_dir, f"{_CONVERSATIONS_DIR}/{conversation_id.hex}"),
            (persistence_dir, _PERSISTENCE_DIR),
            (workspace_dir, _WORKSPACE_DIR),
        ):
            flags.extend(("-v", f"{host}:{target}"))
        for volume in self.config.conversation_container_volumes:
            flags.extend(("-v", volume))
        network = self.config.conversation_container_network
        # On a shared network the server reaches the container by name, so
        # nothing is published on the Docker host.
        flags.extend(("--network", network) if network else ("-p", "127.0.0.1::8000"))
        if self.config.conversation_container_memory:
            flags.extend(("--memory", self.config.conversation_container_memory))
        if self.config.conversation_container_cpus is not None:
            flags.extend(("--cpus", str(self.config.conversation_container_cpus)))
        if self.config.conversation_container_pids_limit is not None:
            flags.extend(
                ("--pids-limit", str(self.config.conversation_container_pids_limit))
            )

        # Named after its conversation, so a deployment can find it; a leftover
        # container of the same conversation is removed first.
        name = f"agent-server-conversation-{conversation_id}"
        execute_command(["docker", "rm", "-f", name])
        command = [
            "docker",
            "run",
            "-d",
            "--rm",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--add-host",
            "host.docker.internal:host-gateway",
            "--label",
            f"{_OWNER_LABEL}={self.owner}",
            "--label",
            f"{_CONVERSATION_LABEL}={conversation_id}",
            "--name",
            name,
            *flags,
            self.config.conversation_image,
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
        ]
        result = subprocess.run(
            command, env=env, capture_output=True, text=True, check=False
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "docker run failed")

        container_id = result.stdout.strip()
        try:
            if network:
                host = f"http://{name}:8000"
            else:
                binding = execute_command(["docker", "port", container_id, "8000/tcp"])
                address, port = binding.stdout.strip().rsplit(":", 1)
                if binding.returncode != 0 or address != "127.0.0.1":
                    raise RuntimeError("Docker did not create a loopback port binding")
                host = f"http://127.0.0.1:{int(port)}"
            container = ConversationContainer(
                host=host,
                api_key=identity.api_key.get_secret_value(),
                container_id=container_id,
            )
            self._wait_until_ready(container)
            return container
        except BaseException:
            execute_command(["docker", "stop", container_id])
            raise

    def _wait_until_ready(self, container: ConversationContainer) -> None:
        deadline = time.monotonic() + self.config.conversation_container_startup_timeout
        while time.monotonic() < deadline:
            try:
                with urlopen(container.host + "/health", timeout=1) as response:
                    if 200 <= response.status < 300:
                        return
            except (URLError, TimeoutError, ConnectionError):
                pass
            running = execute_command(
                [
                    "docker",
                    "inspect",
                    "-f",
                    "{{.State.Running}}",
                    container.container_id,
                ]
            )
            if running.stdout.strip() != "true":
                raise RuntimeError("Conversation container stopped during startup")
            time.sleep(1)
        raise RuntimeError("Conversation container failed to become healthy in time")
