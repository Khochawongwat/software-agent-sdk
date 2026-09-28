"""Conversation routes for the Docker runtime."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from pathlib import Path
from typing import Annotated
from uuid import UUID, uuid4

import httpx
from fastapi import APIRouter, HTTPException, Query, Request, WebSocket
from starlette.responses import JSONResponse, Response, StreamingResponse

from openhands.agent_server.config import Config
from openhands.agent_server.dependencies import get_conversation_service
from openhands.agent_server.docker_runtime.mediation import (
    materialize_secrets,
    prepare_start,
    serialize_start,
)
from openhands.agent_server.docker_runtime.proxy import (
    bridge_websocket,
    proxy_http,
    strip_auth_query,
)
from openhands.agent_server.docker_runtime.registry import (
    ConversationContainer,
    DockerConversationRegistry,
)
from openhands.agent_server.models import (
    ConversationRuntimeInfo,
    ConversationRuntimeStatus,
    UpdateSecretsRequest,
)
from openhands.agent_server.persistence import get_llm_profile_store
from openhands.agent_server.utils import safe_rmtree
from openhands.sdk.conversation.request import StartConversationRequest
from openhands.sdk.conversation.title_utils import generate_title_from_message
from openhands.sdk.llm import LLM, TextContent
from openhands.sdk.logger import get_logger
from openhands.sdk.profiles.resolver import DanglingMcpServerRef, ProfileNotFound


logger = get_logger(__name__)
# Background title tasks, kept referenced until they finish.
_background_tasks: set[asyncio.Task] = set()


def get_registry(request: Request) -> DockerConversationRegistry:
    registry = request.app.state.conversation_registry
    if not isinstance(registry, DockerConversationRegistry):
        raise HTTPException(503, "Docker conversation runtime is unavailable")
    return registry


async def _container(
    registry: DockerConversationRegistry, conversation_id: UUID
) -> ConversationContainer:
    if not registry.provisioning.manifest_path(conversation_id).is_file():
        raise HTTPException(404, "Conversation not found")
    try:
        return await registry.get_or_create(conversation_id)
    except Exception as exc:
        logger.exception("Could not start conversation container %s", conversation_id)
        raise HTTPException(502, "Could not start conversation container") from exc


def _upstream_path(request: Request, path: str) -> str:
    query = strip_auth_query("?" + request.url.query).lstrip("?")
    return f"{path}?{query}" if query else path


async def _proxy_with_session(
    registry: DockerConversationRegistry,
    conversation_id: UUID,
    container: ConversationContainer,
    request: Request,
    *,
    upstream_path: str,
    body: bytes | None = None,
) -> StreamingResponse:
    """Proxy one request while holding an outer session attachment.

    The attachment is released when the streamed response finishes (or the
    client disconnects), so a long-lived proxied stream keeps the container
    from being evicted while a client is still reading from it.
    """
    registry.attach_session(conversation_id)

    async def detach() -> None:
        registry.detach_session(conversation_id)

    try:
        return await proxy_http(
            request,
            container,
            upstream_path=upstream_path,
            body=body,
            on_close=detach,
        )
    except BaseException:
        await detach()
        raise


def _first_message_text(request: StartConversationRequest) -> str | None:
    message = request.initial_message
    if message is None:
        return None
    text = "\n".join(c.text for c in message.content if isinstance(c, TextContent))
    return text.strip() or None


def _title_llm(request: StartConversationRequest, config: Config) -> LLM | None:
    """Pick the title LLM as a local conversation would: the title profile,
    else the agent's LLM (None, or ACP's placeholder, means truncation). The
    profile loads here, where profiles are stored; a container has none."""
    if request.title_llm_profile:
        try:
            return get_llm_profile_store().load(
                request.title_llm_profile, cipher=config.cipher
            )
        except (FileNotFoundError, ValueError) as exc:
            logger.warning(
                "Failed to load title LLM profile %r: %s. Falling back to the "
                "agent's LLM.",
                request.title_llm_profile,
                exc,
            )
    return getattr(request.agent, "llm", None)


async def _set_title(
    service,
    container: ConversationContainer,
    conversation_id: UUID,
    message: str,
    request: StartConversationRequest,
    config: Config,
) -> None:
    """Title the conversation from its first message, then save it through the
    container's own API, so its meta.json and the outer catalog agree."""
    try:
        llm = await asyncio.to_thread(_title_llm, request, config)
        title = await asyncio.to_thread(generate_title_from_message, message, llm, 50)
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.patch(
                f"{container.host}/api/conversations/{conversation_id}",
                headers={"X-Session-API-Key": container.api_key},
                json={"title": title},
            )
            response.raise_for_status()
        await service.refresh_persisted_conversation(conversation_id)
    except Exception:
        logger.warning(
            "Auto-title generation failed for conversation %s",
            conversation_id,
            exc_info=True,
        )


# ponytail: one fixed limit; make it a setting if a workspace takes longer.
WORKSPACE_COMMAND_TIMEOUT_S = 600


class WorkspaceCommandError(Exception):
    """The workspace command failed or printed no usable directory."""


def prepare_workspace(
    command: list[str], working_dir: Path, conversation_id: UUID, tags: object
) -> Path:
    """Run ``conversation_workspace_command``; the directory it printed."""
    try:
        result = subprocess.run(
            [*command, str(working_dir), str(conversation_id)],
            capture_output=True,
            text=True,
            timeout=WORKSPACE_COMMAND_TIMEOUT_S,
            env={**os.environ, "OH_CONVERSATION_TAGS": json.dumps(tags or {})},
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise WorkspaceCommandError(f"Workspace command could not run: {exc}") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()[-500:]
        raise WorkspaceCommandError(f"Workspace command failed: {detail}")
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    workspace = Path(lines[-1]) if lines else None
    if workspace is None or not workspace.is_absolute() or not workspace.is_dir():
        raise WorkspaceCommandError(
            "Workspace command did not print an existing absolute directory"
        )
    return workspace


docker_conversation_router = APIRouter(
    prefix="/conversations", tags=["Docker Conversations"]
)


@docker_conversation_router.post("")
async def start_conversation(
    request: Request,
    include_skills: Annotated[bool, Query()] = False,
) -> JSONResponse:
    try:
        body = json.loads(await request.body())
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise HTTPException(400, "Invalid JSON body") from exc
    if not isinstance(body, dict):
        raise HTTPException(422, "Expected a JSON object")

    workspace = body.get("workspace")
    if workspace is not None and (
        not isinstance(workspace, dict)
        or workspace.get("kind", "LocalWorkspace") != "LocalWorkspace"
    ):
        raise HTTPException(422, "Docker conversations require a local workspace")
    working_dir = (workspace or {}).get("working_dir", "/workspace")
    host_workspace = None if working_dir == "/workspace" else Path(working_dir)

    try:
        conversation_id = (
            UUID(body["conversation_id"]) if body.get("conversation_id") else uuid4()
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, "Invalid conversation_id") from exc
    body["conversation_id"] = str(conversation_id)
    body["workspace"] = {"kind": "LocalWorkspace", "working_dir": "/workspace"}

    registry = get_registry(request)
    command = registry.config.conversation_workspace_command
    if host_workspace is not None and command:
        if registry.provisioning.manifest_path(conversation_id).is_file():
            # Made when the conversation first started; that one stays.
            host_workspace = None
        else:
            try:
                host_workspace = await asyncio.to_thread(
                    prepare_workspace,
                    command,
                    host_workspace,
                    conversation_id,
                    body.get("tags"),
                )
            except WorkspaceCommandError as exc:
                raise HTTPException(502, str(exc)) from exc
    try:
        prepared, launched = await prepare_start(body, registry.config)
        # The outer server titles the conversation (_set_title): the
        # container has no LLM profiles to title it with.
        title_message = _first_message_text(prepared) if prepared.autotitle else None
        prepared = prepared.model_copy(update={"autotitle": False})
        identity = registry.provisioning.create(conversation_id, host_workspace)
        if launched is not None and identity.launched_agent_profile is None:
            identity = identity.model_copy(update={"launched_agent_profile": launched})
            registry.provisioning.save(identity)
        container = await registry.get_or_create(conversation_id)
        payload = serialize_start(prepared, identity)
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{container.host}/api/conversations",
                params={"include_skills": include_skills},
                headers={"X-Session-API-Key": container.api_key},
                json=payload,
            )
    except ProfileNotFound as exc:
        raise HTTPException(404, str(exc)) from exc
    except DanglingMcpServerRef as exc:
        raise HTTPException(
            422,
            {"message": str(exc), "dangling_mcp_server_refs": exc.missing},
        ) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except httpx.HTTPError as exc:
        await registry.stop(conversation_id)
        raise HTTPException(
            502, "Conversation container did not accept the request"
        ) from exc
    except Exception as exc:
        await registry.stop(conversation_id)
        logger.exception("Could not create conversation container")
        raise HTTPException(502, "Could not create conversation container") from exc

    content = response.json() if response.content else None
    if response.is_error:
        await registry.stop(conversation_id)
        content = {"detail": "Conversation runtime rejected the request"}
    else:
        service = get_conversation_service(request)
        await service.refresh_persisted_conversation(conversation_id)
        if title_message:
            task = asyncio.create_task(
                _set_title(
                    service,
                    container,
                    conversation_id,
                    title_message,
                    prepared,
                    registry.config,
                )
            )
            _background_tasks.add(task)
            task.add_done_callback(_background_tasks.discard)
    return JSONResponse(content=content, status_code=response.status_code)


@docker_conversation_router.get(
    "/{conversation_id}/runtime", response_model=ConversationRuntimeInfo
)
async def runtime_info(
    conversation_id: UUID, request: Request
) -> ConversationRuntimeInfo:
    registry = get_registry(request)
    if not registry.conversation_dir(conversation_id).joinpath("meta.json").is_file():
        raise HTTPException(404, "Conversation not found")
    return registry.runtime_info(conversation_id)


@docker_conversation_router.post("/{conversation_id}/runtime/credentials")
async def runtime_credentials(conversation_id: UUID, request: Request) -> JSONResponse:
    """Return the key for an authenticated client attached to this runtime."""
    registry = get_registry(request)
    if not registry.conversation_dir(conversation_id).joinpath("meta.json").is_file():
        raise HTTPException(404, "Conversation not found")
    identity = registry.provisioning.load(conversation_id)
    return JSONResponse({"session_api_key": identity.api_key.get_secret_value()})


@docker_conversation_router.delete("/{conversation_id}/runtime", status_code=204)
async def release_runtime(conversation_id: UUID, request: Request) -> Response:
    """Stop the container while retaining conversation and workspace state."""
    registry = get_registry(request)
    if not registry.conversation_dir(conversation_id).joinpath("meta.json").is_file():
        raise HTTPException(404, "Conversation not found")
    try:
        await registry.stop(conversation_id)
        await get_conversation_service(request).refresh_persisted_conversation(
            conversation_id
        )
    except Exception as exc:
        logger.exception("Could not release conversation runtime %s", conversation_id)
        raise HTTPException(502, "Could not release conversation runtime") from exc
    return Response(status_code=204)


@docker_conversation_router.post(
    "/{conversation_id}/runtime/reprovision", response_model=ConversationRuntimeInfo
)
async def reprovision_runtime(
    conversation_id: UUID, request: Request
) -> ConversationRuntimeInfo:
    registry = get_registry(request)
    await _container(registry, conversation_id)
    return ConversationRuntimeInfo(
        runtime_status=ConversationRuntimeStatus.AVAILABLE, can_resume=True
    )


@docker_conversation_router.delete("/{conversation_id}")
async def delete_conversation(conversation_id: UUID, request: Request) -> Response:
    registry = get_registry(request)
    if not registry.provisioning.manifest_path(conversation_id).is_file():
        raise HTTPException(404, "Conversation not found")

    if not await registry.begin_delete(conversation_id):
        raise HTTPException(409, "Conversation deletion is already in progress")

    # Stopping the container closes its conversation service. The outer server
    # owns the bind-mounted state and removes it after Docker has unmounted it;
    # asking the inner server to remove the mount root leaves that root in a
    # partially deleted state.
    try:
        await registry.stop(conversation_id)
        registry.provisioning.manifest_path(conversation_id).unlink(missing_ok=True)
        await asyncio.to_thread(
            safe_rmtree, registry.provisioning.runtime_dir(conversation_id)
        )
        await asyncio.to_thread(safe_rmtree, registry.conversation_dir(conversation_id))
        await get_conversation_service(request).refresh_persisted_conversation(
            conversation_id
        )
    finally:
        await registry.finish_delete(conversation_id)
    return Response(status_code=200)


@docker_conversation_router.api_route(
    "/{conversation_id}",
    methods=["POST", "PUT", "PATCH", "OPTIONS"],
)
async def proxy_conversation_root(
    conversation_id: UUID, request: Request
) -> StreamingResponse:
    return await proxy_conversation(conversation_id, "", request)


@docker_conversation_router.api_route(
    "/{conversation_id}/{tail:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
)
async def proxy_conversation(
    conversation_id: UUID, tail: str, request: Request
) -> StreamingResponse:
    if tail.split("/", 1)[0] in {
        "credential-bindings",
        "fork",
        "switch_llm",
        "switch_profile",
    }:
        raise HTTPException(501, "This operation is unavailable in Docker runtime mode")
    registry = get_registry(request)
    container = await _container(registry, conversation_id)
    upstream_path = _upstream_path(request, request.url.path)
    if tail == "secrets" and request.method == "POST":
        try:
            update = UpdateSecretsRequest.model_validate_json(await request.body())
        except ValueError as exc:
            raise HTTPException(422, "Invalid secrets payload") from exc
        profile = registry.provisioning.load(conversation_id).launched_agent_profile
        secrets = update.secrets
        if profile is not None:
            secrets = {
                name: source
                for name, source in secrets.items()
                if profile.allows_secret(name)
            }
        materialized = await asyncio.to_thread(materialize_secrets, secrets)
        body = UpdateSecretsRequest(secrets=materialized).model_dump(
            mode="json", context={"expose_secrets": "plaintext"}
        )
        response = await _proxy_with_session(
            registry,
            conversation_id,
            container,
            request,
            upstream_path=upstream_path,
            body=json.dumps(body).encode(),
        )
    else:
        response = await _proxy_with_session(
            registry,
            conversation_id,
            container,
            request,
            upstream_path=upstream_path,
        )
    if request.method not in {"GET", "HEAD", "OPTIONS"} and response.status_code < 400:
        await get_conversation_service(request).refresh_persisted_conversation(
            conversation_id
        )
    return response


docker_workspace_router = APIRouter(prefix="/conversations", tags=["Docker Workspace"])


@docker_workspace_router.get("/{conversation_id}/workspace/{file_path:path}")
async def proxy_workspace_file(
    conversation_id: UUID, file_path: str, request: Request
) -> StreamingResponse:
    registry = get_registry(request)
    container = await _container(registry, conversation_id)
    return await _proxy_with_session(
        registry,
        conversation_id,
        container,
        request,
        upstream_path=_upstream_path(
            request, f"/api/conversations/{conversation_id}/workspace/{file_path}"
        ),
    )


docker_sockets_router = APIRouter(prefix="/sockets", tags=["Docker WebSockets"])
docker_session_sockets_router = APIRouter(prefix="/sockets", tags=["Docker WebSockets"])


async def _proxy_socket(
    websocket: WebSocket,
    conversation_id: UUID,
    session_api_key: str | None,
    socket_name: str,
) -> None:
    from openhands.agent_server.sockets import _accept_authenticated_websocket

    if not await _accept_authenticated_websocket(websocket, session_api_key):
        return
    registry = websocket.app.state.conversation_registry
    if not isinstance(registry, DockerConversationRegistry):
        await websocket.close(code=1011)
        return
    try:
        container = await _container(registry, conversation_id)
    except HTTPException as exc:
        await websocket.close(code=1008 if exc.status_code == 404 else 1011)
        return
    query = strip_auth_query("?" + websocket.url.query).lstrip("?")
    path = f"/sockets/{socket_name}/{conversation_id}"
    # Hold the attachment for the whole bridge lifetime so the idle-eviction
    # loop cannot stop the container under a live events/session websocket.
    registry.attach_session(conversation_id)
    try:
        await bridge_websocket(
            websocket,
            container,
            upstream_path=f"{path}?{query}" if query else path,
        )
    finally:
        registry.detach_session(conversation_id)
        await websocket.app.state.conversation_service.refresh_persisted_conversation(
            conversation_id
        )


@docker_sockets_router.websocket("/events/{conversation_id}")
async def proxy_events(
    websocket: WebSocket,
    conversation_id: UUID,
    session_api_key: Annotated[str | None, Query(alias="session_api_key")] = None,
) -> None:
    await _proxy_socket(websocket, conversation_id, session_api_key, "events")


@docker_session_sockets_router.websocket("/session/{conversation_id}")
async def proxy_session(
    websocket: WebSocket,
    conversation_id: UUID,
    session_api_key: Annotated[str | None, Query(alias="session_api_key")] = None,
) -> None:
    await _proxy_socket(websocket, conversation_id, session_api_key, "session")
