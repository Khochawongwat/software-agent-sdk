import json
import sys
from unittest.mock import AsyncMock
from uuid import UUID

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr

from openhands.agent_server.config import Config
from openhands.agent_server.docker_runtime.registry import DockerConversationRegistry
from openhands.agent_server.docker_runtime.routers import (
    WorkspaceCommandError,
    docker_conversation_router,
    prepare_workspace,
)


CONVERSATION = UUID("9e3a59c5-e149-4758-8aec-2833e585f8c2")


def python(code: str) -> list[str]:
    return [sys.executable, "-c", code]


def test_the_printed_directory_becomes_the_workspace(tmp_path):
    made = tmp_path / "made"
    record = tmp_path / "record.json"
    command = python(
        "import json, os, pathlib, sys;"
        f"pathlib.Path({str(made)!r}).mkdir();"
        f"pathlib.Path({str(record)!r}).write_text(json.dumps("
        "[sys.argv[1:], json.loads(os.environ['OH_CONVERSATION_TAGS'])]));"
        f"print('making a copy'); print({str(made)!r})"
    )

    workspace = prepare_workspace(
        command, tmp_path / "repo", CONVERSATION, {"kind": "latest"}
    )

    assert workspace == made
    assert json.loads(record.read_text()) == [
        [str(tmp_path / "repo"), str(CONVERSATION)],
        {"kind": "latest"},
    ]


def test_a_failing_command_is_refused_with_its_message(tmp_path):
    command = python("import sys; sys.exit('no such repo')")

    with pytest.raises(WorkspaceCommandError) as caught:
        prepare_workspace(command, tmp_path, CONVERSATION, None)

    assert str(caught.value) == "Workspace command failed: no such repo"


def test_output_that_is_not_an_existing_directory_is_refused(tmp_path):
    command = python(f"print({str(tmp_path / 'missing')!r})")

    with pytest.raises(WorkspaceCommandError) as caught:
        prepare_workspace(command, tmp_path, CONVERSATION, None)

    assert str(caught.value) == (
        "Workspace command did not print an existing absolute directory"
    )


def test_a_failed_workspace_command_rejects_the_start_before_any_runtime(tmp_path):
    config = Config(
        conversations_path=tmp_path / "conversations",
        secret_key=SecretStr("outer-key"),
        conversation_workspace_command=python("import sys; sys.exit('clone failed')"),
    )
    app = FastAPI()
    registry = DockerConversationRegistry(config)
    app.state.conversation_registry = registry
    app.state.conversation_service = AsyncMock()
    app.include_router(docker_conversation_router, prefix="/api")

    with TestClient(app) as client:
        response = client.post(
            "/api/conversations",
            json={
                "conversation_id": str(CONVERSATION),
                "workspace": {"working_dir": str(tmp_path)},
            },
        )

    assert (response.status_code, response.json()) == (
        502,
        {"detail": "Workspace command failed: clone failed"},
    )
    assert registry.provisioning.manifest_path(CONVERSATION).exists() is False


def test_an_existing_conversation_keeps_its_workspace_without_rerunning(
    tmp_path, monkeypatch
):
    ran = tmp_path / "ran"
    config = Config(
        conversations_path=tmp_path / "conversations",
        secret_key=SecretStr("outer-key"),
        conversation_workspace_command=python(
            f"import pathlib; pathlib.Path({str(ran)!r}).touch()"
        ),
    )
    app = FastAPI()
    registry = DockerConversationRegistry(config)
    first = tmp_path / "first"
    registry.provisioning.create(CONVERSATION, first)
    app.state.conversation_registry = registry
    app.state.conversation_service = AsyncMock()
    app.include_router(docker_conversation_router, prefix="/api")

    async def stop_at_prepare(_body, _config):
        raise ValueError("stopped here")

    monkeypatch.setattr(
        "openhands.agent_server.docker_runtime.routers.prepare_start", stop_at_prepare
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/conversations",
            json={
                "conversation_id": str(CONVERSATION),
                "workspace": {"working_dir": str(tmp_path / "repo")},
            },
        )

    assert response.status_code == 422
    assert ran.exists() is False
    assert registry.provisioning.load(CONVERSATION).workspace_path == first.resolve()
