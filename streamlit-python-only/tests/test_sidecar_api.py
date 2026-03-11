from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, BaseMessage

from continue_better_py.run_state import RunStateStore
from continue_better_py.runtime import RuntimeDependencies, RuntimeEngine
from continue_better_py.schemas import ChatMessage
from continue_better_py.sidecar_app import create_sidecar_app
from continue_better_py.tool_registry import create_default_tool_registry


class FakeProvider:
    def __init__(self, mode: str) -> None:
        self.mode = mode


class FakeModel:
    def __init__(self, responses: list[AIMessage]) -> None:
        self.responses = responses
        self.invocations: list[list[BaseMessage]] = []

    def bind_tools(self, _tools):
        return self

    def invoke(self, messages: list[BaseMessage]):
        self.invocations.append(messages)
        return self.responses.pop(0)


def build_test_client(tmp_path: Path, responses: list[AIMessage], provider_mode: str = "native") -> TestClient:
    model = FakeModel(responses)
    store = RunStateStore()
    store.root = tmp_path
    tmp_path.joinpath("approvals").mkdir(parents=True, exist_ok=True)
    tmp_path.joinpath("clarifications").mkdir(parents=True, exist_ok=True)
    runtime = RuntimeEngine(
        RuntimeDependencies(
            model_factory=lambda _profile, _model: model,
            provider_resolver=lambda _profile, _model: FakeProvider(provider_mode),
            tool_registry_factory=create_default_tool_registry,
            state_store=store,
        )
    )
    return TestClient(create_sidecar_app(runtime=runtime))


def parse_sse_payloads(text: str) -> list[dict]:
    payloads = []
    for packet in text.split("\n\n"):
        if not packet.startswith("data: "):
            continue
        payloads.append(json.loads(packet[6:]))
    return payloads


def test_sidecar_chat_stream_returns_completion_tokens(tmp_path: Path):
    client = build_test_client(tmp_path, [AIMessage(content="Hello world.")])
    response = client.post(
        "/v1/chat/stream",
        json={
            "sessionId": "s1",
            "messages": [ChatMessage(role="user", content="hi").model_dump()],
            "workspaceRoot": str(tmp_path),
        },
    )
    assert response.status_code == 200
    events = parse_sse_payloads(response.text)
    assert any(event["type"] == "token" for event in events)
    assert any(event["type"] == "run_state" and event["state"] == "completed" for event in events)


def test_sidecar_approval_resume_endpoint_returns_stream(tmp_path: Path):
    client = build_test_client(
        tmp_path,
        [
            AIMessage(content="", tool_calls=[{"id": "call-1", "name": "write_file", "args": {"path": "a.txt", "content": "hello"}}]),
            AIMessage(content="Approved path complete."),
        ],
    )
    first = client.post(
        "/v1/chat/stream",
        json={
            "sessionId": "s1",
            "messages": [ChatMessage(role="user", content="write").model_dump()],
            "workspaceRoot": str(tmp_path),
        },
    )
    approval_events = parse_sse_payloads(first.text)
    approval = next(event for event in approval_events if event["type"] == "approval_required")
    resumed = client.post(
        "/v1/approvals/respond/stream",
        json={"approval_id": approval["actionId"], "decision": "approved"},
    )
    assert resumed.status_code == 200
    resumed_events = parse_sse_payloads(resumed.text)
    assert any(event["type"] == "tool_result" for event in resumed_events)
    assert any(event["type"] == "run_state" and event["state"] == "completed" for event in resumed_events)
