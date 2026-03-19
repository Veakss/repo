from __future__ import annotations

import json
from pathlib import Path
import time

import mongomock
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, BaseMessage

from streamlit_python_only.providers import ProviderCapabilities
from streamlit_python_only.rag import RagService
from streamlit_python_only.run_state import RunStateStore
from streamlit_python_only.runtime import RuntimeDependencies, RuntimeEngine
from streamlit_python_only.schemas import ChatMessage
from streamlit_python_only.sidecar_app import create_sidecar_app
from streamlit_python_only.terminal_manager import TerminalManager
from streamlit_python_only.tool_registry import create_default_tool_registry


class FakeProvider:
    def __init__(self, mode: str, capabilities: ProviderCapabilities | None = None) -> None:
        self.mode = mode
        self.capabilities = capabilities
        self.provider = capabilities.provider_family if capabilities else ("thales" if mode == "textual_replay" else "openrouter")
        self.base_url = "https://api.corp.thales/corp/genai-llm-small/v1" if self.provider == "thales" else "https://openrouter.ai/api/v1"
        self.api_key = "test-key"
        self.model = "mistral" if self.provider == "thales" else "test-model"


class FakeModel:
    def __init__(self, responses: list[AIMessage]) -> None:
        self.responses = responses
        self.invocations: list[list[BaseMessage]] = []

    def bind_tools(self, _tools):
        return self

    def invoke(self, messages: list[BaseMessage]):
        self.invocations.append(messages)
        return self.responses.pop(0)


def build_test_client(
    tmp_path: Path,
    responses: list[AIMessage],
    provider_mode: str = "native",
    provider_capabilities: ProviderCapabilities | None = None,
    rag_service: RagService | None = None,
) -> TestClient:
    model = FakeModel(responses)
    store = RunStateStore()
    terminals = TerminalManager()
    rag = rag_service or RagService(
        client=mongomock.MongoClient(),
        database_name=f"streamlit_python_only_sidecar_rag_{tmp_path.name}",
        artifacts_root=tmp_path,
    )
    store.root = tmp_path
    tmp_path.joinpath("approvals").mkdir(parents=True, exist_ok=True)
    tmp_path.joinpath("clarifications").mkdir(parents=True, exist_ok=True)
    runtime = RuntimeEngine(
        RuntimeDependencies(
            model_factory=lambda _profile, _model: model,
            provider_resolver=lambda _profile, _model: FakeProvider(provider_mode, provider_capabilities),
            tool_registry_factory=lambda workspace_root, session_id, run_id, tool_toggles=None: create_default_tool_registry(
                workspace_root,
                session_id=session_id,
                run_id=run_id,
                terminal_manager=terminals,
                rag_service=rag,
                tool_toggles=tool_toggles,
            ),
            state_store=store,
            terminal_manager=terminals,
            rag_service=rag,
        )
    )
    return TestClient(create_sidecar_app(runtime=runtime, terminal_manager=terminals, rag_service=rag))


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
            "messages": [ChatMessage(role="user", content="write a.txt with hello").model_dump()],
            "workspaceRoot": str(tmp_path),
            "toolToggles": {"clarification": False},
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


def test_sidecar_terminal_endpoints_support_interactive_session(tmp_path: Path):
    client = build_test_client(tmp_path, [AIMessage(content="unused")])
    created = client.post("/v1/terminals", json={"workspace_root": str(tmp_path), "session_id": "s1", "run_id": "run-sidecar-1", "owner": "user"})
    assert created.status_code == 200
    terminal = created.json()["terminal"]
    terminal_id = terminal["terminalId"]

    wrote = client.post(f"/v1/terminals/{terminal_id}/write", json={"data": "echo sidecar-terminal\n", "source": "user"})
    assert wrote.status_code == 200
    assert wrote.json()["events"][0]["type"] == "terminal_input"
    snapshot = {}
    for _ in range(20):
        snapshot = client.get(f"/v1/terminals/{terminal_id}").json()["terminal"]
        if "sidecar-terminal" in snapshot.get("tail", ""):
            break
        time.sleep(0.05)
    assert "sidecar-terminal" in snapshot.get("tail", "")

    listed = client.get("/v1/terminals", params={"run_id": "run-sidecar-1", "alive_only": "true"})
    assert listed.status_code == 200
    assert listed.json()["terminals"][0]["terminalId"] == terminal_id
    assert listed.json()["activeTerminalId"] == terminal_id

    closed = client.post(f"/v1/terminals/{terminal_id}/close")
    assert closed.status_code == 200
    assert any(event["type"] == "terminal_closed" for event in closed.json()["events"])


def test_sidecar_can_close_run_terminals(tmp_path: Path):
    client = build_test_client(tmp_path, [AIMessage(content="unused")])
    first = client.post("/v1/terminals", json={"workspace_root": str(tmp_path), "session_id": "s1", "run_id": "run-close-sidecar", "owner": "agent"})
    second = client.post("/v1/terminals", json={"workspace_root": str(tmp_path), "session_id": "s1", "run_id": "run-close-sidecar", "owner": "user"})
    assert first.status_code == 200
    assert second.status_code == 200

    closed = client.post("/v1/runs/run-close-sidecar/terminals/close", params={"reason": "cleanup"})
    assert closed.status_code == 200
    payload = closed.json()
    assert len(payload["terminals"]) == 2
    assert any(event["type"] == "terminal_closed" for event in payload["events"])


def test_sidecar_terminal_snapshot_wait_and_resolve(tmp_path: Path):
    client = build_test_client(tmp_path, [AIMessage(content="unused")])
    created = client.post("/v1/terminals", json={"workspace_root": str(tmp_path), "session_id": "s1", "run_id": "run-resolve-1", "owner": "agent"})
    terminal_id = created.json()["terminal"]["terminalId"]

    resolved = client.post("/v1/terminals/resolve", json={"run_id": "run-resolve-1", "session_id": "s1"})
    assert resolved.status_code == 200
    assert resolved.json()["terminalId"] == terminal_id

    client.post(f"/v1/terminals/{terminal_id}/write", json={"data": "echo sidecar-snapshot\n", "source": "agent"})
    waited = client.post(f"/v1/terminals/{terminal_id}/wait_for_output", json={"pattern": "sidecar-snapshot", "timeout_ms": 5000, "regex": False})
    assert waited.status_code == 200
    assert waited.json()["matched"] is True

    snapshot = client.get(f"/v1/terminals/{terminal_id}/snapshot")
    assert snapshot.status_code == 200
    assert "sidecar-snapshot" in snapshot.json()["snapshot"]["tail"]
    assert snapshot.json()["terminal"]["terminalId"] == terminal_id


def test_sidecar_capabilities_and_models_expose_provider_capabilities(tmp_path: Path):
    client = build_test_client(
        tmp_path,
        [AIMessage(content="unused")],
        provider_mode="textual_replay",
        provider_capabilities=ProviderCapabilities(
            supports_native_tools=True,
            supports_textual_replay=True,
            supports_multi_tool_turn=False,
            max_tool_calls_per_turn=1,
            requires_sequential_tool_loop=True,
            config_source="enterprise_yaml",
            provider_family="thales",
            config_path=str(tmp_path / "tools" / "config.yaml"),
        ),
    )
    capabilities = client.get("/v1/capabilities")
    assert capabilities.status_code == 200
    assert "providerCapabilities" in capabilities.json()
    assert "maxToolCallsPerTurn" in capabilities.json()["providerCapabilities"]
    assert capabilities.json()["providerCapabilities"]["requiresSequentialToolLoop"] is True
    assert capabilities.json()["providerCapabilities"]["providerFamily"] == "thales"
    assert capabilities.json()["providerCapabilities"]["configSource"] == "enterprise_yaml"

    models = client.get("/v1/models")
    assert models.status_code == 200
    assert "providerCapabilities" in models.json()
    assert "supportsTextualReplay" in models.json()["providerCapabilities"]


def test_sidecar_rag_lookup_exposes_followup_meta(tmp_path: Path):
    rag_service = RagService(
        client=mongomock.MongoClient(),
        database_name=f"streamlit_python_only_sidecar_rag_lookup_{tmp_path.name}",
        artifacts_root=tmp_path,
    )
    source = tmp_path.joinpath("roadmap_status.md")
    source.write_text(
        "# Matrix Fixture\nCurrent checkpoint: AURORA_PHASE4\nPhase 4 status: very advanced and close to closure.\n",
        encoding="utf-8",
    )
    rag_service.import_file("session", "s1", str(source), "matrix/roadmap_status.md")
    rag_service.enqueue_index_job("session", "s1")
    client = build_test_client(tmp_path, [AIMessage(content="unused")], rag_service=rag_service)

    first = client.post("/v1/rag/lookup", json={"question": "What checkpoint is recorded in the indexed document?", "session_id": "s1"})
    assert first.status_code == 200
    assert first.json()["meta"]["strongHitCount"] >= 1

    second = client.post("/v1/rag/lookup", json={"question": "And what is the Phase 4 status?", "session_id": "s1"})
    assert second.status_code == 200
    payload = second.json()
    assert payload["meta"]["followupContextUsed"] is True
    assert "matrix/roadmap_status.md" in payload["meta"]["topDocumentPaths"]


def test_sidecar_chat_stream_surfaces_final_contract_diagnostics(tmp_path: Path):
    client = build_test_client(
        tmp_path,
        [
            AIMessage(content="", tool_calls=[{"id": "call-1", "name": "run_terminal", "args": {"command": "printf MATRIX_TERMINAL_OK"}}]),
            AIMessage(content="Done."),
            AIMessage(content="MATRIX_TERMINAL_OK"),
        ],
        provider_mode="native",
    )
    response = client.post(
        "/v1/chat/stream",
        json={
            "sessionId": "s1",
            "messages": [ChatMessage(role="user", content="Use the terminal to print MATRIX_TERMINAL_OK, then reply with exactly MATRIX_TERMINAL_OK.").model_dump()],
            "workspaceRoot": str(tmp_path),
            "policyProfile": "always_allow",
        },
    )
    assert response.status_code == 200
    events = parse_sse_payloads(response.text)
    assert any(event["type"] == "run_diagnostic" and event["code"] == "final_contract_detected" for event in events)
    assert any(event["type"] == "run_diagnostic" and event["code"] == "final_contract_repair_requested" for event in events)


def test_sidecar_chat_stream_surfaces_goal_gap_diagnostics(tmp_path: Path):
    fixtures = tmp_path / "matrix_fixtures"
    fixtures.mkdir(parents=True, exist_ok=True)
    fixtures.joinpath("project_stack.md").write_text("Stack: Python, Streamlit, FastAPI\n", encoding="utf-8")
    fixtures.joinpath("product_identity.md").write_text("Product name: AI Technical Assistant\n", encoding="utf-8")
    client = build_test_client(
        tmp_path,
        [
            AIMessage(content="", tool_calls=[{"id": "call-read-1", "name": "read_file", "args": {"path": "matrix_fixtures/project_stack.md"}}]),
            AIMessage(content="The stack is Python and Streamlit."),
            AIMessage(content="AI Technical Assistant uses Python and Streamlit."),
        ],
        provider_mode="native",
    )
    response = client.post(
        "/v1/chat/stream",
        json={
            "sessionId": "s1",
            "messages": [ChatMessage(role="user", content="Read matrix_fixtures/project_stack.md, then read matrix_fixtures/product_identity.md, then answer with the product name and stack.").model_dump()],
            "workspaceRoot": str(tmp_path),
            "policyProfile": "always_allow",
        },
    )
    assert response.status_code == 200
    events = parse_sse_payloads(response.text)
    goal_gap_events = [event for event in events if event["type"] == "run_diagnostic" and event["code"] == "goal_gap_summary"]
    assert goal_gap_events
    assert all("missingEvidence" in event["data"] for event in goal_gap_events)


def test_sidecar_chat_stream_surfaces_rag_diagnostics(tmp_path: Path):
    rag_service = RagService(
        client=mongomock.MongoClient(),
        database_name=f"streamlit_python_only_sidecar_rag_diag_{tmp_path.name}",
        artifacts_root=tmp_path,
    )
    source = tmp_path.joinpath("notes.md")
    source.write_text("This note mentions only apples.", encoding="utf-8")
    rag_service.import_file("session", "s1", str(source), "docs/notes.md")
    rag_service.enqueue_index_job("session", "s1")
    client = build_test_client(
        tmp_path,
        [
            AIMessage(content="", tool_calls=[{"id": "call-rag", "name": "rag_lookup", "args": {"question": "What is the Phase 4 status?"}}]),
            AIMessage(content="Phase 4 status is approved."),
            AIMessage(content=""),
        ],
        provider_mode="native",
        rag_service=rag_service,
    )
    response = client.post(
        "/v1/chat/stream",
        json={
            "sessionId": "s1",
            "messages": [ChatMessage(role="user", content="With the RAG for this session, what is the Phase 4 status? Answer very briefly.").model_dump()],
            "workspaceRoot": str(tmp_path),
            "policyProfile": "always_allow",
            "forceToolUse": "rag",
        },
    )
    assert response.status_code == 200
    events = parse_sse_payloads(response.text)
    assert any(event["type"] == "run_diagnostic" and event["code"] == "rag_lookup_no_strong_hits" for event in events)
    assert any(event["type"] == "run_diagnostic" and event["code"] == "rag_grounding_weak" for event in events)
