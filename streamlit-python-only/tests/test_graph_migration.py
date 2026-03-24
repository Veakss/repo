from __future__ import annotations

from pathlib import Path

import mongomock
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from streamlit_python_only.mcp_tools import build_mcp_tool_definition
from streamlit_python_only.agent.evaluation.diagnostics import build_goal_gap_diagnostics
from streamlit_python_only.providers import ProviderCapabilities
from streamlit_python_only.rag import RagService
from streamlit_python_only.run_state import RunStateStore
from streamlit_python_only.runtime import RuntimeDependencies
from streamlit_python_only.runtime import RuntimeEngine
from streamlit_python_only.schemas import ChatMessage, SidecarChatRequest
from streamlit_python_only.terminal_manager import TerminalManager
from streamlit_python_only.tool_registry import create_default_tool_registry


class EchoInput(BaseModel):
    value: str = Field(description="Echo value.")


class FakeProvider:
    def __init__(self) -> None:
        self.mode = "native"
        self.capabilities = ProviderCapabilities(
            supports_native_tools=True,
            supports_textual_replay=False,
            supports_multi_tool_turn=True,
            max_tool_calls_per_turn=8,
            requires_sequential_tool_loop=False,
            config_source="test",
            provider_family="openrouter",
        )
        self.provider = "openrouter"


class FakeModel:
    def bind_tools(self, _tools):
        return self

    def invoke(self, _messages):
        raise AssertionError("This test should not invoke the model")


def make_runtime(tmp_path: Path) -> RuntimeEngine:
    store = RunStateStore()
    store.root = tmp_path
    terminals = TerminalManager()
    rag = RagService(
        client=mongomock.MongoClient(),
        database_name=f"streamlit_python_only_graph_migration_{tmp_path.name}",
        artifacts_root=tmp_path,
    )
    deps = RuntimeDependencies(
        model_factory=lambda _profile, _model: FakeModel(),
        provider_resolver=lambda _profile, _model: FakeProvider(),
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
    return RuntimeEngine(dependencies=deps)


def test_mcp_tool_definition_adapts_to_tool_metadata() -> None:
    definition = build_mcp_tool_definition(
        "mcp_echo",
        "Echo a value through an MCP-style adapter.",
        EchoInput,
        lambda value: value,
        module_id="mcp_bridge",
        risk_level="safe",
        supports_thales=False,
        returns_evidence_kind="bridge",
    )

    assert definition.tool.name == "mcp_echo"
    assert definition.module_id == "mcp_bridge"
    assert definition.risk_level == "safe"
    assert definition.supports_thales is False
    assert definition.returns_evidence_kind == "bridge"


def test_preflight_populates_graph_state_slices(tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path)
    request = SidecarChatRequest(sessionId="s1", workspaceRoot=str(tmp_path), messages=[ChatMessage(role="user", content="Read foo.txt and summarize it")])
    state = runtime._initial_state(request, "run-1", [HumanMessage(content="Read foo.txt and summarize it")])
    updated = runtime._preflight_node(state)

    assert updated["conversation"]["runId"] == "run-1"
    assert updated["runtime"]["workspaceRoot"] == str(tmp_path)
    assert "summary" in updated["goal"]
    assert updated["procedure_state"]["currentStepIndex"] == 0
    assert updated["procedure_state"]["phase"] == "completed"
    assert updated["procedure_state"]["completedTools"] == []
    assert "gap" in updated["evidence"]
    assert updated["decision"]["graphRoute"] is None
    assert updated["output"]["candidateFinal"] == ""


def test_runtime_uses_dedicated_graph_runner(tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path)
    request = SidecarChatRequest(sessionId="s1", workspaceRoot=str(tmp_path), messages=[ChatMessage(role="user", content="Read foo.txt and summarize it")])
    state = runtime._initial_state(request, "run-2", [HumanMessage(content="Read foo.txt and summarize it")])

    assert runtime.graph_runner is not runtime
    assert runtime.graph is runtime.graph_runner.graph

    updated = runtime.graph_runner._preflight_node(state)

    assert updated["conversation"]["runId"] == "run-2"
    assert updated["runtime"]["workspaceRoot"] == str(tmp_path)


def test_graph_runner_builds_canonical_textual_replay_message(tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path)
    message = runtime.graph_runner._build_canonical_tool_replay_message(
        tool_name="read_file",
        tool_args={"path": "foo.txt"},
        result_text="hello world",
        status="succeeded",
        task="Read foo.txt and summarize it.",
        done=["- read_file succeeded"],
        remaining=["- Answer briefly in the user language."],
    )

    content = str(message.content)
    assert "TASK" in content
    assert "LAST_ACTION" in content
    assert "OBSERVATION" in content
    assert "DONE" in content
    assert "REMAINING" in content
    assert "NEXT_ACTION_RULE" in content


def test_graph_runner_prepares_initial_clarification_state(tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path)
    request = SidecarChatRequest(
        sessionId="s1",
        workspaceRoot=str(tmp_path),
        messages=[ChatMessage(role="user", content="Help me create a food application.")],
        toolToggles={"clarification": True},
    )
    state = runtime._initial_state(request, "run-3", [HumanMessage(content="Help me create a food application.")])
    updated = runtime.graph_runner.prepare_initial_chat_state(state)

    assert updated["pending_clarification"] is not None
    assert updated["tool_events"]
    assert updated["procedure_state"]["blocker"] == "clarification"


def test_graph_runner_no_longer_depends_on_legacy_mixin(tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path)
    legacy_path = Path("/Users/victor/Documents/continue-better/streamlit-python-only/src/streamlit_python_only/runtime_graph_orchestration.py")

    assert runtime.graph_runner.__class__.__name__ == "AgentGraphRunner"
    assert not legacy_path.exists()


def test_goal_gap_diagnostics_use_evaluation_engine(tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path)
    request = SidecarChatRequest(sessionId="s1", workspaceRoot=str(tmp_path), messages=[ChatMessage(role="user", content="Read foo.txt and summarize it")])
    state = runtime._initial_state(request, "run-4", [HumanMessage(content="Read foo.txt and summarize it")])
    runtime._refresh_goal_tracking(state, "")

    diagnostics = build_goal_gap_diagnostics(runtime, "run-4", state, "test")

    assert [item["code"] for item in diagnostics] == ["goal_summary", "realization_summary", "goal_gap_summary"]
