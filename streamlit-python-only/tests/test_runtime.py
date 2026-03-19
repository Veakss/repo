from __future__ import annotations

from pathlib import Path
import time

import mongomock
import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from streamlit_python_only.providers import ProviderCapabilities
from streamlit_python_only.rag import RagService
from streamlit_python_only.run_state import RunStateStore
from streamlit_python_only.runtime import RuntimeDependencies, RuntimeEngine, build_multi_step_contract, build_runtime_system_prompt, detect_exact_output_target
from streamlit_python_only.schemas import ApprovalDecisionRequest, ClarificationDecisionRequest, ChatMessage, SidecarChatRequest
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
    def __init__(self, responses: list[AIMessage], delay_sec: float = 0.0) -> None:
        self.responses = responses
        self.delay_sec = delay_sec
        self.invocations: list[list[BaseMessage]] = []
        self.bound_tool_names: list[str] = []

    def bind_tools(self, tools):
        self.bound_tool_names = [tool.name for tool in tools]
        return self

    def invoke(self, messages: list[BaseMessage]):
        self.invocations.append(messages)
        if self.delay_sec > 0:
            time.sleep(self.delay_sec)
        if not self.responses:
            raise AssertionError("No fake response left")
        return self.responses.pop(0)


class ClarificationAwareModel:
    def __init__(self) -> None:
        self.invocations: list[list[BaseMessage]] = []
        self.bound_tool_names: list[str] = []
        self._asked_once = False

    def bind_tools(self, tools):
        self.bound_tool_names = [tool.name for tool in tools]
        return self

    def invoke(self, messages: list[BaseMessage]):
        self.invocations.append(messages)
        if not self._asked_once:
            self._asked_once = True
            return AIMessage(
                content="",
                tool_calls=[{"id": "call-1", "name": "request_clarification", "args": {"question": "Which file should I edit?", "option_a": "app.py", "option_b": "main.py"}}],
            )
        has_resume_context = any(
            isinstance(message, SystemMessage)
            and "The clarification has been answered." in str(message.content)
            and "User clarification answer: Use app.py" in str(message.content)
            for message in messages
        )
        if not has_resume_context:
            return AIMessage(
                content="",
                tool_calls=[{"id": "call-2", "name": "request_clarification", "args": {"question": "I still need the target file."}}],
            )
        return AIMessage(content="I will update app.py.")


@pytest.fixture
def anyio_backend():
    return "asyncio"


def build_rag_service(tmp_path: Path) -> RagService:
    return RagService(
        client=mongomock.MongoClient(),
        database_name=f"streamlit_python_only_runtime_rag_{tmp_path.name}",
        artifacts_root=tmp_path,
    )


def make_runtime(
    fake_model: FakeModel,
    tmp_path: Path,
    provider_mode: str = "native",
    provider_capabilities: ProviderCapabilities | None = None,
    rag_service: RagService | None = None,
) -> RuntimeEngine:
    store = RunStateStore()
    terminals = TerminalManager()
    rag = rag_service or build_rag_service(tmp_path)
    store.root = tmp_path
    tmp_path.joinpath("approvals").mkdir(parents=True, exist_ok=True)
    tmp_path.joinpath("clarifications").mkdir(parents=True, exist_ok=True)
    deps = RuntimeDependencies(
        model_factory=lambda _profile, _model: fake_model,
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
    return RuntimeEngine(dependencies=deps)


def _collect_terminal_events(subscription, expected_types: set[str], timeout_sec: float = 5.0) -> list[dict]:
    deadline = time.time() + timeout_sec
    events: list[dict] = []
    while time.time() < deadline and expected_types.difference({event.get("type") for event in events}):
        remaining = max(0.05, deadline - time.time())
        try:
            events.append(subscription.get(timeout=min(0.5, remaining)))
        except Exception:
            pass
    return events


@pytest.mark.anyio
async def test_runtime_emits_approval_for_risky_write(tmp_path: Path):
    model = FakeModel([AIMessage(content="", tool_calls=[{"id": "call-1", "name": "write_file", "args": {"path": "a.txt", "content": "hello"}}])])
    runtime = make_runtime(model, tmp_path)
    request = SidecarChatRequest(sessionId="s1", messages=[ChatMessage(role="user", content="write a file")], workspaceRoot=str(tmp_path))
    events = [event async for event in runtime.stream_chat(request, [AIMessage(content="ignored")])]
    assert any(event["type"] == "approval_required" for event in events)
    assert any(event["type"] == "run_state" and event["state"] == "awaiting_approval" for event in events)


@pytest.mark.anyio
async def test_runtime_emits_clarification_request(tmp_path: Path):
    model = FakeModel([AIMessage(content="", tool_calls=[{"id": "call-1", "name": "request_clarification", "args": {"question": "Which filename?", "option_a": "main.py", "option_b": "app.py"}}])])
    runtime = make_runtime(model, tmp_path)
    request = SidecarChatRequest(sessionId="s1", messages=[ChatMessage(role="user", content="clarify")], workspaceRoot=str(tmp_path))
    events = [event async for event in runtime.stream_chat(request, [AIMessage(content="ignored")])]
    clarification = next(event for event in events if event["type"] == "clarification_required")
    assert clarification["question"] == "Which filename?"
    assert len(clarification["options"]) == 2


@pytest.mark.anyio
async def test_approval_resume_uses_tool_message_in_native_mode(tmp_path: Path):
    model = FakeModel(
        [
            AIMessage(content="", tool_calls=[{"id": "call-1", "name": "write_file", "args": {"path": "a.txt", "content": "hello"}}]),
            AIMessage(content="Done after approval."),
        ]
    )
    runtime = make_runtime(model, tmp_path, provider_mode="native")
    request = SidecarChatRequest(sessionId="s1", messages=[ChatMessage(role="user", content="write a file")], workspaceRoot=str(tmp_path))
    chat_events = [event async for event in runtime.stream_chat(request, [AIMessage(content="ignored")])]
    approval_event = next(event for event in chat_events if event["type"] == "approval_required")
    resume_events = [event async for event in runtime.stream_approval_decision(ApprovalDecisionRequest(approval_id=approval_event["actionId"], decision="approved"))]
    assert any(event["type"] == "tool_result" for event in resume_events)
    assert any(event["type"] == "token" for event in resume_events)
    assert any(isinstance(message, ToolMessage) for message in model.invocations[-1])


@pytest.mark.anyio
async def test_approval_resume_uses_textual_replay_for_thales_mode(tmp_path: Path):
    model = FakeModel(
        [
            AIMessage(content="", tool_calls=[{"id": "call-1", "name": "write_file", "args": {"path": "a.txt", "content": "hello"}}]),
            AIMessage(content="Done after textual replay."),
        ]
    )
    runtime = make_runtime(model, tmp_path, provider_mode="textual_replay")
    request = SidecarChatRequest(sessionId="s1", messages=[ChatMessage(role="user", content="write a file")], workspaceRoot=str(tmp_path))
    chat_events = [event async for event in runtime.stream_chat(request, [AIMessage(content="ignored")])]
    approval_event = next(event for event in chat_events if event["type"] == "approval_required")
    _ = [event async for event in runtime.stream_approval_decision(ApprovalDecisionRequest(approval_id=approval_event["actionId"], decision="approved"))]
    assert any(isinstance(message, SystemMessage) and "Tool execution replay" in message.content for message in model.invocations[-1])


@pytest.mark.anyio
async def test_clarification_resume_completes_run(tmp_path: Path):
    model = FakeModel(
        [
            AIMessage(content="", tool_calls=[{"id": "call-1", "name": "request_clarification", "args": {"question": "Need scope?"}}]),
            AIMessage(content="Clarified answer."),
        ]
    )
    runtime = make_runtime(model, tmp_path, provider_mode="native")
    request = SidecarChatRequest(sessionId="s1", messages=[ChatMessage(role="user", content="clarify")], workspaceRoot=str(tmp_path))
    chat_events = [event async for event in runtime.stream_chat(request, [AIMessage(content="ignored")])]
    clarification_event = next(event for event in chat_events if event["type"] == "clarification_required")
    resume_events = [event async for event in runtime.stream_clarification_decision(ClarificationDecisionRequest(clarification_id=clarification_event["clarificationId"], answer="Use app.py"))]
    assert any(event["type"] == "token" for event in resume_events)
    assert any(event["type"] == "run_state" and event["state"] == "completed" for event in resume_events)


@pytest.mark.anyio
async def test_clarification_resume_adds_context_and_avoids_repeat_request(tmp_path: Path):
    model = ClarificationAwareModel()
    runtime = make_runtime(model, tmp_path, provider_mode="native")
    request = SidecarChatRequest(sessionId="s1", messages=[ChatMessage(role="user", content="Please edit the right file")], workspaceRoot=str(tmp_path))
    chat_events = [event async for event in runtime.stream_chat(request, [AIMessage(content="ignored")])]
    clarification_event = next(event for event in chat_events if event["type"] == "clarification_required")
    resume_events = [
        event
        async for event in runtime.stream_clarification_decision(
            ClarificationDecisionRequest(clarification_id=clarification_event["clarificationId"], answer="Use app.py")
        )
    ]
    assert any(event["type"] == "run_state" and event["state"] == "completed" for event in resume_events)
    assert not any(event["type"] == "clarification_required" for event in resume_events)
    resumed_messages = model.invocations[-1]
    assert any(
        isinstance(message, SystemMessage)
        and "The clarification has been answered." in str(message.content)
        and "Original clarification question: Which file should I edit?" in str(message.content)
        and "User clarification answer: Use app.py" in str(message.content)
        for message in resumed_messages
    )


@pytest.mark.anyio
async def test_clarification_resume_can_trigger_second_structured_clarification_for_broad_answer(tmp_path: Path):
    model = FakeModel(
        [
            AIMessage(content="", tool_calls=[{"id": "call-1", "name": "request_clarification", "args": {"question": "What kind of app do you want exactly?"}}]),
        ]
    )
    runtime = make_runtime(model, tmp_path, provider_mode="native")
    request = SidecarChatRequest(
        sessionId="s1",
        messages=[ChatMessage(role="user", content="Help me create a food application.")],
        workspaceRoot=str(tmp_path),
    )
    chat_events = [event async for event in runtime.stream_chat(request, [AIMessage(content="ignored")])]
    clarification_event = next(event for event in chat_events if event["type"] == "clarification_required")
    resume_events = [
        event
        async for event in runtime.stream_clarification_decision(
            ClarificationDecisionRequest(clarification_id=clarification_event["clarificationId"], answer="A mobile app.")
        )
    ]
    followup_clarification = next(event for event in resume_events if event["type"] == "clarification_required")
    assert followup_clarification["question"] == "Which mobile platform do you want exactly: iPhone, Android, or cross-platform?"
    assert any(event["type"] == "run_state" and event["state"] == "awaiting_clarification" for event in resume_events)
    assert not any(event["type"] == "token" for event in resume_events)


def test_detect_exact_output_target_handles_terminal_variants_generally():
    assert detect_exact_output_target(
        "Use the terminal to print MATRIX_TERMINAL_OK, then reply with exactly MATRIX_TERMINAL_OK."
    ) == "MATRIX_TERMINAL_OK"
    assert detect_exact_output_target(
        "Run a terminal command that prints MATRIX_VARIANT_OK and return exactly MATRIX_VARIANT_OK."
    ) == "MATRIX_VARIANT_OK"
    assert detect_exact_output_target(
        "Use the terminal to print MATRIX_ONLY_OK, then answer only MATRIX_ONLY_OK."
    ) == "MATRIX_ONLY_OK"
    assert detect_exact_output_target(
        "Run a terminal command that prints MATRIX_STRICT_OK. After that, say exactly MATRIX_STRICT_OK with no extra words."
    ) == "MATRIX_STRICT_OK"
    assert detect_exact_output_target(
        "Print MATRIX_RETURN_OK in the terminal, then return exactly MATRIX_RETURN_OK."
    ) == "MATRIX_RETURN_OK"
    assert detect_exact_output_target("Say exactly MESSAGE_ORDER_OK.") == "MESSAGE_ORDER_OK."


def test_build_multi_step_contract_detects_inputs_and_answer_fields_generally():
    contract = build_multi_step_contract(
        "Read matrix_fixtures/project_stack.md and matrix_fixtures/product_identity.md, then inspect the workspace root and answer with the product name and stack."
    )
    assert contract.expected_step_count >= 3
    assert {"kind": "file", "target": "matrix_fixtures/project_stack.md"} in contract.required_inputs
    assert {"kind": "file", "target": "matrix_fixtures/product_identity.md"} in contract.required_inputs
    assert {"kind": "directory", "target": "."} in contract.required_inputs
    assert "product_name" in contract.required_answer_fields
    assert "stack" in contract.required_answer_fields


def test_runtime_system_prompt_stays_minimal_for_multistep(tmp_path: Path):
    class _Registry:
        def enabled_tool_names(self) -> list[str]:
            return ["read_file", "write_file", "run_terminal", "list_directory"]

    registry = _Registry()
    prompt = build_runtime_system_prompt(
        registry,
        selected_group="mini_project_code_loop_ordered",
        provider_capabilities={"requiresSequentialToolLoop": True},
    ).lower()
    assert "continue only while verified evidence is still missing" in prompt
    assert "stop immediately once the goal gap is closed" in prompt
    assert "do not stop after the first partial result" not in prompt
    assert "after each tool result, either produce the final answer or emit exactly one next tool call if the goal gap is still open" in prompt


@pytest.mark.anyio
async def test_runtime_multistep_repairs_completed_two_step_task_from_evidence_map(tmp_path: Path):
    fixtures = tmp_path / "matrix_fixtures"
    fixtures.mkdir(parents=True, exist_ok=True)
    fixtures.joinpath("project_stack.md").write_text("Stack: Python, Streamlit, FastAPI\n", encoding="utf-8")
    fixtures.joinpath("product_identity.md").write_text("Product name: AI Technical Assistant\n", encoding="utf-8")

    model = FakeModel(
        [
            AIMessage(content="", tool_calls=[{"id": "call-read-1", "name": "read_file", "args": {"path": "matrix_fixtures/project_stack.md"}}]),
            AIMessage(content="", tool_calls=[{"id": "call-read-2", "name": "read_file", "args": {"path": "matrix_fixtures/product_identity.md"}}]),
            AIMessage(content="AI Technical Assistant uses Python and Streamlit."),
        ]
    )
    runtime = make_runtime(model, tmp_path, provider_mode="native")
    request = SidecarChatRequest(
        sessionId="s1",
        messages=[
            ChatMessage(
                role="user",
                content="Read matrix_fixtures/project_stack.md and matrix_fixtures/product_identity.md, then answer very briefly with the product name and stack.",
            )
        ],
        workspaceRoot=str(tmp_path),
        policyProfile="always_allow",
    )
    events = [event async for event in runtime.stream_chat(request, [HumanMessage(content=request.messages[0].content)])]
    tool_calls = [event for event in events if event.get("type") == "tool_call"]
    assert any(event["type"] == "run_diagnostic" and event["code"] == "multistep_contract_detected" for event in events)
    assert [event["name"] for event in tool_calls[:2]] == ["read_file", "read_file"]
    assert len(model.invocations) == 3
    progress_event = next(event for event in events if event["type"] == "run_diagnostic" and event["code"] == "multistep_progress_snapshot")
    assert progress_event["data"]["requiredInputCount"] == 2
    assert progress_event["data"]["completionRatio"] >= 1.0
    goal_gap = next(event for event in events if event["type"] == "run_diagnostic" and event["code"] == "goal_gap_summary")
    assert goal_gap["data"]["isComplete"] is True
    assert goal_gap["data"]["missingEvidence"] == []
    assert goal_gap["data"]["missingFacts"] == []
    final_text = "".join(str(event.get("token") or "") for event in events if event.get("type") == "token").strip()
    lowered = final_text.lower()
    assert "ai technical assistant" in lowered
    assert "python" in lowered or "streamlit" in lowered
    assert any(event["type"] == "run_diagnostic" and event["code"] == "finished_after_sufficient_evidence" for event in events)
    assert any(event["type"] == "run_diagnostic" and event["code"] == "stopped_because_gap_closed" for event in events)


@pytest.mark.anyio
async def test_runtime_multistep_continues_until_missing_evidence_is_collected(tmp_path: Path):
    fixtures = tmp_path / "matrix_fixtures"
    fixtures.mkdir(parents=True, exist_ok=True)
    fixtures.joinpath("project_stack.md").write_text("Stack: Python, Streamlit, FastAPI\n", encoding="utf-8")
    fixtures.joinpath("product_identity.md").write_text("Product name: AI Technical Assistant\n", encoding="utf-8")

    model = FakeModel(
        [
            AIMessage(content="", tool_calls=[{"id": "call-read-1", "name": "read_file", "args": {"path": "matrix_fixtures/project_stack.md"}}]),
            AIMessage(content="", tool_calls=[{"id": "call-read-2", "name": "read_file", "args": {"path": "matrix_fixtures/product_identity.md"}}]),
            AIMessage(content="AI Technical Assistant uses Python and Streamlit."),
        ]
    )
    runtime = make_runtime(model, tmp_path, provider_mode="native")
    request = SidecarChatRequest(
        sessionId="s1",
        messages=[
            ChatMessage(
                role="user",
                content="Read matrix_fixtures/project_stack.md, then read matrix_fixtures/product_identity.md, then answer briefly with the product name and stack.",
            )
        ],
        workspaceRoot=str(tmp_path),
        policyProfile="always_allow",
    )
    events = [event async for event in runtime.stream_chat(request, [HumanMessage(content=request.messages[0].content)])]
    final_text = "".join(str(event.get("token") or "") for event in events if event.get("type") == "token").strip()
    lowered = final_text.lower()
    tool_calls = [event for event in events if event.get("type") == "tool_call"]
    assert [event["name"] for event in tool_calls[:2]] == ["read_file", "read_file"]
    assert len(tool_calls) >= 2
    assert "ai technical assistant" in lowered
    assert "python" in lowered or "streamlit" in lowered
    second_invocation = model.invocations[1]
    assert any(
        isinstance(message, SystemMessage)
        and (
            "still missing verified evidence" in str(message.content).lower()
            or "missing evidence" in str(message.content).lower()
        )
        for message in second_invocation
    )
    evidence_event = next(event for event in events if event["type"] == "run_diagnostic" and event["code"] == "multistep_evidence_summary")
    assert evidence_event["data"]["fileCount"] >= 1


@pytest.mark.anyio
async def test_runtime_explicitly_acquires_missing_file_evidence(tmp_path: Path):
    fixtures = tmp_path / "matrix_fixtures"
    fixtures.mkdir(parents=True, exist_ok=True)
    fixtures.joinpath("project_stack.md").write_text("Stack: Python, Streamlit, FastAPI\n", encoding="utf-8")
    fixtures.joinpath("product_identity.md").write_text("Product name: AI Technical Assistant\n", encoding="utf-8")

    model = FakeModel(
        [
            AIMessage(content="", tool_calls=[{"id": "call-read-1", "name": "read_file", "args": {"path": "matrix_fixtures/project_stack.md"}}]),
            AIMessage(content="The stack is Python and Streamlit."),
            AIMessage(content="AI Technical Assistant uses Python and Streamlit."),
        ]
    )
    runtime = make_runtime(model, tmp_path, provider_mode="native")
    request = SidecarChatRequest(
        sessionId="s1",
        messages=[
            ChatMessage(
                role="user",
                content="Read matrix_fixtures/project_stack.md, then read matrix_fixtures/product_identity.md, then answer briefly with the product name and stack.",
            )
        ],
        workspaceRoot=str(tmp_path),
        policyProfile="always_allow",
    )
    events = [event async for event in runtime.stream_chat(request, [HumanMessage(content=request.messages[0].content)])]
    final_text = "".join(str(event.get("token") or "") for event in events if event.get("type") == "token").strip()
    lowered = final_text.lower()
    tool_calls = [event for event in events if event.get("type") == "tool_call"]
    assert [event["name"] for event in tool_calls[:2]] == ["read_file", "read_file"]
    assert len(tool_calls) >= 2
    assert "ai technical assistant" in lowered
    assert "python" in lowered or "streamlit" in lowered
    assert any(event["type"] == "run_diagnostic" and event["code"] == "explicit_evidence_acquisition" for event in events)


@pytest.mark.anyio
async def test_runtime_explicitly_acquires_missing_file_evidence_with_sequential_provider(tmp_path: Path):
    fixtures = tmp_path / "matrix_fixtures"
    fixtures.mkdir(parents=True, exist_ok=True)
    fixtures.joinpath("project_stack.md").write_text("Stack: Python, Streamlit, FastAPI\n", encoding="utf-8")
    fixtures.joinpath("product_identity.md").write_text("Product name: AI Technical Assistant\n", encoding="utf-8")

    model = FakeModel(
        [
            AIMessage(content="", tool_calls=[{"id": "call-read-1", "name": "read_file", "args": {"path": "matrix_fixtures/project_stack.md"}}]),
            AIMessage(content="The stack is Python and Streamlit."),
            AIMessage(content="AI Technical Assistant uses Python and Streamlit."),
        ]
    )
    runtime = make_runtime(
        model,
        tmp_path,
        provider_mode="native",
        provider_capabilities=ProviderCapabilities(
            supports_native_tools=True,
            supports_textual_replay=False,
            supports_multi_tool_turn=False,
            max_tool_calls_per_turn=1,
            requires_sequential_tool_loop=True,
            config_source="test",
            provider_family="openrouter",
        ),
    )
    request = SidecarChatRequest(
        sessionId="s1",
        messages=[
            ChatMessage(
                role="user",
                content="Read matrix_fixtures/project_stack.md, then read matrix_fixtures/product_identity.md, then answer briefly with the product name and stack.",
            )
        ],
        workspaceRoot=str(tmp_path),
        policyProfile="always_allow",
    )
    events = [event async for event in runtime.stream_chat(request, [HumanMessage(content=request.messages[0].content)])]
    final_text = "".join(str(event.get("token") or "") for event in events if event.get("type") == "token").strip().lower()
    tool_calls = [event for event in events if event.get("type") == "tool_call"]
    assert [event["name"] for event in tool_calls[:2]] == ["read_file", "read_file"]
    assert len(tool_calls) >= 2
    assert "ai technical assistant" in final_text
    assert any(event["type"] == "run_diagnostic" and event["code"] == "explicit_evidence_acquisition" for event in events)
    assert any(event["type"] == "run_diagnostic" and event["code"] == "stopped_because_missing_evidence" for event in events)


@pytest.mark.anyio
async def test_runtime_emits_goal_gap_summary_when_more_evidence_is_needed(tmp_path: Path):
    fixtures = tmp_path / "matrix_fixtures"
    fixtures.mkdir(parents=True, exist_ok=True)
    fixtures.joinpath("project_stack.md").write_text("Stack: Python, Streamlit, FastAPI\n", encoding="utf-8")
    fixtures.joinpath("product_identity.md").write_text("Product name: AI Technical Assistant\n", encoding="utf-8")

    model = FakeModel([AIMessage(content="unused")])
    runtime = make_runtime(model, tmp_path, provider_mode="native")
    request = SidecarChatRequest(
        sessionId="s1",
        messages=[ChatMessage(role="user", content="Read matrix_fixtures/project_stack.md, then read matrix_fixtures/product_identity.md, then answer with the product name and stack.")],
        workspaceRoot=str(tmp_path),
        policyProfile="always_allow",
    )
    state = runtime._initial_state(request, "run-gap", [HumanMessage(content=request.messages[0].content)])
    state["evidence_map"] = runtime._update_evidence_map(
        state,
        "read_file",
        {"path": "matrix_fixtures/project_stack.md"},
        "Stack: Python, Streamlit, FastAPI\n",
    )
    assessment = runtime._refresh_goal_tracking(state, "The stack is Python and Streamlit.")
    assert "file:matrix_fixtures/product_identity.md" in assessment.gap.missing_evidence
    assert assessment.gap.can_gather_more_evidence is True


@pytest.mark.anyio
async def test_runtime_goal_gap_summary_marks_completion_when_answer_is_grounded(tmp_path: Path):
    tmp_path.joinpath("a.txt").write_text("Product name: AI Technical Assistant\n", encoding="utf-8")
    model = FakeModel(
        [
            AIMessage(content="", tool_calls=[{"id": "call-read", "name": "read_file", "args": {"path": "a.txt"}}]),
            AIMessage(content="AI Technical Assistant."),
        ]
    )
    runtime = make_runtime(model, tmp_path, provider_mode="native")
    request = SidecarChatRequest(
        sessionId="s1",
        messages=[ChatMessage(role="user", content="Read a.txt and answer with the product name only.")],
        workspaceRoot=str(tmp_path),
        policyProfile="always_allow",
    )
    events = [event async for event in runtime.stream_chat(request, [HumanMessage(content=request.messages[0].content)])]
    goal_gap = next(event for event in events if event["type"] == "run_diagnostic" and event["code"] == "goal_gap_summary")
    assert goal_gap["data"]["isComplete"] is True
    assert goal_gap["data"]["missingEvidence"] == []
    assert goal_gap["data"]["missingFacts"] == []
    assert any(event["type"] == "run_state" and event["state"] == "completed" for event in events)


@pytest.mark.anyio
async def test_runtime_multistep_reuses_verified_facts_from_previous_turns(tmp_path: Path):
    fixtures = tmp_path / "matrix_fixtures"
    fixtures.mkdir(parents=True, exist_ok=True)
    fixtures.joinpath("project_stack.md").write_text("Stack: Python, Streamlit, FastAPI\n", encoding="utf-8")

    model = FakeModel(
        [
            AIMessage(content="", tool_calls=[{"id": "call-read-stack", "name": "read_file", "args": {"path": "matrix_fixtures/project_stack.md"}}]),
            AIMessage(content="AURORA_PHASE4 and Python / Streamlit / FastAPI."),
            AIMessage(content="AURORA_PHASE4 and Python / Streamlit / FastAPI."),
        ]
    )
    runtime = make_runtime(model, tmp_path, provider_mode="native")
    request = SidecarChatRequest(
        sessionId="s1",
        messages=[
            ChatMessage(
                role="user",
                content="Now read matrix_fixtures/project_stack.md and answer in one short sentence with the checkpoint plus the main stack.",
            )
        ],
        workspaceRoot=str(tmp_path),
        policyProfile="always_allow",
    )
    history = [
        HumanMessage(content="Read matrix_fixtures/roadmap_status.md and tell me only the recorded checkpoint."),
        AIMessage(content="AURORA_PHASE4"),
        HumanMessage(content=request.messages[0].content),
    ]
    events = [event async for event in runtime.stream_chat(request, history)]
    final_text = "".join(str(event.get("token") or "") for event in events if event.get("type") == "token").strip()
    lowered = final_text.lower()
    assert "aurora_phase4" in lowered
    assert "python" in lowered or "streamlit" in lowered
    assert any(event["type"] == "run_diagnostic" and event["code"] == "multistep_contract_detected" for event in events)


@pytest.mark.anyio
async def test_runtime_third_turn_forces_missing_file_read_with_sequential_provider(tmp_path: Path):
    fixtures = tmp_path / "matrix_fixtures"
    fixtures.mkdir(parents=True, exist_ok=True)
    fixtures.joinpath("product_identity.md").write_text("Product name: AI Technical Assistant\n", encoding="utf-8")

    model = FakeModel(
        [
            AIMessage(content="The recorded checkpoint is: AURORA_PHASE4. The main stack signals are: Python, Streamlit, FastAPI."),
            AIMessage(content="AI Technical Assistant. AURORA_PHASE4. Python / Streamlit / FastAPI."),
        ]
    )
    runtime = make_runtime(
        model,
        tmp_path,
        provider_mode="native",
        provider_capabilities=ProviderCapabilities(
            supports_native_tools=True,
            supports_textual_replay=False,
            supports_multi_tool_turn=False,
            max_tool_calls_per_turn=1,
            requires_sequential_tool_loop=True,
            config_source="test",
            provider_family="openrouter",
        ),
    )
    request = SidecarChatRequest(
        sessionId="s1",
        messages=[
            ChatMessage(
                role="user",
                content="Now read matrix_fixtures/product_identity.md and answer in one short sentence with the product name, checkpoint, and stack.",
            )
        ],
        workspaceRoot=str(tmp_path),
        policyProfile="always_allow",
    )
    history = [
        HumanMessage(content="Read matrix_fixtures/roadmap_status.md and tell me only the recorded checkpoint."),
        AIMessage(content="AURORA_PHASE4"),
        HumanMessage(content="Now read matrix_fixtures/project_stack.md and tell me only the main stack signals."),
        AIMessage(content="Python, Streamlit, FastAPI."),
        HumanMessage(content=request.messages[0].content),
    ]
    events = [event async for event in runtime.stream_chat(request, history)]
    tool_calls = [event for event in events if event.get("type") == "tool_call"]
    assert [event["name"] for event in tool_calls] == ["read_file"]
    final_text = "".join(str(event.get("token") or "") for event in events if event.get("type") == "token").strip().lower()
    assert "ai technical assistant" in final_text
    assert any(event["type"] == "run_diagnostic" and event["code"] == "explicit_evidence_acquisition" for event in events)


@pytest.mark.anyio
async def test_runtime_can_force_five_file_reads_one_by_one(tmp_path: Path):
    fixtures = tmp_path / "matrix_fixtures"
    fixtures.mkdir(parents=True, exist_ok=True)
    for idx in range(1, 6):
        fixtures.joinpath(f"f{idx}.md").write_text(f"Product name: File {idx}\n", encoding="utf-8")

    model = FakeModel(
        [
            AIMessage(content="", tool_calls=[{"id": "call-read-1", "name": "read_file", "args": {"path": "matrix_fixtures/f1.md"}}]),
            AIMessage(content="I only know File 1 so far."),
            AIMessage(content="I know Files 1 and 2."),
            AIMessage(content="I know Files 1 to 3."),
            AIMessage(content="I know Files 1 to 4."),
            AIMessage(content="Files 1, 2, 3, 4, and 5."),
            AIMessage(content="Files 1, 2, 3, 4, and 5."),
        ]
    )
    runtime = make_runtime(
        model,
        tmp_path,
        provider_mode="native",
        provider_capabilities=ProviderCapabilities(
            supports_native_tools=True,
            supports_textual_replay=False,
            supports_multi_tool_turn=False,
            max_tool_calls_per_turn=1,
            requires_sequential_tool_loop=True,
            config_source="test",
            provider_family="openrouter",
        ),
    )
    request = SidecarChatRequest(
        sessionId="s1",
        messages=[
            ChatMessage(
                role="user",
                content=(
                    "Read matrix_fixtures/f1.md, then matrix_fixtures/f2.md, then matrix_fixtures/f3.md, "
                    "then matrix_fixtures/f4.md, then matrix_fixtures/f5.md, then answer with all file names."
                ),
            )
        ],
        workspaceRoot=str(tmp_path),
        policyProfile="always_allow",
    )
    events = [event async for event in runtime.stream_chat(request, [HumanMessage(content=request.messages[0].content)])]
    tool_calls = [event for event in events if event.get("type") == "tool_call"]
    assert [event["name"] for event in tool_calls[:5]] == ["read_file", "read_file", "read_file", "read_file", "read_file"]


@pytest.mark.anyio
async def test_runtime_stops_repeated_same_tool_loop(tmp_path: Path):
    fixtures = tmp_path / "matrix_fixtures"
    fixtures.mkdir(parents=True, exist_ok=True)
    fixtures.joinpath("project_stack.md").write_text("Stack: Python, Streamlit, FastAPI\n", encoding="utf-8")

    model = FakeModel(
        [
            AIMessage(content="", tool_calls=[{"id": "call-read-1", "name": "read_file", "args": {"path": "matrix_fixtures/project_stack.md"}}]),
            AIMessage(content="", tool_calls=[{"id": "call-read-2", "name": "read_file", "args": {"path": "matrix_fixtures/project_stack.md"}}]),
            AIMessage(content="", tool_calls=[{"id": "call-read-3", "name": "read_file", "args": {"path": "matrix_fixtures/project_stack.md"}}]),
            AIMessage(content="", tool_calls=[{"id": "call-read-4", "name": "read_file", "args": {"path": "matrix_fixtures/project_stack.md"}}]),
            AIMessage(content="The stack is Python and Streamlit."),
        ]
    )
    runtime = make_runtime(model, tmp_path, provider_mode="native")
    request = SidecarChatRequest(
        sessionId="s1",
        messages=[
            ChatMessage(
                role="user",
                content="Read matrix_fixtures/project_stack.md and tell me the stack signals.",
            )
        ],
        workspaceRoot=str(tmp_path),
        policyProfile="always_allow",
    )
    events = [event async for event in runtime.stream_chat(request, [HumanMessage(content=request.messages[0].content)])]
    final_text = "".join(str(event.get("token") or "") for event in events if event.get("type") == "token").strip()
    assert "python" in final_text.lower()
    assert any(event["type"] == "run_diagnostic" and event["code"] == "repeated_tool_call_loop" for event in events)


@pytest.mark.anyio
async def test_runtime_terminal_tool_emits_terminal_events(tmp_path: Path):
    model = FakeModel(
        [
            AIMessage(content="", tool_calls=[{"id": "call-1", "name": "run_terminal", "args": {"command": "whoami"}}]),
            AIMessage(content="terminal complete"),
        ]
    )
    runtime = make_runtime(model, tmp_path, provider_mode="native")
    request = SidecarChatRequest(
        sessionId="s1",
        messages=[ChatMessage(role="user", content="run a single command")],
        workspaceRoot=str(tmp_path),
        policyProfile="always_allow",
    )
    events = [event async for event in runtime.stream_chat(request, [AIMessage(content="ignored")])]
    assert any(event["type"] == "terminal_opened" for event in events)
    assert any(event["type"] == "terminal_exit" and event["exitCode"] == 0 for event in events)
    assert any(event["type"] == "run_diagnostic" and event["code"] == "provider_mode" for event in events)
    assert any(event["type"] == "run_state" and event["state"] == "completed" for event in events)


@pytest.mark.anyio
async def test_runtime_falls_back_to_grounded_terminal_summary_in_requested_language(tmp_path: Path):
    model = FakeModel(
        [
            AIMessage(content="", tool_calls=[{"id": "call-1", "name": "run_terminal", "args": {"command": "echo LANGUAGE_EN_OK"}}]),
            AIMessage(content=""),
            AIMessage(content=""),
        ]
    )
    runtime = make_runtime(model, tmp_path, provider_mode="native")
    request = SidecarChatRequest(
        sessionId="s1",
        messages=[ChatMessage(role="user", content="Use the terminal to print LANGUAGE_EN_OK, then answer very briefly in English by explaining the result.")],
        workspaceRoot=str(tmp_path),
        policyProfile="always_allow",
    )
    events = [
        event
        async for event in runtime.stream_chat(
            request,
            [HumanMessage(content=request.messages[0].content)],
        )
    ]
    final_text = "".join(str(event.get("token") or "") for event in events if event.get("type") == "token").strip()
    assert final_text == 'The terminal result was "LANGUAGE_EN_OK".'
    assert any(event["type"] == "run_diagnostic" and event["code"] == "final_contract_repair_requested" for event in events)
    assert any(event["type"] == "run_state" and event["state"] == "completed" for event in events)


@pytest.mark.anyio
async def test_runtime_repairs_empty_final_answer_and_emits_metrics(tmp_path: Path):
    tmp_path.joinpath("a.txt").write_text("hello", encoding="utf-8")
    model = FakeModel(
        [
            AIMessage(content="", tool_calls=[{"id": "call-read-1", "name": "read_file", "args": {"path": "a.txt"}}]),
            AIMessage(content=""),
            AIMessage(content="Recovered from the file contents."),
        ]
    )
    runtime = make_runtime(model, tmp_path, provider_mode="native")
    request = SidecarChatRequest(
        sessionId="s1",
        messages=[ChatMessage(role="user", content="Read a.txt and answer briefly.")],
        workspaceRoot=str(tmp_path),
        policyProfile="always_allow",
    )
    events = [event async for event in runtime.stream_chat(request, [HumanMessage(content=request.messages[0].content)])]
    assert any(event["type"] == "run_diagnostic" and event["code"] == "empty_final_answer" for event in events)
    assert any(event["type"] == "run_diagnostic" and event["code"] == "verify_guardrail_repair" for event in events)
    metrics = next(event for event in events if event["type"] == "run_diagnostic" and event["code"] == "run_metrics")
    assert metrics["data"]["toolCalls"] >= 1


@pytest.mark.anyio
async def test_runtime_repairs_action_claim_without_tool(tmp_path: Path):
    tmp_path.joinpath("a.txt").write_text("hello", encoding="utf-8")
    model = FakeModel(
        [
            AIMessage(content="", tool_calls=[{"id": "call-read-1", "name": "read_file", "args": {"path": "a.txt"}}]),
            AIMessage(content="I opened the file a.txt and checked it."),
            AIMessage(content="I read a.txt and it contains hello."),
        ]
    )
    runtime = make_runtime(model, tmp_path, provider_mode="native")
    request = SidecarChatRequest(
        sessionId="s1",
        messages=[ChatMessage(role="user", content="Read a.txt and tell me what it contains.")],
        workspaceRoot=str(tmp_path),
        policyProfile="always_allow",
    )
    events = [event async for event in runtime.stream_chat(request, [HumanMessage(content=request.messages[0].content)])]
    final_text = "".join(str(event.get("token") or "") for event in events if event.get("type") == "token").strip().lower()
    assert "opened the file" not in final_text
    assert "contains hello" in final_text
    assert any(event["type"] == "run_diagnostic" and event["code"] == "action_claim_without_tool_open_file" for event in events)


@pytest.mark.anyio
async def test_runtime_repairs_followup_summary_with_previous_assistant_grounding(tmp_path: Path):
    model = FakeModel(
        [
            AIMessage(content="The workspace contains a mix of backend, frontend, and scripts."),
            AIMessage(content="The streamlit python only workspace contains backend, frontend, and scripts."),
        ]
    )
    runtime = make_runtime(model, tmp_path, provider_mode="native")
    request = SidecarChatRequest(
        sessionId="s1",
        messages=[ChatMessage(role="user", content="And now summarize it in one line.")],
        workspaceRoot="/Users/victor/Documents/continue-better/streamlit-python-only",
        policyProfile="always_allow",
    )
    history = [
        HumanMessage(content="Use the terminal to print the current directory and list the workspace root."),
        AIMessage(content="The current directory is `/Users/victor/Documents/continue-better/streamlit-python-only` in the streamlit python only workspace, and the workspace root contains README.md, artifacts/, backend/, backend.log."),
        HumanMessage(content="And now summarize it in one line."),
    ]
    events = [event async for event in runtime.stream_chat(request, history)]
    final_text = "".join(str(event.get("token") or "") for event in events if event.get("type") == "token").strip()
    assert "streamlit python only workspace" in final_text.lower()
    assert any(event["type"] == "run_diagnostic" and event["code"] == "final_contract_repair_requested" for event in events)
    assert any(event["type"] == "run_state" and event["state"] == "completed" for event in events)


@pytest.mark.anyio
async def test_runtime_repairs_followup_context_recall_when_model_returns_empty(tmp_path: Path):
    model = FakeModel(
        [
            AIMessage(content=""),
            AIMessage(content=""),
        ]
    )
    runtime = make_runtime(model, tmp_path, provider_mode="native")
    request = SidecarChatRequest(
        sessionId="s1",
        messages=[ChatMessage(role="user", content="Now remind me of that checkpoint in one short line.")],
        workspaceRoot=str(tmp_path),
        policyProfile="always_allow",
    )
    history = [
        HumanMessage(content="Read matrix_fixtures/roadmap_status.md and tell me very briefly what checkpoint is recorded."),
        AIMessage(content="The recorded checkpoint is **AURORA_PHASE4**."),
        HumanMessage(content="Now remind me of that checkpoint in one short line."),
    ]
    events = [event async for event in runtime.stream_chat(request, history)]
    final_text = "".join(str(event.get("token") or "") for event in events if event.get("type") == "token").strip()
    assert "aurora_phase4" in final_text.lower()
    assert any(event["type"] == "run_diagnostic" and event["code"] == "final_contract_repair_requested" for event in events)


@pytest.mark.anyio
async def test_runtime_repairs_grounded_list_directory_followup_without_reusing_previous_answer(tmp_path: Path):
    roadmap = tmp_path.joinpath("matrix_fixtures", "roadmap_status.md")
    roadmap.parent.mkdir(parents=True, exist_ok=True)
    roadmap.write_text("Current checkpoint: AURORA_PHASE4\n", encoding="utf-8")
    tmp_path.joinpath("frontend").mkdir()
    tmp_path.joinpath("backend").mkdir()
    tmp_path.joinpath("src").mkdir()

    model = FakeModel(
        [
            AIMessage(content="", tool_calls=[{"id": "call-read", "name": "read_file", "args": {"path": "matrix_fixtures/roadmap_status.md"}}]),
            AIMessage(content="The recorded checkpoint is **AURORA_PHASE4**."),
            AIMessage(content="", tool_calls=[{"id": "call-list", "name": "list_directory", "args": {}}]),
            AIMessage(content="The recorded checkpoint is **AURORA_PHASE4**."),
            AIMessage(content=""),
        ]
    )
    runtime = make_runtime(model, tmp_path, provider_mode="native")

    first_request = SidecarChatRequest(
        sessionId="s1",
        messages=[ChatMessage(role="user", content="Read matrix_fixtures/roadmap_status.md and tell me very briefly what checkpoint is recorded.")],
        workspaceRoot=str(tmp_path),
        policyProfile="always_allow",
    )
    first_events = [event async for event in runtime.stream_chat(first_request, [HumanMessage(content=first_request.messages[0].content)])]
    first_text = "".join(event["token"] for event in first_events if event["type"] == "token").strip()
    assert "AURORA_PHASE4" in first_text

    second_request = SidecarChatRequest(
        sessionId="s1",
        messages=[ChatMessage(role="user", content="Now inspect the workspace root and tell me briefly what kind of project this is.")],
        workspaceRoot=str(tmp_path),
        policyProfile="always_allow",
    )
    second_events = [event async for event in runtime.stream_chat(second_request, [HumanMessage(content=second_request.messages[0].content)])]
    second_text = "".join(event["token"] for event in second_events if event["type"] == "token").strip()
    lowered = second_text.lower()
    assert "aurora_phase4" not in lowered
    assert "project" in lowered or "app" in lowered or "application" in lowered
    assert "frontend" in lowered or "backend" in lowered or "python" in lowered
    assert any(event["type"] == "run_diagnostic" and event["code"] == "final_contract_repair_requested" for event in second_events)


@pytest.mark.anyio
async def test_runtime_terminal_policy_block_emits_diagnostic(tmp_path: Path):
    model = FakeModel(
        [
            AIMessage(content="", tool_calls=[{"id": "call-1", "name": "run_terminal", "args": {"command": "python -c \"print(1)\" && python -c \"print(2)\""}}]),
            AIMessage(content="blocked"),
        ]
    )
    runtime = make_runtime(model, tmp_path, provider_mode="native")
    request = SidecarChatRequest(sessionId="s1", messages=[ChatMessage(role="user", content="run unsafe command")], workspaceRoot=str(tmp_path))
    events = [event async for event in runtime.stream_chat(request, [AIMessage(content="ignored")])]
    assert any(event["type"] == "run_diagnostic" and event["code"] == "terminal_command_blocked" for event in events)
    assert any(event["type"] == "terminal_error" for event in events)


@pytest.mark.anyio
async def test_runtime_can_complete_interactive_terminal_task(tmp_path: Path):
    model = FakeModel(
        [
            AIMessage(content="", tool_calls=[{"id": "call-open", "name": "open_terminal", "args": {"cwd": "."}}]),
            AIMessage(content="", tool_calls=[{"id": "call-write", "name": "terminal_write", "args": {"data": "echo runtime-terminal\\n"}}]),
            AIMessage(content="", tool_calls=[{"id": "call-wait", "name": "terminal_wait_for_output", "args": {"pattern": "runtime-terminal", "timeoutMs": 4000}}]),
            AIMessage(content="", tool_calls=[{"id": "call-close", "name": "terminal_close", "args": {}}]),
            AIMessage(content="interactive terminal complete"),
        ]
    )
    runtime = make_runtime(model, tmp_path, provider_mode="native")
    request = SidecarChatRequest(
        sessionId="s1",
        messages=[ChatMessage(role="user", content="use the terminal interactively")],
        workspaceRoot=str(tmp_path),
        policyProfile="always_allow",
    )
    events = [event async for event in runtime.stream_chat(request, [AIMessage(content="ignored")])]
    assert any(event["type"] == "terminal_opened" for event in events)
    assert any(event["type"] == "terminal_input" and event["source"] == "agent" for event in events)
    assert any(event["type"] == "terminal_closed" for event in events)
    assert any(event["type"] == "terminal_control_changed" for event in events)
    assert any(event["type"] == "run_diagnostic" and event["code"] == "terminal_id_inferred" for event in events)
    assert any(event["type"] == "tool_call" and event["name"] == "terminal_wait_for_output" for event in events)
    assert any(event["type"] == "run_state" and event["state"] == "completed" for event in events)


@pytest.mark.anyio
async def test_runtime_respects_tool_toggles_for_web(tmp_path: Path):
    model = FakeModel([AIMessage(content="No tools used.")])
    runtime = make_runtime(model, tmp_path, provider_mode="native")
    request = SidecarChatRequest(
        sessionId="s1",
        messages=[ChatMessage(role="user", content="search the web")],
        workspaceRoot=str(tmp_path),
        toolToggles={"webSearch": False, "rag": True, "appActions": True, "clarification": True},
    )
    _ = [event async for event in runtime.stream_chat(request, [AIMessage(content="ignored")])]
    assert "web_search" not in model.bound_tool_names
    assert "rag_lookup" in model.bound_tool_names


@pytest.mark.anyio
async def test_runtime_injects_forced_rag_instruction(tmp_path: Path):
    model = FakeModel([AIMessage(content="Used forced mode."), AIMessage(content="RAG is unavailable for this request.")])
    runtime = make_runtime(model, tmp_path, provider_mode="native")
    request = SidecarChatRequest(
        sessionId="s1",
        messages=[ChatMessage(role="user", content="find this in local docs")],
        workspaceRoot=str(tmp_path),
        forceToolUse="rag",
    )
    _ = [event async for event in runtime.stream_chat(request, [AIMessage(content="ignored")])]
    assert any(isinstance(message, SystemMessage) and "must use rag_lookup" in message.content for message in model.invocations[0])


@pytest.mark.anyio
async def test_runtime_injects_base_system_prompt(tmp_path: Path):
    model = FakeModel([AIMessage(content="Prompted.")])
    runtime = make_runtime(model, tmp_path, provider_mode="native")
    request = SidecarChatRequest(
        sessionId="s1",
        messages=[ChatMessage(role="user", content="Use the interactive terminal to run pwd")],
        workspaceRoot=str(tmp_path),
    )
    _ = [event async for event in runtime.stream_chat(request, [AIMessage(content="ignored")])]
    system_messages = [message.content for message in model.invocations[0] if isinstance(message, SystemMessage)]
    joined = "\n".join(system_messages)
    assert "local tool-enabled runtime" in joined
    assert "interactive terminal workflow" in joined
    assert "Never return raw tool-call JSON as a final answer." in joined


@pytest.mark.anyio
async def test_runtime_injects_sequential_tool_prompt_for_thales_enterprise(tmp_path: Path):
    model = FakeModel([AIMessage(content="Prompted.")])
    runtime = make_runtime(
        model,
        tmp_path,
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
    request = SidecarChatRequest(
        sessionId="s1",
        messages=[ChatMessage(role="user", content="Inspect the repo and answer.")],
        workspaceRoot=str(tmp_path),
    )
    _ = [event async for event in runtime.stream_chat(request, [HumanMessage(content=request.messages[0].content)])]
    system_messages = [message.content for message in model.invocations[0] if isinstance(message, SystemMessage)]
    joined = "\n".join(system_messages)
    assert "at most one tool call per assistant turn" in joined.lower()
    assert "do not plan multiple tool calls" in joined.lower()


@pytest.mark.anyio
async def test_runtime_rag_followup_reuses_session_doc_context(tmp_path: Path):
    rag_service = build_rag_service(tmp_path)
    source = tmp_path.joinpath("roadmap_status.md")
    source.write_text(
        "# Matrix Fixture\nCurrent checkpoint: AURORA_PHASE4\nPhase 4 status: very advanced and close to closure.\n",
        encoding="utf-8",
    )
    rag_service.import_file("session", "s1", str(source), "matrix/roadmap_status.md")
    rag_service.enqueue_index_job("session", "s1")
    model = FakeModel(
        [
            AIMessage(content="", tool_calls=[{"id": "call-rag-1", "name": "rag_lookup", "args": {"question": "With the RAG for this session, what checkpoint is recorded in the indexed document?"}}]),
            AIMessage(content="AURORA_PHASE4 (matrix/roadmap_status.md:2)"),
            AIMessage(content="", tool_calls=[{"id": "call-rag-2", "name": "rag_lookup", "args": {"question": "And what is the Phase 4 status?"}}]),
            AIMessage(content="Phase 4 status is very advanced and close to closure. (matrix/roadmap_status.md:3)"),
        ]
    )
    runtime = make_runtime(model, tmp_path, provider_mode="native", rag_service=rag_service)

    first_request = SidecarChatRequest(
        sessionId="s1",
        messages=[ChatMessage(role="user", content="With the RAG for this session, what checkpoint is recorded in the indexed document? Answer very briefly.")],
        workspaceRoot=str(tmp_path),
        policyProfile="always_allow",
        forceToolUse="rag",
    )
    first_events = [event async for event in runtime.stream_chat(first_request, [HumanMessage(content=first_request.messages[0].content)])]
    first_text = "".join(event["token"] for event in first_events if event["type"] == "token").strip()
    assert "AURORA_PHASE4" in first_text

    second_request = SidecarChatRequest(
        sessionId="s1",
        messages=[ChatMessage(role="user", content="And what is the Phase 4 status? Answer very briefly.")],
        workspaceRoot=str(tmp_path),
        policyProfile="always_allow",
        forceToolUse="rag",
    )
    second_events = [event async for event in runtime.stream_chat(second_request, [HumanMessage(content=second_request.messages[0].content)])]
    second_text = "".join(event["token"] for event in second_events if event["type"] == "token").strip()
    assert "very advanced" in second_text.lower() or "close to closure" in second_text.lower()
    assert "matrix/roadmap_status.md:3" in second_text
    assert any(event["type"] == "run_diagnostic" and event["code"] == "rag_followup_context_used" for event in second_events)


@pytest.mark.anyio
async def test_runtime_rag_weak_hits_fall_back_to_cautious_answer(tmp_path: Path):
    rag_service = build_rag_service(tmp_path)
    source = tmp_path.joinpath("notes.md")
    source.write_text("This document only talks about apples and oranges.", encoding="utf-8")
    rag_service.import_file("session", "s1", str(source), "docs/notes.md")
    rag_service.enqueue_index_job("session", "s1")
    model = FakeModel(
        [
            AIMessage(content="", tool_calls=[{"id": "call-rag", "name": "rag_lookup", "args": {"question": "What is the Phase 4 status?"}}]),
            AIMessage(content="Phase 4 status is approved."),
            AIMessage(content=""),
        ]
    )
    runtime = make_runtime(model, tmp_path, provider_mode="native", rag_service=rag_service)
    request = SidecarChatRequest(
        sessionId="s1",
        messages=[ChatMessage(role="user", content="With the RAG for this session, what is the Phase 4 status? Answer very briefly.")],
        workspaceRoot=str(tmp_path),
        policyProfile="always_allow",
        forceToolUse="rag",
    )
    events = [event async for event in runtime.stream_chat(request, [HumanMessage(content=request.messages[0].content)])]
    final_text = "".join(event["token"] for event in events if event["type"] == "token").strip()
    assert "couldn't find enough reliable information" in final_text.lower()
    assert any(event["type"] == "run_diagnostic" and event["code"] == "rag_lookup_no_strong_hits" for event in events)
    assert any(event["type"] == "run_diagnostic" and event["code"] == "rag_grounding_weak" for event in events)
    assert any(event["type"] == "run_diagnostic" and event["code"] == "final_contract_repair_requested" for event in events)


@pytest.mark.anyio
async def test_runtime_repairs_rag_answer_that_ignores_salient_evidence(tmp_path: Path):
    rag_service = build_rag_service(tmp_path)
    source = tmp_path.joinpath("roadmap_status.md")
    source.write_text(
        "# Matrix Fixture\nCurrent checkpoint: AURORA_PHASE4\nPhase 4 status: very advanced and close to closure.\n",
        encoding="utf-8",
    )
    rag_service.import_file("session", "s1", str(source), "matrix/roadmap_status.md")
    rag_service.enqueue_index_job("session", "s1")
    model = FakeModel(
        [
            AIMessage(content="", tool_calls=[{"id": "call-rag", "name": "rag_lookup", "args": {"question": "What is the Phase 4 status?"}}]),
            AIMessage(content="Phase 4 status is Complete. (matrix/roadmap_status.md:3)"),
            AIMessage(content=""),
        ]
    )
    runtime = make_runtime(model, tmp_path, provider_mode="native", rag_service=rag_service)
    request = SidecarChatRequest(
        sessionId="s1",
        messages=[ChatMessage(role="user", content="With the RAG for this session, what is the Phase 4 status? Answer very briefly with a citation.")],
        workspaceRoot=str(tmp_path),
        policyProfile="always_allow",
        forceToolUse="rag",
    )
    events = [event async for event in runtime.stream_chat(request, [HumanMessage(content=request.messages[0].content)])]
    final_text = "".join(event["token"] for event in events if event["type"] == "token").strip()
    lowered = final_text.lower()
    assert "very advanced" in lowered or "close to closure" in lowered
    assert "matrix/roadmap_status.md" in final_text
    assert any(event["type"] == "run_diagnostic" and event["code"] == "rag_grounding_weak" for event in events)
    assert any(event["type"] == "run_diagnostic" and event["code"] == "final_contract_repair_requested" for event in events)


@pytest.mark.anyio
async def test_runtime_repairs_exact_output_after_terminal_tool(tmp_path: Path):
    model = FakeModel(
        [
            AIMessage(content="", tool_calls=[{"id": "call-1", "name": "run_terminal", "args": {"command": "printf MATRIX_TERMINAL_OK"}}]),
            AIMessage(content="The command succeeded."),
            AIMessage(content="MATRIX_TERMINAL_OK"),
        ]
    )
    runtime = make_runtime(model, tmp_path, provider_mode="native")
    request = SidecarChatRequest(
        sessionId="s1",
        messages=[ChatMessage(role="user", content="Use the terminal to print MATRIX_TERMINAL_OK, then reply with exactly MATRIX_TERMINAL_OK.")],
        workspaceRoot=str(tmp_path),
        policyProfile="always_allow",
    )
    events = [event async for event in runtime.stream_chat(request, [HumanMessage(content=request.messages[0].content)])]
    final_text = "".join(event["token"] for event in events if event["type"] == "token").strip()
    assert final_text == "MATRIX_TERMINAL_OK"
    assert any(event["type"] == "run_diagnostic" and event["code"] == "final_contract_detected" for event in events)
    assert any(event["type"] == "run_diagnostic" and event["code"] == "exact_output_mismatch" for event in events)
    assert any(event["type"] == "run_diagnostic" and event["code"] == "final_contract_repair_requested" for event in events)


@pytest.mark.anyio
async def test_runtime_repairs_web_answer_with_source_links(tmp_path: Path, monkeypatch):
    class FakeResponse:
        text = (
            '<a class="result__a" href="https://example.com/openai-a">OpenAI A</a>'
            '<a class="result__a" href="https://example.com/openai-b">OpenAI B</a>'
        )

        def raise_for_status(self):
            return None

    monkeypatch.setattr("streamlit_python_only.tooling.httpx.get", lambda *args, **kwargs: FakeResponse())
    model = FakeModel(
        [
            AIMessage(content="", tool_calls=[{"id": "call-web", "name": "web_search", "args": {"query": "latest OpenAI news", "limit": 2}}]),
            AIMessage(content="Short OpenAI update."),
            AIMessage(content="Short OpenAI update. https://example.com/openai-a https://example.com/openai-b"),
        ]
    )
    runtime = make_runtime(model, tmp_path, provider_mode="native")
    request = SidecarChatRequest(
        sessionId="s1",
        messages=[ChatMessage(role="user", content="Give me a very short summary of the latest OpenAI news with two source links.")],
        workspaceRoot=str(tmp_path),
        policyProfile="always_allow",
    )
    events = [event async for event in runtime.stream_chat(request, [HumanMessage(content=request.messages[0].content)])]
    final_text = "".join(event["token"] for event in events if event["type"] == "token").strip()
    assert "https://example.com/openai-a" in final_text
    assert "https://example.com/openai-b" in final_text
    assert any(event["type"] == "run_diagnostic" and event["code"] == "sources_missing_in_final_answer" for event in events)
    assert any(event["type"] == "run_diagnostic" and event["code"] == "final_contract_repair_requested" for event in events)


@pytest.mark.anyio
async def test_runtime_falls_back_to_web_summary_when_repair_response_is_empty(tmp_path: Path, monkeypatch):
    class FakeResponse:
        text = (
            '<a class="result__a" href="https://example.com/openai-a">OpenAI A</a>'
            '<a class="result__a" href="https://example.com/openai-b">OpenAI B</a>'
        )

        def raise_for_status(self):
            return None

    monkeypatch.setattr("streamlit_python_only.tooling.httpx.get", lambda *args, **kwargs: FakeResponse())
    model = FakeModel(
        [
            AIMessage(content="", tool_calls=[{"id": "call-web", "name": "web_search", "args": {"query": "latest OpenAI news", "limit": 2}}]),
            AIMessage(content=""),
            AIMessage(content=""),
        ]
    )
    runtime = make_runtime(model, tmp_path, provider_mode="native")
    request = SidecarChatRequest(
        sessionId="s1",
        messages=[ChatMessage(role="user", content="Give me a very short summary of the latest OpenAI news with two source links.")],
        workspaceRoot=str(tmp_path),
        policyProfile="always_allow",
    )
    events = [event async for event in runtime.stream_chat(request, [HumanMessage(content=request.messages[0].content)])]
    final_text = "".join(event["token"] for event in events if event["type"] == "token").strip()
    assert "https://example.com/openai-a" in final_text
    assert "https://example.com/openai-b" in final_text


@pytest.mark.anyio
async def test_runtime_falls_back_to_terminal_summary_when_repair_response_is_empty(tmp_path: Path):
    model = FakeModel(
        [
            AIMessage(content="", tool_calls=[{"id": "call-pwd", "name": "run_terminal", "args": {"command": "pwd"}}]),
            AIMessage(content="", tool_calls=[{"id": "call-ls", "name": "run_terminal", "args": {"command": "ls"}}]),
            AIMessage(content=""),
            AIMessage(content=""),
        ]
    )
    runtime = make_runtime(model, tmp_path, provider_mode="native")
    request = SidecarChatRequest(
        sessionId="s1",
        messages=[ChatMessage(role="user", content="Use the terminal to print the current directory and list the workspace root, then summarize in one line.")],
        workspaceRoot=str(tmp_path),
        policyProfile="always_allow",
    )
    events = [event async for event in runtime.stream_chat(request, [HumanMessage(content=request.messages[0].content)])]
    final_text = "".join(event["token"] for event in events if event["type"] == "token").strip()
    assert "current directory" in final_text.lower()
    assert Path(str(tmp_path)).name in final_text


@pytest.mark.anyio
async def test_runtime_repairs_wrong_language_response(tmp_path: Path):
    model = FakeModel([AIMessage(content="Hello."), AIMessage(content="Bonjour.")])
    runtime = make_runtime(model, tmp_path, provider_mode="native")
    request = SidecarChatRequest(
        sessionId="s1",
        messages=[ChatMessage(role="user", content="Explique très brièvement ce que tu fais.")],
        workspaceRoot=str(tmp_path),
    )
    events = [event async for event in runtime.stream_chat(request, [HumanMessage(content=request.messages[0].content)])]
    final_text = "".join(event["token"] for event in events if event["type"] == "token").strip()
    assert final_text == "Bonjour."
    assert any(event["type"] == "run_diagnostic" and event["code"] == "wrong_response_language" for event in events)


@pytest.mark.anyio
async def test_runtime_textual_replay_collapses_multiple_tool_calls(tmp_path: Path):
    tmp_path.joinpath("notes.txt").write_text("hello", encoding="utf-8")
    model = FakeModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "call-read", "name": "read_file", "args": {"path": "notes.txt"}},
                    {"id": "call-list", "name": "list_directory", "args": {"path": "."}},
                ],
            ),
            AIMessage(content="notes.txt contains hello."),
            AIMessage(content="notes.txt contains hello, and the sequential provider should inspect the directory on a later turn."),
        ]
    )
    runtime = make_runtime(model, tmp_path, provider_mode="textual_replay")
    request = SidecarChatRequest(
        sessionId="s1",
        messages=[ChatMessage(role="user", content="Read notes.txt and inspect the directory.")],
        workspaceRoot=str(tmp_path),
        policyProfile="always_allow",
    )
    events = [event async for event in runtime.stream_chat(request, [HumanMessage(content=request.messages[0].content)])]
    tool_calls = [event for event in events if event["type"] == "tool_call"]
    first_tool_call = next(event for event in tool_calls if event["type"] == "tool_call")
    assert first_tool_call["name"] == "read_file"
    assert any(event["type"] == "run_diagnostic" and event["code"] == "provider_tool_calls_collapsed" for event in events)
    assert any(event["type"] == "run_diagnostic" and event["code"] == "provider_family" and "thales" in event["message"].lower() for event in events)
    assert any(event["type"] == "run_diagnostic" and event["code"] == "provider_config_source" for event in events)


@pytest.mark.anyio
async def test_runtime_native_mode_keeps_multiple_tool_calls(tmp_path: Path):
    tmp_path.joinpath("notes.txt").write_text("hello", encoding="utf-8")
    model = FakeModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "call-read", "name": "read_file", "args": {"path": "notes.txt"}},
                    {"id": "call-list", "name": "list_directory", "args": {"path": "."}},
                ],
            ),
            AIMessage(content="notes.txt contains hello and the directory includes notes.txt."),
        ]
    )
    runtime = make_runtime(model, tmp_path, provider_mode="native")
    request = SidecarChatRequest(
        sessionId="s1",
        messages=[ChatMessage(role="user", content="Read notes.txt and inspect the directory.")],
        workspaceRoot=str(tmp_path),
        policyProfile="always_allow",
    )
    events = [event async for event in runtime.stream_chat(request, [HumanMessage(content=request.messages[0].content)])]
    tool_calls = [event for event in events if event["type"] == "tool_call"]
    assert len(tool_calls) == 2
    assert not any(event["type"] == "run_diagnostic" and event["code"] == "provider_tool_calls_collapsed" for event in events)


@pytest.mark.anyio
async def test_runtime_limits_final_repair_to_one_attempt(tmp_path: Path):
    model = FakeModel([AIMessage(content="Hello."), AIMessage(content="Still hello.")])
    runtime = make_runtime(model, tmp_path, provider_mode="native")
    request = SidecarChatRequest(
        sessionId="s1",
        messages=[ChatMessage(role="user", content="Réponds brièvement en français.")],
        workspaceRoot=str(tmp_path),
    )
    events = [event async for event in runtime.stream_chat(request, [HumanMessage(content=request.messages[0].content)])]
    assert sum(1 for event in events if event["type"] == "run_diagnostic" and event["code"] == "final_contract_repair_requested") == 1
    assert any(event["type"] == "run_diagnostic" and event["code"] == "final_contract_repair_failed" for event in events)
    assert len(model.invocations) == 2


@pytest.mark.anyio
async def test_runtime_repairs_exact_output_for_say_exactly_prompt(tmp_path: Path):
    model = FakeModel([AIMessage(content="MESSAGE_ORDER_OK"), AIMessage(content="MESSAGE_ORDER_OK.")])
    runtime = make_runtime(model, tmp_path, provider_mode="native")
    request = SidecarChatRequest(
        sessionId="s1",
        messages=[ChatMessage(role="user", content="Say exactly MESSAGE_ORDER_OK.")],
        workspaceRoot=str(tmp_path),
    )
    events = [event async for event in runtime.stream_chat(request, [HumanMessage(content=request.messages[0].content)])]
    final_text = "".join(event["token"] for event in events if event["type"] == "token").strip()
    assert final_text == "MESSAGE_ORDER_OK."
    assert any(event["type"] == "run_diagnostic" and event["code"] == "exact_output_mismatch" for event in events)
    assert any(event["type"] == "run_diagnostic" and event["code"] == "final_contract_repair_requested" for event in events)


@pytest.mark.anyio
async def test_runtime_emits_heartbeat_for_slow_runs(tmp_path: Path):
    model = FakeModel([AIMessage(content="Completed after delay.")], delay_sec=10.2)
    runtime = make_runtime(model, tmp_path, provider_mode="native")
    request = SidecarChatRequest(
        sessionId="s1",
        messages=[ChatMessage(role="user", content="say hi")],
        workspaceRoot=str(tmp_path),
    )
    events = [event async for event in runtime.stream_chat(request, [AIMessage(content="ignored")])]
    assert any(event["type"] == "run_diagnostic" and event["code"] == "runtime_heartbeat" for event in events)
    assert any(event["type"] == "run_state" and event["state"] == "completed" for event in events)


@pytest.mark.anyio
async def test_runtime_uses_early_clarification_for_broad_product_prompt(tmp_path: Path):
    model = FakeModel([AIMessage(content="Should not be called.")])
    runtime = make_runtime(model, tmp_path, provider_mode="native")
    request = SidecarChatRequest(
        sessionId="s1",
        messages=[ChatMessage(role="user", content="Help me create a food application.")],
        workspaceRoot=str(tmp_path),
        toolToggles={"clarification": True},
    )
    events = [event async for event in runtime.stream_chat(request, [HumanMessage(content="Help me create a food application.")])]
    assert model.invocations == []
    assert any(event["type"] == "tool_call" and event["name"] == "request_clarification" for event in events)
    assert any(event["type"] == "clarification_required" for event in events)
    assert any(event["type"] == "run_state" and event["state"] == "awaiting_clarification" for event in events)


@pytest.mark.anyio
async def test_runtime_ask_when_necessary_requests_approval_for_moderate_tools(tmp_path: Path):
    model = FakeModel([AIMessage(content="", tool_calls=[{"id": "call-1", "name": "open_terminal", "args": {"cwd": "."}}])])
    runtime = make_runtime(model, tmp_path, provider_mode="native")
    request = SidecarChatRequest(
        sessionId="s1",
        messages=[ChatMessage(role="user", content="open a terminal")],
        workspaceRoot=str(tmp_path),
        policyProfile="ask_when_necessary",
        toolToggles={"clarification": False},
    )
    events = [event async for event in runtime.stream_chat(request, [HumanMessage(content="open a terminal")])]
    assert any(event["type"] == "approval_required" and event["name"] == "open_terminal" for event in events)


@pytest.mark.anyio
async def test_runtime_always_ask_requests_approval_for_safe_tools(tmp_path: Path):
    model = FakeModel([AIMessage(content="", tool_calls=[{"id": "call-1", "name": "read_file", "args": {"path": "notes.txt"}}])])
    tmp_path.joinpath("notes.txt").write_text("hello", encoding="utf-8")
    runtime = make_runtime(model, tmp_path, provider_mode="native")
    request = SidecarChatRequest(
        sessionId="s1",
        messages=[ChatMessage(role="user", content="read notes.txt")],
        workspaceRoot=str(tmp_path),
        policyProfile="always_ask",
    )
    events = [event async for event in runtime.stream_chat(request, [HumanMessage(content="read notes.txt")])]
    assert any(event["type"] == "approval_required" and event["name"] == "read_file" for event in events)


def test_terminal_manager_emits_structured_terminal_action_events(tmp_path: Path):
    manager = TerminalManager()
    created = manager.create_terminal(
        workspace_root=str(tmp_path),
        session_id="session-1",
        run_id="run-1",
        owner="agent",
    )
    terminal_id = created["terminal"]["terminalId"]
    resolution = manager.resolve_terminal_reference("__FROM_TOOL__", run_id="run-1", session_id="session-1")
    assert resolution["terminalId"] == terminal_id

    subscription, initial, record = manager.stream_subscription(terminal_id)
    assert any(event["type"] == "terminal_control_changed" for event in initial)

    wrote = manager.write_terminal(terminal_id, "echo terminal-manager\n", "agent")
    assert wrote["events"][0]["type"] == "terminal_input"
    waited = manager.wait_for_output(terminal_id, pattern="terminal-manager", timeout_ms=5000)
    assert waited["matched"] is True

    interrupted = manager.interrupt_terminal(terminal_id, "user")
    assert interrupted["events"][0]["inputKind"] == "interrupt"

    resized = manager.resize_terminal(terminal_id, 132, 44)
    assert resized["events"][0]["type"] == "terminal_resized"
    control = manager.set_control(terminal_id, "user", "handoff")
    assert control["events"][0]["owner"] == "user"

    closed = manager.close_terminal(terminal_id, "agent_closed")
    assert any(event["type"] == "terminal_closed" for event in closed["events"])
    streamed = _collect_terminal_events(
        subscription,
        {"terminal_input", "terminal_resized", "terminal_control_changed", "terminal_closed", "terminal_exit"},
    )
    manager.remove_subscription(record, subscription)
    event_types = {event.get("type") for event in streamed}
    assert {"terminal_input", "terminal_resized", "terminal_control_changed", "terminal_closed", "terminal_exit"}.issubset(event_types)


def test_terminal_manager_can_close_all_run_terminals(tmp_path: Path):
    manager = TerminalManager()
    first = manager.create_terminal(workspace_root=str(tmp_path), session_id="session-1", run_id="run-close", owner="agent")
    second = manager.create_terminal(workspace_root=str(tmp_path), session_id="session-1", run_id="run-close", owner="user")
    listed = manager.list_terminals(run_id="run-close", alive_only=True)
    assert len(listed["terminals"]) == 2

    closed = manager.close_run_terminals("run-close", reason="cleanup")
    assert len(closed["terminals"]) == 2
    assert any(event["type"] == "terminal_closed" for event in closed["events"])

    first_snapshot = manager.get_terminal(first["terminal"]["terminalId"])["terminal"]
    second_snapshot = manager.get_terminal(second["terminal"]["terminalId"])["terminal"]
    assert first_snapshot["alive"] is False
    assert second_snapshot["alive"] is False


def test_terminal_manager_resolve_is_ambiguous_with_multiple_run_terminals(tmp_path: Path):
    manager = TerminalManager()
    manager.create_terminal(workspace_root=str(tmp_path), session_id="session-1", run_id="run-ambiguous", owner="agent")
    manager.create_terminal(workspace_root=str(tmp_path), session_id="session-1", run_id="run-ambiguous", owner="user")
    manager._active_by_run.pop("run-ambiguous", None)
    manager._active_by_session.pop("session-1", None)

    with pytest.raises(ValueError, match="multiple terminals"):
        manager.resolve_terminal_reference(None, run_id="run-ambiguous", session_id="session-1")


def test_terminal_manager_benchmark_smoke(tmp_path: Path):
    manager = TerminalManager()
    start_open = time.perf_counter()
    created = manager.create_terminal(workspace_root=str(tmp_path), session_id="bench-session", run_id="bench-run", owner="agent")
    open_ms = (time.perf_counter() - start_open) * 1000
    terminal_id = created["terminal"]["terminalId"]

    start_write = time.perf_counter()
    manager.write_terminal(terminal_id, "echo benchmark-smoke\n", "agent")
    waited = manager.wait_for_output(terminal_id, pattern="benchmark-smoke", timeout_ms=5000)
    roundtrip_ms = (time.perf_counter() - start_write) * 1000

    start_close = time.perf_counter()
    manager.close_terminal(terminal_id, "benchmark")
    close_ms = (time.perf_counter() - start_close) * 1000

    assert waited["matched"] is True
    assert open_ms < 5000
    assert roundtrip_ms < 5000
    assert close_ms < 5000
