from __future__ import annotations

from pathlib import Path
import time

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from continue_better_py.run_state import RunStateStore
from continue_better_py.runtime import RuntimeDependencies, RuntimeEngine
from continue_better_py.schemas import ApprovalDecisionRequest, ClarificationDecisionRequest, ChatMessage, SidecarChatRequest
from continue_better_py.terminal_manager import TerminalManager
from continue_better_py.tool_registry import create_default_tool_registry


class FakeProvider:
    def __init__(self, mode: str) -> None:
        self.mode = mode


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


def make_runtime(fake_model: FakeModel, tmp_path: Path, provider_mode: str = "native") -> RuntimeEngine:
    store = RunStateStore()
    terminals = TerminalManager()
    store.root = tmp_path
    tmp_path.joinpath("approvals").mkdir(parents=True, exist_ok=True)
    tmp_path.joinpath("clarifications").mkdir(parents=True, exist_ok=True)
    deps = RuntimeDependencies(
        model_factory=lambda _profile, _model: fake_model,
        provider_resolver=lambda _profile, _model: FakeProvider(provider_mode),
        tool_registry_factory=lambda workspace_root, session_id, run_id, tool_toggles=None: create_default_tool_registry(
            workspace_root,
            session_id=session_id,
            run_id=run_id,
            terminal_manager=terminals,
            tool_toggles=tool_toggles,
        ),
        state_store=store,
        terminal_manager=terminals,
    )
    return RuntimeEngine(dependencies=deps)


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
async def test_runtime_terminal_tool_emits_terminal_events(tmp_path: Path):
    model = FakeModel(
        [
            AIMessage(content="", tool_calls=[{"id": "call-1", "name": "run_terminal", "args": {"command": "pwd"}}]),
            AIMessage(content="terminal complete"),
        ]
    )
    runtime = make_runtime(model, tmp_path, provider_mode="native")
    request = SidecarChatRequest(sessionId="s1", messages=[ChatMessage(role="user", content="run pwd")], workspaceRoot=str(tmp_path))
    events = [event async for event in runtime.stream_chat(request, [AIMessage(content="ignored")])]
    assert any(event["type"] == "terminal_opened" for event in events)
    assert any(event["type"] == "terminal_exit" and event["exitCode"] == 0 for event in events)
    assert any(event["type"] == "run_diagnostic" and event["code"] == "provider_mode" for event in events)
    assert any(event["type"] == "run_state" and event["state"] == "completed" for event in events)


@pytest.mark.anyio
async def test_runtime_terminal_policy_block_emits_diagnostic(tmp_path: Path):
    model = FakeModel(
        [
            AIMessage(content="", tool_calls=[{"id": "call-1", "name": "run_terminal", "args": {"command": "ls && pwd"}}]),
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
            AIMessage(content="", tool_calls=[{"id": "call-write", "name": "terminal_write", "args": {"terminalId": "__FROM_TOOL__", "data": "pwd\\n"}}]),
            AIMessage(content="", tool_calls=[{"id": "call-wait", "name": "terminal_wait_for_output", "args": {"terminalId": "__FROM_TOOL__", "pattern": str(tmp_path), "timeoutMs": 4000}}]),
            AIMessage(content="", tool_calls=[{"id": "call-close", "name": "terminal_close", "args": {"terminalId": "__FROM_TOOL__"}}]),
            AIMessage(content="interactive terminal complete"),
        ]
    )
    runtime = make_runtime(model, tmp_path, provider_mode="native")

    original_invoke = model.invoke

    def invoke_with_terminal_id(messages: list[BaseMessage]):
        for queued in model.responses:
            if queued.tool_calls:
                for tool_call in queued.tool_calls:
                    if tool_call["args"].get("terminalId") == "__FROM_TOOL__":
                        terminal_message = next(
                            (
                                message
                                for message in reversed(messages)
                                if isinstance(message, ToolMessage) and "Terminal:" in str(message.content)
                            ),
                            None,
                        )
                        if terminal_message:
                            lines = str(terminal_message.content).splitlines()
                            terminal_id = next((line.split(":", 1)[1].strip() for line in lines if line.startswith("Terminal:")), "")
                            if terminal_id:
                                tool_call["args"]["terminalId"] = terminal_id
        return original_invoke(messages)

    model.invoke = invoke_with_terminal_id  # type: ignore[method-assign]
    request = SidecarChatRequest(
        sessionId="s1",
        messages=[ChatMessage(role="user", content="use the terminal interactively")],
        workspaceRoot=str(tmp_path),
        policyProfile="always_allow",
    )
    events = [event async for event in runtime.stream_chat(request, [AIMessage(content="ignored")])]
    assert any(event["type"] == "terminal_opened" for event in events)
    assert any(event["type"] == "terminal_control_changed" for event in events)
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
    model = FakeModel([AIMessage(content="Used forced mode.")])
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
