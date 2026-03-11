from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import Annotated

from continue_better_py.events import (
    approval_decision,
    approval_required,
    clarification_required,
    done,
    error_event,
    run_phase,
    run_diagnostic,
    run_state,
    terminal_error,
    terminal_exit,
    terminal_opened,
    token,
)
from continue_better_py.providers import ResolvedProvider, build_chat_model, resolve_provider
from continue_better_py.run_state import RunStateStore, deserialize_messages, serialize_messages
from continue_better_py.schemas import ApprovalDecisionRequest, ClarificationDecisionRequest, SidecarChatRequest
from continue_better_py.terminal_manager import TerminalManager
from continue_better_py.tool_registry import ToolRegistry, create_default_tool_registry


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    session_id: str
    run_id: str
    workspace_root: str
    profile: str | None
    model: str | None
    provider_mode: str
    policy_profile: str
    final_text: str
    pending_approval: dict[str, Any] | None
    pending_clarification: dict[str, Any] | None
    tool_events: list[dict[str, Any]]


@dataclass(slots=True)
class RuntimeDependencies:
    model_factory: Callable[[str | None, str | None], Any]
    provider_resolver: Callable[[str | None, str | None], ResolvedProvider]
    tool_registry_factory: Callable[[str, str | None, str | None], ToolRegistry]
    state_store: RunStateStore
    terminal_manager: TerminalManager = field(default_factory=TerminalManager)


def split_for_streaming(text: str) -> list[str]:
    parts = text.split()
    if not parts:
        return [text]
    return [f"{part} " for part in parts]


def parse_text_tool_calls(text: str) -> list[dict[str, Any]]:
    trimmed = str(text or "").strip()
    if not trimmed:
        return []
    candidates = [line.strip() for line in trimmed.splitlines() if line.strip()] if "\n" in trimmed else [trimmed]
    parsed: list[dict[str, Any]] = []
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except Exception:
            continue
        if isinstance(payload, dict) and isinstance(payload.get("name"), str) and isinstance(payload.get("arguments"), dict):
            parsed.append({"id": str(uuid.uuid4()), "name": payload["name"], "args": payload["arguments"]})
    return parsed


def build_tool_replay_message(name: str, args: dict[str, Any], result: str, status: str) -> SystemMessage:
    return SystemMessage(
        content=(
            "Tool execution replay for provider compatibility.\n"
            f"Tool name: {name}\n"
            f"Arguments JSON: {json.dumps(args, ensure_ascii=False)}\n"
            f"Tool status: {status}\n"
            "Tool result:\n"
            f"{result}\n"
            "Continue the task using this result and do not rely on OpenAI tool-role replay."
        )
    )


def decode_terminal_payload(raw: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(str(raw or ""))
    except Exception:
        return None
    if not isinstance(payload, dict) or "terminalId" not in payload:
        return None
    return payload


def decode_terminal_tool_payload(raw: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(str(raw or ""))
    except Exception:
        return None
    if not isinstance(payload, dict) or payload.get("kind") != "terminal_tool":
        return None
    return payload


class RuntimeEngine:
    def __init__(self, dependencies: RuntimeDependencies | None = None) -> None:
        if dependencies is None:
            terminal_manager = TerminalManager()
            dependencies = RuntimeDependencies(
                model_factory=build_chat_model,
                provider_resolver=resolve_provider,
                tool_registry_factory=lambda workspace_root, session_id, run_id: create_default_tool_registry(
                    workspace_root,
                    session_id=session_id,
                    run_id=run_id,
                    terminal_manager=terminal_manager,
                ),
                state_store=RunStateStore(),
                terminal_manager=terminal_manager,
            )
        self.deps = dependencies
        self.graph = self._create_graph()

    def capabilities_payload(self, workspace_root: str, session_id: str | None = None) -> list[dict]:
        return self.deps.tool_registry_factory(workspace_root, session_id, None).module_payloads()

    def _materialize_tool_result(
        self,
        run_id: str,
        tool_name: str,
        tool_id: str,
        raw_result: str,
    ) -> tuple[list[dict[str, Any]], str, bool]:
        if tool_name != "run_terminal":
            terminal_tool_payload = decode_terminal_tool_payload(raw_result)
            if terminal_tool_payload:
                events = list(terminal_tool_payload.get("events", []))
                terminal = terminal_tool_payload.get("terminal", {}) if isinstance(terminal_tool_payload.get("terminal"), dict) else {}
                snapshot = terminal_tool_payload.get("snapshot", {}) if isinstance(terminal_tool_payload.get("snapshot"), dict) else {}
                matched = terminal_tool_payload.get("matched")
                tail = str(terminal_tool_payload.get("tail") or terminal.get("tail") or snapshot.get("tail") or "")
                preview_lines = [
                    f"Tool: {terminal_tool_payload.get('tool', tool_name)}",
                    f"Terminal: {terminal.get('terminalId') or snapshot.get('terminalId') or ''}",
                ]
                if terminal.get("owner"):
                    preview_lines.append(f"Owner: {terminal['owner']}")
                if matched is not None:
                    preview_lines.append(f"Matched: {matched}")
                if tail:
                    preview_lines.extend(["Tail:", tail[-2000:]])
                formatted = "\n".join(line for line in preview_lines if line)
                events.append(
                    {
                        "type": "tool_result",
                        "runId": run_id,
                        "actionId": tool_id,
                        "name": tool_name,
                        "ok": True,
                        "preview": formatted[:400],
                    }
                )
                return events, formatted, True
            return (
                [
                    {
                        "type": "tool_result",
                        "runId": run_id,
                        "actionId": tool_id,
                        "name": tool_name,
                        "ok": True,
                        "preview": raw_result[:400],
                    }
                ],
                raw_result,
                True,
            )
        payload = decode_terminal_payload(raw_result)
        if not payload:
            return (
                [
                    {
                        "type": "tool_result",
                        "runId": run_id,
                        "actionId": tool_id,
                        "name": tool_name,
                        "ok": False,
                        "preview": raw_result[:400],
                    },
                    run_diagnostic(run_id, "terminal_payload_invalid", "Terminal tool returned an invalid payload.", level="warn"),
                ],
                raw_result,
                False,
            )
        terminal_id = str(payload.get("terminalId") or uuid.uuid4())
        command = str(payload.get("command") or "")
        cwd = str(payload.get("cwd") or "")
        stdout = str(payload.get("stdout") or "")
        stderr = str(payload.get("stderr") or "")
        blocked = bool(payload.get("blocked"))
        exit_code = int(payload.get("exitCode") or 0)
        output = "\n".join(part for part in [stdout.strip(), stderr.strip()] if part).strip()
        events = [terminal_opened(run_id, terminal_id, command, cwd)]
        if blocked:
            events.append(run_diagnostic(run_id, "terminal_command_blocked", stderr or "Terminal command blocked by policy.", level="warn"))
            events.append(terminal_error(run_id, terminal_id, stderr or "Terminal command blocked by policy."))
        else:
            events.append(terminal_exit(run_id, terminal_id, exit_code, output[:8000]))
            if exit_code != 0:
                events.append(run_diagnostic(run_id, "terminal_nonzero_exit", f"Terminal command exited with code {exit_code}.", level="warn"))
        formatted = "\n".join(
            line
            for line in [
                f"Command: {command}",
                f"CWD: {cwd}",
                f"Exit code: {exit_code}",
                "Output:",
                output or "(no output)",
            ]
            if line
        )
        events.append(
            {
                "type": "tool_result",
                "runId": run_id,
                "actionId": tool_id,
                "name": tool_name,
                "ok": not blocked and exit_code == 0,
                "preview": formatted[:400],
            }
        )
        return events, formatted, (not blocked and exit_code == 0)

    async def _stream_result(self, run_id: str, result: AgentState):
        for event in result.get("tool_events", []):
            yield event
        if result.get("pending_approval"):
            approval = result["pending_approval"]
            self.deps.state_store.save("approvals", approval["approvalId"], {**self._serialize_snapshot(result)})
            yield run_state(run_id, "awaiting_approval")
            yield approval_required(
                run_id=run_id,
                approval_id=approval["approvalId"],
                name=approval["name"],
                arguments=json.dumps(approval["arguments"], ensure_ascii=False),
                risk_level=approval["riskLevel"],
            )
            yield done()
            return
        if result.get("pending_clarification"):
            clarification = result["pending_clarification"]
            self.deps.state_store.save("clarifications", clarification["clarificationId"], {**self._serialize_snapshot(result)})
            yield run_state(run_id, "awaiting_clarification")
            yield clarification_required(
                run_id=run_id,
                clarification_id=clarification["clarificationId"],
                question=clarification["question"],
                options=clarification["options"],
            )
            yield done()
            return
        final_text = str(result.get("final_text", "") or "").strip() or "(empty response)"
        yield run_phase(run_id, "finish")
        for piece in split_for_streaming(final_text):
            yield token(piece)
            await asyncio.sleep(0)
        yield run_state(run_id, "completed")
        yield done()
        return

    def _create_graph(self):
        def agent(state: AgentState) -> AgentState:
            registry = self.deps.tool_registry_factory(state["workspace_root"], state["session_id"], state["run_id"])
            model = self.deps.model_factory(state.get("profile"), state.get("model"))
            provider = self.deps.provider_resolver(state.get("profile"), state.get("model"))
            response = model.bind_tools(registry.enabled_tools()).invoke(state["messages"])
            final_text = response.content if isinstance(response.content, str) else str(response.content)
            return {
                "messages": [response],
                "provider_mode": provider.mode,
                "final_text": final_text,
            }

        def tools(state: AgentState) -> AgentState:
            registry = self.deps.tool_registry_factory(state["workspace_root"], state["session_id"], state["run_id"])
            last_message = state["messages"][-1]
            emitted_messages: list[BaseMessage] = []
            tool_events: list[dict[str, Any]] = []
            pending_approval = None
            pending_clarification = None
            tool_calls = []

            if isinstance(last_message, AIMessage):
                tool_calls = list(last_message.tool_calls or [])
                if not tool_calls:
                    tool_calls = parse_text_tool_calls(last_message.content)

            for call in tool_calls:
                tool_name = str(call["name"])
                tool_args = call.get("args", {})
                tool_id = str(call.get("id") or uuid.uuid4())
                registered = registry.get(tool_name)
                if not registered or not registered.enabled:
                    continue
                tool_events.append(
                    {
                        "type": "tool_call",
                        "runId": state["run_id"],
                        "actionId": tool_id,
                        "name": tool_name,
                        "arguments": json.dumps(tool_args, ensure_ascii=False),
                        "riskLevel": registered.risk_level,
                    }
                )
                if tool_name == "request_clarification":
                    options = []
                    for key in ("option_a", "option_b", "option_c"):
                        value = tool_args.get(key)
                        if isinstance(value, str) and value.strip():
                            options.append({"label": value.strip(), "value": value.strip()})
                    pending_clarification = {
                        "clarificationId": str(uuid.uuid4()),
                        "runId": state["run_id"],
                        "question": str(tool_args.get("question", "Could you clarify?")).strip() or "Could you clarify?",
                        "options": options,
                    }
                    break
                if registered.risk_level == "risky" and state["policy_profile"] != "always_allow":
                    pending_approval = {
                        "approvalId": str(uuid.uuid4()),
                        "runId": state["run_id"],
                        "name": tool_name,
                        "arguments": tool_args,
                        "riskLevel": registered.risk_level,
                        "toolCallId": tool_id,
                    }
                    break
                result = str(registered.tool.invoke(tool_args))
                result_events, message_result, _ok = self._materialize_tool_result(state["run_id"], tool_name, tool_id, result)
                tool_events.extend(result_events)
                if state["provider_mode"] == "textual_replay":
                    emitted_messages.append(build_tool_replay_message(tool_name, tool_args, message_result, "succeeded"))
                else:
                    emitted_messages.append(ToolMessage(content=message_result, tool_call_id=tool_id))

            return {
                "messages": emitted_messages,
                "pending_approval": pending_approval,
                "pending_clarification": pending_clarification,
                "tool_events": list(state.get("tool_events", [])) + tool_events,
            }

        def route_after_agent(state: AgentState) -> str:
            last_message = state["messages"][-1]
            if isinstance(last_message, AIMessage):
                has_calls = (last_message.tool_calls and len(last_message.tool_calls) > 0) or parse_text_tool_calls(last_message.content)
                if has_calls:
                    return "tools"
            return END

        def route_after_tools(state: AgentState) -> str:
            if state.get("pending_approval") or state.get("pending_clarification"):
                return END
            return "agent"

        graph = StateGraph(AgentState)
        graph.add_node("agent", agent)
        graph.add_node("tools", tools)
        graph.set_entry_point("agent")
        graph.add_conditional_edges("agent", route_after_agent, {"tools": "tools", END: END})
        graph.add_conditional_edges("tools", route_after_tools, {"agent": "agent", END: END})
        return graph.compile()

    def _initial_state(self, request: SidecarChatRequest, run_id: str, messages: list[BaseMessage]) -> AgentState:
        provider = self.deps.provider_resolver(request.profile, request.model)
        return {
            "messages": messages,
            "session_id": request.sessionId,
            "run_id": run_id,
            "workspace_root": request.workspaceRoot or "",
            "profile": request.profile,
            "model": request.model,
            "provider_mode": provider.mode,
            "policy_profile": request.policyProfile,
            "final_text": "",
            "pending_approval": None,
            "pending_clarification": None,
            "tool_events": [],
        }

    def _serialize_snapshot(self, state: AgentState) -> dict[str, Any]:
        return {
            "run_id": state["run_id"],
            "workspace_root": state["workspace_root"],
            "profile": state.get("profile"),
            "model": state.get("model"),
            "provider_mode": state["provider_mode"],
            "policy_profile": state["policy_profile"],
            "messages": serialize_messages(state["messages"]),
            "session_id": state["session_id"],
            "final_text": state.get("final_text", ""),
            "pending_approval": state.get("pending_approval"),
            "pending_clarification": state.get("pending_clarification"),
        }

    def _deserialize_snapshot(self, payload: dict[str, Any]) -> AgentState:
        return {
            "messages": deserialize_messages(payload["messages"]),
            "session_id": payload["session_id"],
            "run_id": payload["run_id"],
            "workspace_root": payload["workspace_root"],
            "profile": payload.get("profile"),
            "model": payload.get("model"),
            "provider_mode": payload["provider_mode"],
            "policy_profile": payload["policy_profile"],
            "final_text": payload.get("final_text", ""),
            "pending_approval": payload.get("pending_approval"),
            "pending_clarification": payload.get("pending_clarification"),
            "tool_events": [],
        }

    async def stream_chat(self, request: SidecarChatRequest, messages: list[BaseMessage]):
        run_id = request.runId or str(uuid.uuid4())
        state = self._initial_state(request, run_id, messages)
        yield run_state(run_id, "running")
        yield run_phase(run_id, "planning")
        yield run_phase(run_id, "execute")
        yield run_diagnostic(run_id, "provider_mode", f"Provider mode: {state['provider_mode']}.")
        yield run_diagnostic(run_id, "tool_routing_mode", "Tool selection is performed in the main model turn (single-pass).")
        try:
            result = self.graph.invoke(state)
        except Exception as exc:
            yield error_event(str(exc))
            yield run_state(run_id, "failed")
            yield done()
            return
        async for event in self._stream_result(run_id, result):
            yield event

    async def stream_approval_decision(self, request: ApprovalDecisionRequest):
        snapshot = self.deps.state_store.pop("approvals", request.approval_id)
        if not snapshot:
            raise KeyError("Approval not found")
        state = self._deserialize_snapshot(snapshot)
        pending = state["pending_approval"]
        run_id = state["run_id"]
        yield approval_decision(run_id, request.approval_id, request.decision)
        if request.decision == "approved":
            registry = self.deps.tool_registry_factory(state["workspace_root"], state["session_id"], state["run_id"])
            registered = registry.get(pending["name"])
            if not registered:
                yield error_event("Approved tool is no longer available")
                yield run_state(run_id, "failed")
                yield done()
                return
            result = str(registered.tool.invoke(pending["arguments"]))
            result_events, message_result, _ok = self._materialize_tool_result(run_id, pending["name"], pending["toolCallId"], result)
            for event in result_events:
                yield event
            if state["provider_mode"] == "textual_replay":
                state["messages"] = state["messages"] + [build_tool_replay_message(pending["name"], pending["arguments"], message_result, "succeeded")]
            else:
                state["messages"] = state["messages"] + [ToolMessage(content=message_result, tool_call_id=pending["toolCallId"])]
        else:
            rejection = (
                build_tool_replay_message(pending["name"], pending["arguments"], "Rejected by user approval policy.", "rejected")
                if state["provider_mode"] == "textual_replay"
                else SystemMessage(content=f"Tool {pending['name']} was rejected by the user. Continue without this tool.")
            )
            state["messages"] = state["messages"] + [rejection]
        state["pending_approval"] = None
        state["pending_clarification"] = None
        state["final_text"] = ""
        yield run_state(run_id, "running")
        yield run_phase(run_id, "repair")
        try:
            result = self.graph.invoke(state)
        except Exception as exc:
            yield error_event(str(exc))
            yield run_state(run_id, "failed")
            yield done()
            return
        async for event in self._stream_result(run_id, result):
            yield event

    async def stream_clarification_decision(self, request: ClarificationDecisionRequest):
        snapshot = self.deps.state_store.pop("clarifications", request.clarification_id)
        if not snapshot:
            raise KeyError("Clarification not found")
        state = self._deserialize_snapshot(snapshot)
        run_id = state["run_id"]
        state["messages"] = state["messages"] + [HumanMessage(content=request.answer)]
        state["pending_approval"] = None
        state["pending_clarification"] = None
        state["final_text"] = ""
        yield run_state(run_id, "running")
        yield run_phase(run_id, "execute", detail="Clarification received")
        try:
            result = self.graph.invoke(state)
        except Exception as exc:
            yield error_event(str(exc))
            yield run_state(run_id, "failed")
            yield done()
            return
        async for event in self._stream_result(run_id, result):
            yield event
