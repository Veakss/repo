from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass
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
    run_state,
    token,
)
from continue_better_py.providers import ResolvedProvider, build_chat_model, resolve_provider
from continue_better_py.run_state import RunStateStore, deserialize_messages, serialize_messages
from continue_better_py.schemas import ApprovalDecisionRequest, ClarificationDecisionRequest, SidecarChatRequest
from continue_better_py.tool_registry import ToolRegistry, create_default_tool_registry


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
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
    tool_registry_factory: Callable[[str], ToolRegistry]
    state_store: RunStateStore


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


class RuntimeEngine:
    def __init__(self, dependencies: RuntimeDependencies | None = None) -> None:
        self.deps = dependencies or RuntimeDependencies(
            model_factory=build_chat_model,
            provider_resolver=resolve_provider,
            tool_registry_factory=create_default_tool_registry,
            state_store=RunStateStore(),
        )
        self.graph = self._create_graph()

    def capabilities_payload(self, workspace_root: str) -> list[dict]:
        return self.deps.tool_registry_factory(workspace_root).module_payloads()

    def _create_graph(self):
        def agent(state: AgentState) -> AgentState:
            registry = self.deps.tool_registry_factory(state["workspace_root"])
            model = self.deps.model_factory(state.get("profile"), state.get("model"))
            provider = self.deps.provider_resolver(state.get("profile"), state.get("model"))
            response = model.bind_tools(registry.enabled_tools()).invoke(state["messages"])
            final_text = response.content if isinstance(response.content, str) else str(response.content)
            return {
                "messages": [response],
                "provider_mode": provider.mode,
                "final_text": final_text,
                "pending_approval": None,
                "pending_clarification": None,
                "tool_events": [],
            }

        def tools(state: AgentState) -> AgentState:
            registry = self.deps.tool_registry_factory(state["workspace_root"])
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
                tool_events.append(
                    {
                        "type": "tool_result",
                        "runId": state["run_id"],
                        "actionId": tool_id,
                        "name": tool_name,
                        "ok": True,
                        "preview": result[:400],
                    }
                )
                if state["provider_mode"] == "textual_replay":
                    emitted_messages.append(build_tool_replay_message(tool_name, tool_args, result, "succeeded"))
                else:
                    emitted_messages.append(ToolMessage(content=result, tool_call_id=tool_id))

            return {
                "messages": emitted_messages,
                "pending_approval": pending_approval,
                "pending_clarification": pending_clarification,
                "tool_events": tool_events,
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
            "final_text": state.get("final_text", ""),
            "pending_approval": state.get("pending_approval"),
            "pending_clarification": state.get("pending_clarification"),
        }

    def _deserialize_snapshot(self, payload: dict[str, Any]) -> AgentState:
        return {
            "messages": deserialize_messages(payload["messages"]),
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
        try:
            result = self.graph.invoke(state)
        except Exception as exc:
            yield error_event(str(exc))
            yield run_state(run_id, "failed")
            yield done()
            return
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

    async def stream_approval_decision(self, request: ApprovalDecisionRequest):
        snapshot = self.deps.state_store.pop("approvals", request.approval_id)
        if not snapshot:
            raise KeyError("Approval not found")
        state = self._deserialize_snapshot(snapshot)
        pending = state["pending_approval"]
        run_id = state["run_id"]
        yield approval_decision(run_id, request.approval_id, request.decision)
        if request.decision == "approved":
            registry = self.deps.tool_registry_factory(state["workspace_root"])
            registered = registry.get(pending["name"])
            if not registered:
                yield error_event("Approved tool is no longer available")
                yield run_state(run_id, "failed")
                yield done()
                return
            result = str(registered.tool.invoke(pending["arguments"]))
            yield {
                "type": "tool_result",
                "runId": run_id,
                "actionId": pending["toolCallId"],
                "name": pending["name"],
                "ok": True,
                "preview": result[:400],
            }
            if state["provider_mode"] == "textual_replay":
                state["messages"] = state["messages"] + [build_tool_replay_message(pending["name"], pending["arguments"], result, "succeeded")]
            else:
                state["messages"] = state["messages"] + [ToolMessage(content=result, tool_call_id=pending["toolCallId"])]
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
