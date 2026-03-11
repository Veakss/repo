from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, TypedDict

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
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
    sse,
    token,
)
from continue_better_py.providers import build_chat_model, list_available_models, resolve_provider
from continue_better_py.schemas import (
    ApprovalDecisionRequest,
    ChatMessage,
    ClarificationDecisionRequest,
    SidecarChatRequest,
)
from continue_better_py.settings import ensure_runtime_dirs, get_settings
from continue_better_py.tooling import build_tool_lookup, build_tools


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


def create_graph():
    def agent(state: AgentState) -> AgentState:
        tools = build_tools(state["workspace_root"])
        model = build_chat_model(profile_name=state.get("profile"), requested_model=state.get("model"))
        provider = resolve_provider(profile_name=state.get("profile"), requested_model=state.get("model"))
        bound_model = model.bind_tools(tools)
        response = bound_model.invoke(state["messages"])
        return {
            "messages": [response],
            "provider_mode": provider.mode,
            "final_text": response.content if isinstance(response.content, str) else str(response.content),
            "pending_approval": None,
            "pending_clarification": None,
        }

    def tool_node(state: AgentState) -> AgentState:
        tools = build_tool_lookup(state["workspace_root"])
        last_message = state["messages"][-1]
        emitted = []
        pending_approval = None
        pending_clarification = None
        if isinstance(last_message, AIMessage):
            for call in last_message.tool_calls:
                definition = tools.get(call["name"])
                if not definition:
                    continue
                if call["name"] == "request_clarification":
                    options = []
                    for key in ("option_a", "option_b", "option_c"):
                        value = call["args"].get(key)
                        if isinstance(value, str) and value.strip():
                            options.append({"label": value.strip(), "value": value.strip()})
                    pending_clarification = {
                        "clarificationId": str(uuid.uuid4()),
                        "runId": state["run_id"],
                        "question": str(call["args"].get("question", "Could you clarify?")).strip() or "Could you clarify?",
                        "options": options,
                    }
                    break
                if definition.risk_level == "risky" and state["policy_profile"] != "always_allow":
                    pending_approval = {
                        "approvalId": str(uuid.uuid4()),
                        "runId": state["run_id"],
                        "name": call["name"],
                        "arguments": call["args"],
                        "riskLevel": definition.risk_level,
                    }
                    break
                result = definition.tool.invoke(call["args"])
                if state["provider_mode"] == "textual_replay":
                    emitted.append(
                        SystemMessage(
                            content=(
                                "Tool execution replay for provider compatibility.\n"
                                f"Tool name: {call['name']}\n"
                                f"Arguments: {call['args']}\n"
                                f"Tool result:\n{result}\n"
                                "Continue the task using this result."
                            )
                        )
                    )
                else:
                    emitted.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
        return {
            "messages": emitted,
            "pending_approval": pending_approval,
            "pending_clarification": pending_clarification,
        }

    def route_after_agent(state: AgentState) -> str:
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"
        return END

    def route_after_tools(state: AgentState) -> str:
        if state.get("pending_approval") or state.get("pending_clarification"):
            return END
        return "agent"

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", route_after_agent, {"tools": "tools", END: END})
    graph.add_conditional_edges("tools", route_after_tools, {"agent": "agent", END: END})
    return graph.compile()


def to_langchain_message(message: ChatMessage):
    if message.role == "assistant":
        return AIMessage(content=message.content)
    if message.role == "system":
        return SystemMessage(content=message.content)
    return HumanMessage(content=message.content)


def create_sidecar_app() -> FastAPI:
    ensure_runtime_dirs()
    settings = get_settings()
    app = FastAPI(title="Continue Better Python Sidecar", version="0.1.0")
    graph = create_graph()
    pending_approvals: dict[str, dict[str, Any]] = {}
    pending_clarifications: dict[str, dict[str, Any]] = {}

    @app.get("/health")
    @app.get("/v1/health")
    def health() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/v1/capabilities")
    def capabilities() -> dict[str, Any]:
        provider = resolve_provider()
        return {
            "appActions": settings.enable_app_actions,
            "webSearch": settings.enable_web_search,
            "rag": settings.enable_rag,
            "clarification": settings.enable_tool_clarification,
            "interactiveTerminal": settings.enable_terminal_tools,
            "providerMode": provider.mode,
            "toolModules": [
                {
                    "id": "files",
                    "label": "Files",
                    "category": "core",
                    "version": "0.1.0",
                    "available": settings.enable_file_tools,
                    "enabled": settings.enable_file_tools,
                    "active": False,
                    "status": "idle",
                    "toolNames": ["list_directory", "read_file", "write_file"],
                    "enabledToolNames": ["list_directory", "read_file", "write_file"] if settings.enable_file_tools else [],
                },
                {
                    "id": "clarification",
                    "label": "Clarification",
                    "category": "control",
                    "version": "0.1.0",
                    "available": settings.enable_tool_clarification,
                    "enabled": settings.enable_tool_clarification,
                    "active": False,
                    "status": "idle",
                    "toolNames": ["request_clarification"],
                    "enabledToolNames": ["request_clarification"] if settings.enable_tool_clarification else [],
                },
            ],
        }

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return list_available_models()

    @app.post("/v1/chat/stream")
    async def chat_stream(request: SidecarChatRequest):
        async def event_generator():
            run_id = request.runId or str(uuid.uuid4())
            workspace_root = request.workspaceRoot or str(settings.resolved_workspace_root)

            yield sse(run_state(run_id, "running"))
            yield sse(run_phase(run_id, "planning"))

            state = {
                "messages": [to_langchain_message(message) for message in request.messages],
                "run_id": run_id,
                "workspace_root": workspace_root,
                "profile": request.profile,
                "model": request.model,
                "provider_mode": resolve_provider(profile_name=request.profile, requested_model=request.model).mode,
                "policy_profile": request.policyProfile,
                "final_text": "",
                "pending_approval": None,
                "pending_clarification": None,
            }

            yield sse(run_phase(run_id, "execute"))
            try:
                result = graph.invoke(state)
            except Exception as exc:
                yield sse(error_event(str(exc)))
                yield sse(run_state(run_id, "failed"))
                yield sse(done())
                return

            approval = result.get("pending_approval")
            if approval:
                pending_approvals[approval["approvalId"]] = {
                    "state": result,
                    "profile": request.profile,
                    "model": request.model,
                }
                yield sse(run_state(run_id, "awaiting_approval"))
                yield sse(
                    approval_required(
                        run_id=run_id,
                        approval_id=approval["approvalId"],
                        name=approval["name"],
                        arguments=json.dumps(approval["arguments"], ensure_ascii=False),
                        risk_level=approval["riskLevel"],
                    )
                )
                yield sse(done())
                return

            clarification = result.get("pending_clarification")
            if clarification:
                pending_clarifications[clarification["clarificationId"]] = {
                    "state": result,
                    "profile": request.profile,
                    "model": request.model,
                }
                yield sse(run_state(run_id, "awaiting_clarification"))
                yield sse(
                    clarification_required(
                        run_id=run_id,
                        clarification_id=clarification["clarificationId"],
                        question=clarification["question"],
                        options=clarification["options"],
                    )
                )
                yield sse(done())
                return

            final_text = str(result.get("final_text", "") or "").strip()

            if not final_text:
                messages = result.get("messages", [])
                if messages:
                    final = messages[-1]
                    final_text = getattr(final, "content", "") if hasattr(final, "content") else str(final)

            yield sse(run_phase(run_id, "finish"))
            for piece in split_for_streaming(final_text or "(empty response)"):
                yield sse(token(piece))
                await asyncio.sleep(0)
            yield sse(run_state(run_id, "completed"))
            yield sse(done())

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    @app.post("/v1/approvals/respond/stream")
    async def approvals_stream(request: ApprovalDecisionRequest):
        snapshot = pending_approvals.pop(request.approval_id, None)
        if not snapshot:
            raise HTTPException(status_code=404, detail="Approval not found")

        async def event_generator():
            state = snapshot["state"]
            run_id = state["run_id"]
            approval = state["pending_approval"]
            yield sse(approval_decision(run_id, request.approval_id, request.decision))
            if request.decision == "approved":
                tool_lookup = build_tool_lookup(state["workspace_root"])
                definition = tool_lookup.get(approval["name"])
                if not definition:
                    yield sse(error_event("Approved tool is no longer available"))
                    yield sse(run_state(run_id, "failed"))
                    yield sse(done())
                    return
                result = definition.tool.invoke(approval["arguments"])
                if state["provider_mode"] == "textual_replay":
                    resumed_messages = state["messages"] + [
                        SystemMessage(
                            content=(
                                "Tool execution replay for provider compatibility.\n"
                                f"Tool name: {approval['name']}\n"
                                f"Arguments: {approval['arguments']}\n"
                                f"Tool result:\n{result}\n"
                                "Continue the task using this result."
                            )
                        )
                    ]
                else:
                    resumed_messages = state["messages"] + [SystemMessage(content=f"Tool {approval['name']} executed successfully.\n{result}")]
            else:
                resumed_messages = state["messages"] + [
                    SystemMessage(
                        content=(
                            f"Tool {approval['name']} was rejected by the user. "
                            "Continue without this tool unless a different approach is necessary."
                        )
                    )
                ]

            resumed_state = {
                **state,
                "messages": resumed_messages,
                "pending_approval": None,
                "pending_clarification": None,
                "final_text": "",
            }
            yield sse(run_state(run_id, "running"))
            yield sse(run_phase(run_id, "repair"))
            try:
                result_state = graph.invoke(resumed_state)
            except Exception as exc:
                yield sse(error_event(str(exc)))
                yield sse(run_state(run_id, "failed"))
                yield sse(done())
                return
            final_text = str(result_state.get("final_text", "") or "").strip() or "(empty response)"
            yield sse(run_phase(run_id, "finish"))
            for piece in split_for_streaming(final_text):
                yield sse(token(piece))
                await asyncio.sleep(0)
            yield sse(run_state(run_id, "completed"))
            yield sse(done())

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    @app.post("/v1/clarifications/respond/stream")
    async def clarifications_stream(request: ClarificationDecisionRequest):
        snapshot = pending_clarifications.pop(request.clarification_id, None)
        if not snapshot:
            raise HTTPException(status_code=404, detail="Clarification not found")

        async def event_generator():
            state = snapshot["state"]
            run_id = state["run_id"]
            resumed_state = {
                **state,
                "messages": state["messages"] + [HumanMessage(content=request.answer)],
                "pending_clarification": None,
                "pending_approval": None,
                "final_text": "",
            }
            yield sse(run_state(run_id, "running"))
            yield sse(run_phase(run_id, "execute", detail="Clarification received"))
            try:
                result_state = graph.invoke(resumed_state)
            except Exception as exc:
                yield sse(error_event(str(exc)))
                yield sse(run_state(run_id, "failed"))
                yield sse(done())
                return
            final_text = str(result_state.get("final_text", "") or "").strip() or "(empty response)"
            yield sse(run_phase(run_id, "finish"))
            for piece in split_for_streaming(final_text):
                yield sse(token(piece))
                await asyncio.sleep(0)
            yield sse(run_state(run_id, "completed"))
            yield sse(done())

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    return app


def split_for_streaming(text: str) -> list[str]:
    parts = text.split()
    if not parts:
        return [text]
    return [f"{part} " for part in parts]
