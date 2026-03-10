from __future__ import annotations

import asyncio
import uuid
from typing import Any, TypedDict

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import Annotated

from continue_better_py.events import done, run_phase, run_state, sse, token
from continue_better_py.providers import build_chat_model, list_available_models, resolve_provider
from continue_better_py.schemas import ChatMessage, SidecarChatRequest
from continue_better_py.settings import ensure_runtime_dirs, get_settings
from continue_better_py.tooling import build_tools


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    workspace_root: str
    profile: str | None
    model: str | None
    provider_mode: str
    final_text: str


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
        }

    def tool_node(state: AgentState) -> AgentState:
        tools = {tool.name: tool for tool in build_tools(state["workspace_root"])}
        last_message = state["messages"][-1]
        emitted = []
        if isinstance(last_message, AIMessage):
            for call in last_message.tool_calls:
                tool = tools.get(call["name"])
                if not tool:
                    continue
                result = tool.invoke(call["args"])
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
        return {"messages": emitted}

    def route_after_agent(state: AgentState) -> str:
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", route_after_agent, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
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
                }
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
                "workspace_root": workspace_root,
                "profile": request.profile,
                "model": request.model,
                "provider_mode": resolve_provider(profile_name=request.profile, requested_model=request.model).mode,
                "final_text": "",
            }

            yield sse(run_phase(run_id, "execute"))
            result = graph.invoke(state)
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

    return app


def split_for_streaming(text: str) -> list[str]:
    parts = text.split()
    if not parts:
        return [text]
    return [f"{part} " for part in parts]
