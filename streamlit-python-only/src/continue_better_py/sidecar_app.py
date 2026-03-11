from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from continue_better_py.events import sse
from continue_better_py.providers import list_available_models, resolve_provider
from continue_better_py.runtime import RuntimeEngine
from continue_better_py.schemas import ApprovalDecisionRequest, ChatMessage, ClarificationDecisionRequest, SidecarChatRequest
from continue_better_py.settings import ensure_runtime_dirs, get_settings


def to_langchain_message(message: ChatMessage):
    if message.role == "assistant":
        return AIMessage(content=message.content)
    if message.role == "system":
        return SystemMessage(content=message.content)
    return HumanMessage(content=message.content)


def create_sidecar_app(runtime: RuntimeEngine | None = None) -> FastAPI:
    ensure_runtime_dirs()
    settings = get_settings()
    app = FastAPI(title="Continue Better Python Sidecar", version="0.1.0")
    runtime = runtime or RuntimeEngine()

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
            "toolModules": runtime.capabilities_payload(str(settings.resolved_workspace_root)),
        }

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return list_available_models()

    @app.post("/v1/chat/stream")
    async def chat_stream(request: SidecarChatRequest):
        async def event_generator():
            workspace_root = request.workspaceRoot or str(settings.resolved_workspace_root)
            effective_request = request.model_copy(update={"workspaceRoot": workspace_root})
            async for event in runtime.stream_chat(
                effective_request,
                [to_langchain_message(message) for message in effective_request.messages],
            ):
                yield sse(event)

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    @app.post("/v1/approvals/respond/stream")
    async def approvals_stream(request: ApprovalDecisionRequest):
        async def event_generator():
            try:
                async for event in runtime.stream_approval_decision(request):
                    yield sse(event)
            except KeyError:
                raise HTTPException(status_code=404, detail="Approval not found")

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    @app.post("/v1/clarifications/respond/stream")
    async def clarifications_stream(request: ClarificationDecisionRequest):
        async def event_generator():
            try:
                async for event in runtime.stream_clarification_decision(request):
                    yield sse(event)
            except KeyError:
                raise HTTPException(status_code=404, detail="Clarification not found")

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    return app
