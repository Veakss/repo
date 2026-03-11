from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from continue_better_py.events import sse
from continue_better_py.providers import list_available_models, resolve_provider
from continue_better_py.rag import RagService
from continue_better_py.runtime import RuntimeEngine
from continue_better_py.schemas import (
    ApprovalDecisionRequest,
    ChatMessage,
    ClarificationDecisionRequest,
    RagImportRequest,
    RagLookupRequest,
    RagMemoryAppendRequest,
    RagProfileCreateRequest,
    RagProfileRenameRequest,
    RagSessionMemoryPatchRequest,
    SidecarChatRequest,
)
from continue_better_py.settings import ensure_runtime_dirs, get_settings


def to_langchain_message(message: ChatMessage):
    if message.role == "assistant":
        return AIMessage(content=message.content)
    if message.role == "system":
        return SystemMessage(content=message.content)
    return HumanMessage(content=message.content)


def create_sidecar_app(runtime: RuntimeEngine | None = None, rag_service: RagService | None = None) -> FastAPI:
    ensure_runtime_dirs()
    settings = get_settings()
    app = FastAPI(title="Continue Better Python Sidecar", version="0.1.0")
    runtime = runtime or RuntimeEngine()
    rag_service = rag_service or RagService(settings=settings)

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

    @app.get("/v1/rag/profiles")
    def rag_list_profiles() -> dict[str, Any]:
        return {"profiles": rag_service.list_profiles()}

    @app.post("/v1/rag/profiles")
    def rag_create_profile(request: RagProfileCreateRequest) -> dict[str, Any]:
        return {"ok": True, "profiles": rag_service.create_profile(request.name)}

    @app.patch("/v1/rag/profiles/{profile_name}")
    def rag_rename_profile(profile_name: str, request: RagProfileRenameRequest) -> dict[str, Any]:
        try:
            return {"ok": True, "profiles": rag_service.rename_profile(profile_name, request.new_name)}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="RAG profile not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.delete("/v1/rag/profiles/{profile_name}")
    def rag_delete_profile(profile_name: str) -> dict[str, Any]:
        return {"ok": True, "profiles": rag_service.delete_profile(profile_name)}

    @app.post("/v1/rag/session/{session_id}/files/import")
    def rag_import_session_file(session_id: str, request: RagImportRequest) -> dict[str, Any]:
        try:
            return rag_service.import_file("session", session_id, request.path, request.path)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="RAG source file not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/rag/profiles/{profile_name}/files/import")
    def rag_import_profile_file(profile_name: str, request: RagImportRequest) -> dict[str, Any]:
        try:
            return rag_service.import_file("profile", profile_name, request.path, request.path)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="RAG source file not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/v1/rag/session/{session_id}/files")
    def rag_get_session_files(session_id: str) -> dict[str, Any]:
        return rag_service.list_files("session", session_id)

    @app.get("/v1/rag/profiles/{profile_name}/files")
    def rag_get_profile_files(profile_name: str) -> dict[str, Any]:
        return rag_service.list_files("profile", profile_name)

    @app.delete("/v1/rag/session/{session_id}/files/{file_id}")
    def rag_delete_session_file(session_id: str, file_id: str) -> dict[str, Any]:
        return rag_service.delete_file("session", session_id, file_id)

    @app.delete("/v1/rag/profiles/{profile_name}/files/{file_id}")
    def rag_delete_profile_file(profile_name: str, file_id: str) -> dict[str, Any]:
        return rag_service.delete_file("profile", profile_name, file_id)

    @app.post("/v1/rag/session/{session_id}/index/jobs")
    def rag_enqueue_session_index_job(session_id: str) -> dict[str, Any]:
        return {"job": rag_service.enqueue_index_job("session", session_id)}

    @app.post("/v1/rag/profiles/{profile_name}/index/jobs")
    def rag_enqueue_profile_index_job(profile_name: str) -> dict[str, Any]:
        return {"job": rag_service.enqueue_index_job("profile", profile_name)}

    @app.get("/v1/rag/index/jobs")
    def rag_list_index_jobs(limit: int | None = Query(None, ge=1, le=200)) -> dict[str, Any]:
        return rag_service.list_jobs(limit or 50)

    @app.get("/v1/rag/index/jobs/{job_id}")
    def rag_get_index_job(job_id: str) -> dict[str, Any]:
        job = rag_service.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="RAG indexing job not found")
        return {"job": job}

    @app.post("/v1/rag/index/jobs/{job_id}/retry")
    def rag_retry_index_job(job_id: str) -> dict[str, Any]:
        try:
            return {"job": rag_service.retry_job(job_id)}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="RAG indexing job not found") from exc

    @app.get("/v1/rag/session/{session_id}/memory")
    def rag_get_session_memory(session_id: str, limit: int | None = Query(None, ge=1, le=300)) -> dict[str, Any]:
        return rag_service.get_memory_state(session_id, limit or 80)

    @app.patch("/v1/rag/session/{session_id}/memory")
    def rag_patch_session_memory(session_id: str, request: RagSessionMemoryPatchRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if request.enabled is not None:
            payload["enabled"] = request.enabled
        if request.threshold_pct is not None:
            payload["thresholdPct"] = request.threshold_pct
        if request.token_budget is not None:
            payload["tokenBudget"] = request.token_budget
        return rag_service.update_memory_config(session_id, payload)

    @app.post("/v1/rag/session/{session_id}/memory/compact")
    def rag_compact_session_memory(session_id: str) -> dict[str, Any]:
        return rag_service.compact_memory(session_id, "manual")

    @app.post("/v1/rag/session/{session_id}/memory/clear")
    def rag_clear_session_memory(session_id: str) -> dict[str, Any]:
        return rag_service.clear_memory(session_id)

    @app.post("/v1/rag/session/{session_id}/memory/append")
    def rag_append_session_memory(session_id: str, request: RagMemoryAppendRequest) -> dict[str, Any]:
        return rag_service.append_memory_turn(session_id, request.prompt, request.answer)

    @app.post("/v1/rag/lookup")
    def rag_lookup(request: RagLookupRequest) -> dict[str, Any]:
        try:
            return rag_service.lookup(request.question, request.session_id, request.scope.model_dump() if request.scope else None)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app
