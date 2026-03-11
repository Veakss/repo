from __future__ import annotations

import json
import uuid
from typing import Any

import httpx
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse

from continue_better_py.events import now_iso, sse
from continue_better_py.schemas import (
    ApprovalDecisionRequest,
    ChatMessage,
    ChatStreamRequest,
    ClarificationDecisionRequest,
    SessionCreateRequest,
    SessionUpdateRequest,
)
from continue_better_py.settings import Settings, ensure_runtime_dirs, get_settings
from continue_better_py.store import MongoStore


def _serialize_history(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [ChatMessage(role=row["role"], content=row["content"]).model_dump() for row in rows]


def _raise_sidecar_error(response: httpx.Response) -> None:
    detail = response.text.strip() or response.reason_phrase
    raise HTTPException(status_code=response.status_code, detail=detail)


async def _raise_streaming_sidecar_error(response: httpx.Response) -> None:
    body = await response.aread()
    detail = body.decode("utf-8", errors="replace").strip() or response.reason_phrase
    raise HTTPException(status_code=response.status_code, detail=detail)


def _render_clarification_message(event: dict[str, Any]) -> str:
    question = str(event.get("question") or "").strip()
    questions = [str(item).strip() for item in event.get("questions", []) if str(item).strip()]
    if question and question not in questions:
        questions.insert(0, question)
    options = []
    for option in event.get("options", [])[:4]:
        if isinstance(option, dict):
            label = str(option.get("label") or "").strip()
            if label:
                options.append(label)
    lines = ["I need clarification before continuing."]
    lines.extend(f"{index}. {item}" for index, item in enumerate(questions[:3], start=1))
    if options:
        lines.append("Quick options:")
        lines.extend(f"- {label}" for label in options)
    return "\n".join(lines)


def create_backend_app(
    store: MongoStore | None = None,
    settings: Settings | None = None,
    sidecar_transport: httpx.AsyncBaseTransport | None = None,
    sidecar_base_url: str | None = None,
) -> FastAPI:
    ensure_runtime_dirs()
    settings = settings or get_settings()
    store = store or MongoStore(settings=settings)
    app = FastAPI(title="Continue Better Python Backend", version="0.2.0")
    resolved_sidecar_url = (sidecar_base_url or settings.orchestrator_sidecar_url).rstrip("/")

    def sidecar_client(timeout: float | None = None) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=resolved_sidecar_url,
            timeout=timeout,
            transport=sidecar_transport,
        )

    def get_run_or_404(run_id: str) -> dict[str, Any]:
        try:
            run = store.get_run(run_id)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        return run

    def get_session_or_404(session_id: str) -> dict[str, Any]:
        try:
            session = store.get_session(session_id)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return session

    @app.get("/health")
    @app.get("/v1/health")
    def health() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/capabilities")
    @app.get("/v1/capabilities")
    async def capabilities() -> dict[str, Any]:
        async with sidecar_client(timeout=15.0) as client:
            response = await client.get("/v1/capabilities")
            if response.status_code >= 400:
                _raise_sidecar_error(response)
            return response.json()

    @app.get("/models")
    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        async with sidecar_client(timeout=15.0) as client:
            response = await client.get("/v1/models")
            if response.status_code >= 400:
                _raise_sidecar_error(response)
            return response.json()

    @app.get("/sessions")
    @app.get("/v1/sessions")
    def list_sessions() -> dict[str, Any]:
        try:
            return {"sessions": store.list_sessions()}
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/sessions")
    @app.post("/v1/sessions")
    def create_session(request: SessionCreateRequest | None = Body(default=None)) -> dict[str, Any]:
        try:
            created = store.create_session(request.title if request else None)
            return {"session_id": created["id"], "session": created}
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/sessions/{session_id}")
    @app.get("/v1/sessions/{session_id}")
    def get_session(session_id: str) -> dict[str, Any]:
        return {"session": get_session_or_404(session_id)}

    @app.patch("/sessions/{session_id}")
    @app.patch("/v1/sessions/{session_id}")
    def update_session(session_id: str, request: SessionUpdateRequest) -> dict[str, Any]:
        try:
            session = store.update_session_title(session_id, request.title)
            return {"session": session}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Session not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.delete("/sessions/{session_id}")
    @app.delete("/v1/sessions/{session_id}")
    def delete_session(session_id: str) -> dict[str, Any]:
        try:
            return store.delete_session(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Session not found") from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/sessions/{session_id}/messages")
    @app.get("/v1/sessions/{session_id}/messages")
    def get_messages(session_id: str, limit: int | None = Query(None, ge=1, le=1000)) -> dict[str, Any]:
        get_session_or_404(session_id)
        try:
            return {"session_id": session_id, "messages": store.get_messages(session_id, limit=limit)}
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/runs")
    @app.get("/v1/runs")
    def list_runs(session_id: str = Query(..., description="Filter runs by session ID")) -> dict[str, Any]:
        get_session_or_404(session_id)
        try:
            return {"runs": store.list_runs_for_session(session_id)}
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/runs/{run_id}/events")
    @app.get("/v1/runs/{run_id}/events")
    def get_run_events(run_id: str) -> dict[str, Any]:
        return {"run": get_run_or_404(run_id)}

    @app.post("/chat/stream")
    @app.post("/v1/chat/stream")
    async def chat_stream(request: ChatStreamRequest):
        get_session_or_404(request.session_id)
        run_id = request.run_id or str(uuid.uuid4())
        try:
            store.add_message(request.session_id, "user", request.message, run_id=run_id)
            store.start_run(
                run_id,
                request.session_id,
                meta_fields={
                    "workspace_root": request.workspace_root or str(settings.resolved_workspace_root),
                    "policy_profile": request.policy_profile,
                    "requested_model": request.model,
                    "requested_profile": request.profile,
                    "allow_writes": request.allow_writes,
                    "started_at": now_iso(),
                },
            )
            message_history = store.get_messages(request.session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Session not found") from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        sidecar_payload = {
            "sessionId": request.session_id,
            "messages": _serialize_history(message_history),
            "workspaceRoot": request.workspace_root or str(settings.resolved_workspace_root),
            "allowWrites": request.allow_writes,
            "policyProfile": request.policy_profile,
            "model": request.model,
            "runId": run_id,
            "profile": request.profile,
            "toolToggles": request.tool_toggles,
            "forceToolUse": request.force_tool_use,
        }

        async def event_generator():
            assistant_tokens: list[str] = []
            clarification_written = False
            async with sidecar_client(timeout=None) as client:
                async with client.stream("POST", "/v1/chat/stream", json=sidecar_payload) as response:
                    if response.status_code >= 400:
                        await _raise_streaming_sidecar_error(response)
                    buffer = ""
                    async for chunk in response.aiter_text():
                        buffer += chunk
                        parts = buffer.split("\n\n")
                        buffer = parts.pop() if parts else buffer
                        for part in parts:
                            if not part.startswith("data: "):
                                continue
                            payload = part[6:].strip()
                            if not payload:
                                continue
                            event = json.loads(payload)
                            if event.get("type") == "token":
                                assistant_tokens.append(event.get("token", ""))
                            if event.get("type") == "clarification_required" and not clarification_written:
                                store.add_message(
                                    request.session_id,
                                    "assistant",
                                    _render_clarification_message(event),
                                    run_id=run_id,
                                )
                                clarification_written = True
                            store.append_run_event(run_id, event)
                            yield sse(event)
            assistant_text = "".join(assistant_tokens).strip()
            if assistant_text:
                store.add_message(request.session_id, "assistant", assistant_text, run_id=run_id)

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    @app.post("/approvals/stream")
    @app.post("/v1/approvals/stream")
    async def approval_stream(request: ApprovalDecisionRequest):
        try:
            store.record_approval_decision(request.approval_id, request.decision)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        async def event_generator():
            assistant_tokens: list[str] = []
            clarification_runs: set[str] = set()
            observed_run_id: str | None = None
            async with sidecar_client(timeout=None) as client:
                async with client.stream("POST", "/v1/approvals/respond/stream", json=request.model_dump()) as response:
                    if response.status_code >= 400:
                        await _raise_streaming_sidecar_error(response)
                    buffer = ""
                    async for chunk in response.aiter_text():
                        buffer += chunk
                        parts = buffer.split("\n\n")
                        buffer = parts.pop() if parts else buffer
                        for part in parts:
                            if not part.startswith("data: "):
                                continue
                            payload = part[6:].strip()
                            if not payload:
                                continue
                            event = json.loads(payload)
                            run_id = event.get("runId")
                            if isinstance(run_id, str) and run_id:
                                observed_run_id = run_id
                            if event.get("type") == "token":
                                assistant_tokens.append(event.get("token", ""))
                            if event.get("type") == "clarification_required" and isinstance(run_id, str) and run_id not in clarification_runs:
                                run = get_run_or_404(run_id)
                                session_id = run["session_id"]
                                store.add_message(session_id, "assistant", _render_clarification_message(event), run_id=run_id)
                                clarification_runs.add(run_id)
                            if isinstance(run_id, str) and run_id:
                                store.append_run_event(run_id, event)
                            yield sse(event)
            assistant_text = "".join(assistant_tokens).strip()
            if assistant_text and observed_run_id:
                run = get_run_or_404(observed_run_id)
                store.add_message(run["session_id"], "assistant", assistant_text, run_id=observed_run_id)

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    @app.post("/clarifications/stream")
    @app.post("/v1/clarifications/stream")
    async def clarification_stream(request: ClarificationDecisionRequest):
        try:
            store.record_clarification_answer(request.clarification_id, request.answer)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        async def event_generator():
            assistant_tokens: list[str] = []
            observed_run_id: str | None = None
            async with sidecar_client(timeout=None) as client:
                synthetic: dict[str, Any] | None = None
                clarification = store.get_clarification(request.clarification_id)
                if clarification and clarification.get("run_id"):
                    synthetic = {
                        "type": "clarification_answered",
                        "runId": clarification["run_id"],
                        "clarificationId": request.clarification_id,
                        "answer": request.answer,
                        "timestamp": now_iso(),
                    }
                    store.append_run_event(clarification["run_id"], synthetic)
                    yield sse(synthetic)
                async with client.stream("POST", "/v1/clarifications/respond/stream", json=request.model_dump()) as response:
                    if response.status_code >= 400:
                        await _raise_streaming_sidecar_error(response)
                    buffer = ""
                    async for chunk in response.aiter_text():
                        buffer += chunk
                        parts = buffer.split("\n\n")
                        buffer = parts.pop() if parts else buffer
                        for part in parts:
                            if not part.startswith("data: "):
                                continue
                            payload = part[6:].strip()
                            if not payload:
                                continue
                            event = json.loads(payload)
                            run_id = event.get("runId")
                            if isinstance(run_id, str) and run_id:
                                observed_run_id = run_id
                                store.append_run_event(run_id, event)
                            if event.get("type") == "token":
                                assistant_tokens.append(event.get("token", ""))
                            yield sse(event)
            assistant_text = "".join(assistant_tokens).strip()
            if assistant_text and observed_run_id:
                run = get_run_or_404(observed_run_id)
                store.add_message(run["session_id"], "assistant", assistant_text, run_id=observed_run_id)

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    return app
