from __future__ import annotations

import json
import uuid
from typing import Any

import httpx
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from continue_better_py.events import sse
from continue_better_py.schemas import ChatMessage, ChatStreamRequest, SessionCreateRequest
from continue_better_py.settings import ensure_runtime_dirs, get_settings
from continue_better_py.store import MongoStore


def _serialize_history(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [ChatMessage(role=row["role"], content=row["content"]).model_dump() for row in rows]


def create_backend_app() -> FastAPI:
    ensure_runtime_dirs()
    settings = get_settings()
    store = MongoStore()
    app = FastAPI(title="Continue Better Python Backend", version="0.1.0")

    @app.get("/health")
    @app.get("/v1/health")
    def health() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/capabilities")
    @app.get("/v1/capabilities")
    async def capabilities() -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(f"{settings.orchestrator_sidecar_url}/v1/capabilities")
            response.raise_for_status()
            return response.json()

    @app.get("/models")
    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(f"{settings.orchestrator_sidecar_url}/v1/models")
            response.raise_for_status()
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

    @app.get("/sessions/{session_id}/messages")
    @app.get("/v1/sessions/{session_id}/messages")
    def get_messages(session_id: str) -> dict[str, Any]:
        try:
            return {"session_id": session_id, "messages": store.get_messages(session_id)}
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/chat/stream")
    @app.post("/v1/chat/stream")
    async def chat_stream(request: ChatStreamRequest):
        run_id = request.run_id or str(uuid.uuid4())
        try:
            store.start_run(run_id, request.session_id)
            store.add_message(request.session_id, "user", request.message)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        sidecar_payload = {
            "sessionId": request.session_id,
            "messages": _serialize_history(store.get_messages(request.session_id)),
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
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "POST",
                    f"{settings.orchestrator_sidecar_url}/v1/chat/stream",
                    json=sidecar_payload,
                ) as response:
                    if response.status_code >= 400:
                        raise HTTPException(status_code=response.status_code, detail=await response.aread())
                    buffer = ""
                    async for chunk in response.aiter_text():
                        buffer += chunk
                        parts = buffer.split("\n\n")
                        buffer = parts.pop() or ""
                        for part in parts:
                            if not part.startswith("data: "):
                                continue
                            payload = part[6:].strip()
                            if not payload:
                                continue
                            event = json.loads(payload)
                            try:
                                store.append_run_event(run_id, event)
                            except RuntimeError:
                                pass
                            if event.get("type") == "token":
                                assistant_tokens.append(event.get("token", ""))
                            yield sse(event)
            assistant_text = "".join(assistant_tokens).strip()
            if assistant_text:
                try:
                    store.add_message(request.session_id, "assistant", assistant_text)
                except RuntimeError:
                    pass

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    return app
