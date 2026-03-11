from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import httpx
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse

from continue_better_py.events import now_iso, sse
from continue_better_py.matrix import MatrixService
from continue_better_py.schemas import (
    ApprovalDecisionRequest,
    ChatMessage,
    ChatStreamRequest,
    ClarificationDecisionRequest,
    MatrixRunRequest,
    RagImportRequest,
    RagLookupRequest,
    RagProfileCreateRequest,
    RagProfileRenameRequest,
    RagSessionMemoryPatchRequest,
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


def _resolve_in_workspace(workspace_root: Path, relative_path: str) -> Path:
    base = workspace_root.resolve()
    resolved = base.joinpath(relative_path).resolve()
    if resolved != base and base not in resolved.parents:
        raise HTTPException(status_code=400, detail="Path escapes workspace root")
    return resolved


def _build_tree(root: Path) -> dict[str, Any]:
    def walk(directory: Path) -> list[dict[str, Any]]:
        children: list[dict[str, Any]] = []
        for child in sorted(directory.iterdir(), key=lambda item: (item.is_file(), item.name.lower())):
            if child.name.startswith("."):
                continue
            node: dict[str, Any] = {
                "name": child.name,
                "path": str(child.relative_to(root)),
                "type": "file" if child.is_file() else "directory",
            }
            if child.is_dir():
                node["children"] = walk(child)
            children.append(node)
        return children

    root.mkdir(parents=True, exist_ok=True)
    return {"root": str(root), "children": walk(root)}


def create_backend_app(
    store: MongoStore | None = None,
    settings: Settings | None = None,
    sidecar_transport: httpx.AsyncBaseTransport | None = None,
    sidecar_base_url: str | None = None,
    matrix_service: MatrixService | None = None,
) -> FastAPI:
    ensure_runtime_dirs()
    settings = settings or get_settings()
    store = store or MongoStore(settings=settings)
    app = FastAPI(title="Continue Better Python Backend", version="0.2.0")
    resolved_sidecar_url = (sidecar_base_url or settings.orchestrator_sidecar_url).rstrip("/")
    matrix_service = matrix_service or MatrixService(store=store, settings=settings)

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

    @app.get("/v1/matrix/catalog")
    def matrix_catalog() -> dict[str, Any]:
        try:
            return {
                "scenarios": matrix_service.list_scenarios(),
                "profiles": matrix_service.list_profiles(),
                "surfaces": matrix_service.list_surfaces(),
            }
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/v1/matrix/reports")
    def matrix_reports(limit: int = Query(25, ge=1, le=200)) -> dict[str, Any]:
        try:
            return {"reports": matrix_service.list_reports(limit)}
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/v1/matrix/reports/{report_id}")
    def matrix_report(report_id: str) -> dict[str, Any]:
        try:
            report = matrix_service.get_report(report_id)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        if not report:
            raise HTTPException(status_code=404, detail="Matrix report not found")
        return {"report": report}

    @app.get("/v1/matrix/compare")
    def matrix_compare(current_report_id: str = Query(...), baseline_report_id: str = Query(...)) -> dict[str, Any]:
        try:
            return {
                "comparison": matrix_service.compare(
                    current_report_id=current_report_id,
                    baseline_report_id=baseline_report_id,
                )
            }
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/v1/matrix/jobs")
    def matrix_jobs(limit: int = Query(25, ge=1, le=200)) -> dict[str, Any]:
        try:
            return {"jobs": matrix_service.list_jobs(limit)}
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/v1/matrix/jobs/{job_id}")
    def matrix_job(job_id: str) -> dict[str, Any]:
        try:
            job = matrix_service.get_job(job_id)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        if not job:
            raise HTTPException(status_code=404, detail="Matrix job not found")
        return {"job": job}

    @app.post("/v1/matrix/jobs")
    def matrix_start_job(request: MatrixRunRequest) -> dict[str, Any]:
        try:
            job = matrix_service.create_job(request.model_dump(exclude_none=True))
            return {"job": job}
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/v1/rag/profiles")
    async def rag_list_profiles() -> dict[str, Any]:
        async with sidecar_client(timeout=20.0) as client:
            response = await client.get("/v1/rag/profiles")
            if response.status_code >= 400:
                _raise_sidecar_error(response)
            return response.json()

    @app.post("/v1/rag/profiles")
    async def rag_create_profile(request: RagProfileCreateRequest) -> dict[str, Any]:
        async with sidecar_client(timeout=20.0) as client:
            response = await client.post("/v1/rag/profiles", json=request.model_dump())
            if response.status_code >= 400:
                _raise_sidecar_error(response)
            return response.json()

    @app.patch("/v1/rag/profiles/{profile_name}")
    async def rag_rename_profile(profile_name: str, request: RagProfileRenameRequest) -> dict[str, Any]:
        async with sidecar_client(timeout=20.0) as client:
            response = await client.patch(f"/v1/rag/profiles/{profile_name}", json=request.model_dump())
            if response.status_code >= 400:
                _raise_sidecar_error(response)
            return response.json()

    @app.delete("/v1/rag/profiles/{profile_name}")
    async def rag_delete_profile(profile_name: str) -> dict[str, Any]:
        async with sidecar_client(timeout=20.0) as client:
            response = await client.delete(f"/v1/rag/profiles/{profile_name}")
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

    @app.get("/fs/tree")
    @app.get("/v1/fs/tree")
    def fs_tree() -> dict[str, Any]:
        return _build_tree(settings.resolved_workspace_root)

    @app.get("/fs/read")
    @app.get("/v1/fs/read")
    def fs_read(path: str = Query(..., description="Path relative to workspace root")) -> dict[str, Any]:
        target = _resolve_in_workspace(settings.resolved_workspace_root, path)
        if not target.exists() or not target.is_file():
            raise HTTPException(status_code=404, detail="File not found")
        return {"path": str(target.relative_to(settings.resolved_workspace_root)), "content": target.read_text(encoding="utf-8")}

    @app.post("/v1/rag/session/{session_id}/files/import")
    async def rag_import_session_file(session_id: str, request: RagImportRequest) -> dict[str, Any]:
        target = _resolve_in_workspace(settings.resolved_workspace_root, request.path)
        if not target.exists() or not target.is_file():
            raise HTTPException(status_code=404, detail="Source file not found in workspace")
        async with sidecar_client(timeout=60.0) as client:
            response = await client.post(f"/v1/rag/session/{session_id}/files/import", json={"path": str(target)})
            if response.status_code >= 400:
                _raise_sidecar_error(response)
            return response.json()

    @app.post("/v1/rag/profiles/{profile_name}/files/import")
    async def rag_import_profile_file(profile_name: str, request: RagImportRequest) -> dict[str, Any]:
        target = _resolve_in_workspace(settings.resolved_workspace_root, request.path)
        if not target.exists() or not target.is_file():
            raise HTTPException(status_code=404, detail="Source file not found in workspace")
        async with sidecar_client(timeout=60.0) as client:
            response = await client.post(f"/v1/rag/profiles/{profile_name}/files/import", json={"path": str(target)})
            if response.status_code >= 400:
                _raise_sidecar_error(response)
            return response.json()

    @app.get("/v1/rag/session/{session_id}/files")
    async def rag_get_session_files(session_id: str) -> dict[str, Any]:
        async with sidecar_client(timeout=20.0) as client:
            response = await client.get(f"/v1/rag/session/{session_id}/files")
            if response.status_code >= 400:
                _raise_sidecar_error(response)
            return response.json()

    @app.get("/v1/rag/profiles/{profile_name}/files")
    async def rag_get_profile_files(profile_name: str) -> dict[str, Any]:
        async with sidecar_client(timeout=20.0) as client:
            response = await client.get(f"/v1/rag/profiles/{profile_name}/files")
            if response.status_code >= 400:
                _raise_sidecar_error(response)
            return response.json()

    @app.delete("/v1/rag/session/{session_id}/files/{file_id}")
    async def rag_delete_session_file(session_id: str, file_id: str) -> dict[str, Any]:
        async with sidecar_client(timeout=20.0) as client:
            response = await client.delete(f"/v1/rag/session/{session_id}/files/{file_id}")
            if response.status_code >= 400:
                _raise_sidecar_error(response)
            return response.json()

    @app.delete("/v1/rag/profiles/{profile_name}/files/{file_id}")
    async def rag_delete_profile_file(profile_name: str, file_id: str) -> dict[str, Any]:
        async with sidecar_client(timeout=20.0) as client:
            response = await client.delete(f"/v1/rag/profiles/{profile_name}/files/{file_id}")
            if response.status_code >= 400:
                _raise_sidecar_error(response)
            return response.json()

    @app.post("/v1/rag/session/{session_id}/index/jobs")
    async def rag_enqueue_session_index_job(session_id: str) -> dict[str, Any]:
        async with sidecar_client(timeout=20.0) as client:
            response = await client.post(f"/v1/rag/session/{session_id}/index/jobs")
            if response.status_code >= 400:
                _raise_sidecar_error(response)
            return response.json()

    @app.post("/v1/rag/profiles/{profile_name}/index/jobs")
    async def rag_enqueue_profile_index_job(profile_name: str) -> dict[str, Any]:
        async with sidecar_client(timeout=20.0) as client:
            response = await client.post(f"/v1/rag/profiles/{profile_name}/index/jobs")
            if response.status_code >= 400:
                _raise_sidecar_error(response)
            return response.json()

    @app.get("/v1/rag/index/jobs")
    async def rag_list_index_jobs(limit: int | None = Query(None, ge=1, le=200)) -> dict[str, Any]:
        async with sidecar_client(timeout=20.0) as client:
            response = await client.get("/v1/rag/index/jobs", params={"limit": limit} if limit else None)
            if response.status_code >= 400:
                _raise_sidecar_error(response)
            return response.json()

    @app.get("/v1/rag/index/jobs/{job_id}")
    async def rag_get_index_job(job_id: str) -> dict[str, Any]:
        async with sidecar_client(timeout=20.0) as client:
            response = await client.get(f"/v1/rag/index/jobs/{job_id}")
            if response.status_code >= 400:
                _raise_sidecar_error(response)
            return response.json()

    @app.post("/v1/rag/index/jobs/{job_id}/retry")
    async def rag_retry_index_job(job_id: str) -> dict[str, Any]:
        async with sidecar_client(timeout=20.0) as client:
            response = await client.post(f"/v1/rag/index/jobs/{job_id}/retry")
            if response.status_code >= 400:
                _raise_sidecar_error(response)
            return response.json()

    @app.get("/v1/rag/session/{session_id}/memory")
    async def rag_get_session_memory(session_id: str, limit: int | None = Query(None, ge=1, le=300)) -> dict[str, Any]:
        async with sidecar_client(timeout=20.0) as client:
            response = await client.get(f"/v1/rag/session/{session_id}/memory", params={"limit": limit} if limit else None)
            if response.status_code >= 400:
                _raise_sidecar_error(response)
            return response.json()

    @app.patch("/v1/rag/session/{session_id}/memory")
    async def rag_patch_session_memory(session_id: str, request: RagSessionMemoryPatchRequest) -> dict[str, Any]:
        async with sidecar_client(timeout=20.0) as client:
            response = await client.patch(f"/v1/rag/session/{session_id}/memory", json=request.model_dump(exclude_none=True))
            if response.status_code >= 400:
                _raise_sidecar_error(response)
            return response.json()

    @app.post("/v1/rag/session/{session_id}/memory/compact")
    async def rag_compact_session_memory(session_id: str) -> dict[str, Any]:
        async with sidecar_client(timeout=20.0) as client:
            response = await client.post(f"/v1/rag/session/{session_id}/memory/compact")
            if response.status_code >= 400:
                _raise_sidecar_error(response)
            return response.json()

    @app.post("/v1/rag/session/{session_id}/memory/clear")
    async def rag_clear_session_memory(session_id: str) -> dict[str, Any]:
        async with sidecar_client(timeout=20.0) as client:
            response = await client.post(f"/v1/rag/session/{session_id}/memory/clear")
            if response.status_code >= 400:
                _raise_sidecar_error(response)
            return response.json()

    @app.post("/v1/rag/lookup")
    async def rag_lookup(request: RagLookupRequest) -> dict[str, Any]:
        async with sidecar_client(timeout=30.0) as client:
            response = await client.post("/v1/rag/lookup", json=request.model_dump(exclude_none=True))
            if response.status_code >= 400:
                _raise_sidecar_error(response)
            return response.json()

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
                async with sidecar_client(timeout=20.0) as client:
                    try:
                        await client.post(
                            f"/v1/rag/session/{request.session_id}/memory/append",
                            json={"prompt": request.message, "answer": assistant_text},
                        )
                    except Exception:
                        pass

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
                async with sidecar_client(timeout=20.0) as client:
                    try:
                        messages = store.get_messages(run["session_id"], limit=5)
                        last_user = next((row["content"] for row in reversed(messages) if row["role"] == "user"), "")
                        await client.post(
                            f"/v1/rag/session/{run['session_id']}/memory/append",
                            json={"prompt": last_user, "answer": assistant_text},
                        )
                    except Exception:
                        pass

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
                async with sidecar_client(timeout=20.0) as client:
                    try:
                        await client.post(
                            f"/v1/rag/session/{run['session_id']}/memory/append",
                            json={"prompt": request.answer, "answer": assistant_text},
                        )
                    except Exception:
                        pass

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    return app
