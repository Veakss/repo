from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import httpx


def iter_sse_events(response: httpx.Response) -> Iterator[dict[str, Any]]:
    buffer = ""
    for chunk in response.iter_text():
        buffer += chunk
        parts = buffer.split("\n\n")
        buffer = parts.pop() or ""
        for part in parts:
            if not part.startswith("data: "):
                continue
            payload = part[6:].strip()
            if payload:
                yield json.loads(payload)


class BackendClient:
    def __init__(self, base_url: str, transport: httpx.BaseTransport | httpx.AsyncBaseTransport | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.transport = transport

    def _client(self, timeout: float | None = 20.0) -> httpx.Client:
        return httpx.Client(base_url=self.base_url, timeout=timeout, transport=self.transport)

    def health(self) -> dict[str, Any]:
        with self._client(timeout=10.0) as client:
            response = client.get("/v1/health")
            response.raise_for_status()
            return response.json()

    def capabilities(self) -> dict[str, Any]:
        with self._client(timeout=10.0) as client:
            response = client.get("/v1/capabilities")
            response.raise_for_status()
            return response.json()

    def models(self) -> dict[str, Any]:
        with self._client(timeout=20.0) as client:
            response = client.get("/v1/models")
            response.raise_for_status()
            return response.json()

    def list_sessions(self) -> list[dict[str, Any]]:
        with self._client() as client:
            response = client.get("/v1/sessions")
            response.raise_for_status()
            return response.json()["sessions"]

    def create_session(self, title: str | None = None) -> dict[str, Any]:
        payload = {"title": title} if title else None
        with self._client() as client:
            response = client.post("/v1/sessions", json=payload)
            response.raise_for_status()
            return response.json()

    def get_session(self, session_id: str) -> dict[str, Any]:
        with self._client() as client:
            response = client.get(f"/v1/sessions/{session_id}")
            response.raise_for_status()
            return response.json()["session"]

    def update_session(self, session_id: str, title: str) -> dict[str, Any]:
        with self._client() as client:
            response = client.patch(f"/v1/sessions/{session_id}", json={"title": title})
            response.raise_for_status()
            return response.json()["session"]

    def delete_session(self, session_id: str) -> dict[str, Any]:
        with self._client() as client:
            response = client.delete(f"/v1/sessions/{session_id}")
            response.raise_for_status()
            return response.json()

    def get_session_messages(self, session_id: str, limit: int | None = None) -> list[dict[str, Any]]:
        params = {"limit": limit} if limit else None
        with self._client() as client:
            response = client.get(f"/v1/sessions/{session_id}/messages", params=params)
            response.raise_for_status()
            return response.json()["messages"]

    def list_runs(self, session_id: str) -> list[dict[str, Any]]:
        with self._client() as client:
            response = client.get("/v1/runs", params={"session_id": session_id})
            response.raise_for_status()
            return response.json()["runs"]

    def get_run(self, run_id: str) -> dict[str, Any]:
        with self._client() as client:
            response = client.get(f"/v1/runs/{run_id}/events")
            response.raise_for_status()
            return response.json()["run"]

    def fs_tree(self) -> dict[str, Any]:
        with self._client() as client:
            response = client.get("/v1/fs/tree")
            response.raise_for_status()
            return response.json()

    def fs_read(self, path: str) -> dict[str, Any]:
        with self._client() as client:
            response = client.get("/v1/fs/read", params={"path": path})
            response.raise_for_status()
            return response.json()

    def create_terminal(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._client(timeout=30.0) as client:
            response = client.post("/v1/terminals", json=payload)
            response.raise_for_status()
            return response.json()

    def get_terminal(self, terminal_id: str) -> dict[str, Any]:
        with self._client(timeout=30.0) as client:
            response = client.get(f"/v1/terminals/{terminal_id}")
            response.raise_for_status()
            return response.json()

    def write_terminal(self, terminal_id: str, data: str, source: str = "user") -> dict[str, Any]:
        with self._client(timeout=30.0) as client:
            response = client.post(f"/v1/terminals/{terminal_id}/write", json={"data": data, "source": source})
            response.raise_for_status()
            return response.json()

    def interrupt_terminal(self, terminal_id: str, source: str = "user") -> dict[str, Any]:
        with self._client(timeout=30.0) as client:
            response = client.post(f"/v1/terminals/{terminal_id}/interrupt", json={"source": source})
            response.raise_for_status()
            return response.json()

    def resize_terminal(self, terminal_id: str, cols: int, rows: int) -> dict[str, Any]:
        with self._client(timeout=30.0) as client:
            response = client.post(f"/v1/terminals/{terminal_id}/resize", json={"cols": cols, "rows": rows})
            response.raise_for_status()
            return response.json()

    def set_terminal_control(self, terminal_id: str, owner: str, reason: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"owner": owner}
        if reason:
            payload["reason"] = reason
        with self._client(timeout=30.0) as client:
            response = client.post(f"/v1/terminals/{terminal_id}/control", json=payload)
            response.raise_for_status()
            return response.json()

    def close_terminal(self, terminal_id: str) -> dict[str, Any]:
        with self._client(timeout=30.0) as client:
            response = client.post(f"/v1/terminals/{terminal_id}/close")
            response.raise_for_status()
            return response.json()

    def stream_terminal(self, terminal_id: str) -> Iterator[dict[str, Any]]:
        with self._client(timeout=None) as client:
            with client.stream("GET", f"/v1/terminals/{terminal_id}/stream") as response:
                response.raise_for_status()
                yield from iter_sse_events(response)

    def rag_list_profiles(self) -> dict[str, Any]:
        with self._client() as client:
            response = client.get("/v1/rag/profiles")
            response.raise_for_status()
            return response.json()

    def rag_create_profile(self, name: str) -> dict[str, Any]:
        with self._client() as client:
            response = client.post("/v1/rag/profiles", json={"name": name})
            response.raise_for_status()
            return response.json()

    def rag_rename_profile(self, profile_name: str, new_name: str) -> dict[str, Any]:
        with self._client() as client:
            response = client.patch(f"/v1/rag/profiles/{profile_name}", json={"new_name": new_name})
            response.raise_for_status()
            return response.json()

    def rag_delete_profile(self, profile_name: str) -> dict[str, Any]:
        with self._client() as client:
            response = client.delete(f"/v1/rag/profiles/{profile_name}")
            response.raise_for_status()
            return response.json()

    def rag_import_session_file(self, session_id: str, path: str) -> dict[str, Any]:
        with self._client(timeout=60.0) as client:
            response = client.post(f"/v1/rag/session/{session_id}/files/import", json={"path": path})
            response.raise_for_status()
            return response.json()

    def rag_import_profile_file(self, profile_name: str, path: str) -> dict[str, Any]:
        with self._client(timeout=60.0) as client:
            response = client.post(f"/v1/rag/profiles/{profile_name}/files/import", json={"path": path})
            response.raise_for_status()
            return response.json()

    def rag_get_session_files(self, session_id: str) -> dict[str, Any]:
        with self._client() as client:
            response = client.get(f"/v1/rag/session/{session_id}/files")
            response.raise_for_status()
            return response.json()

    def rag_get_profile_files(self, profile_name: str) -> dict[str, Any]:
        with self._client() as client:
            response = client.get(f"/v1/rag/profiles/{profile_name}/files")
            response.raise_for_status()
            return response.json()

    def rag_delete_session_file(self, session_id: str, file_id: str) -> dict[str, Any]:
        with self._client() as client:
            response = client.delete(f"/v1/rag/session/{session_id}/files/{file_id}")
            response.raise_for_status()
            return response.json()

    def rag_delete_profile_file(self, profile_name: str, file_id: str) -> dict[str, Any]:
        with self._client() as client:
            response = client.delete(f"/v1/rag/profiles/{profile_name}/files/{file_id}")
            response.raise_for_status()
            return response.json()

    def rag_enqueue_session_index_job(self, session_id: str) -> dict[str, Any]:
        with self._client() as client:
            response = client.post(f"/v1/rag/session/{session_id}/index/jobs")
            response.raise_for_status()
            return response.json()

    def rag_enqueue_profile_index_job(self, profile_name: str) -> dict[str, Any]:
        with self._client() as client:
            response = client.post(f"/v1/rag/profiles/{profile_name}/index/jobs")
            response.raise_for_status()
            return response.json()

    def rag_list_index_jobs(self, limit: int | None = None) -> dict[str, Any]:
        with self._client() as client:
            response = client.get("/v1/rag/index/jobs", params={"limit": limit} if limit else None)
            response.raise_for_status()
            return response.json()

    def rag_get_session_memory(self, session_id: str, limit: int | None = None) -> dict[str, Any]:
        with self._client() as client:
            response = client.get(f"/v1/rag/session/{session_id}/memory", params={"limit": limit} if limit else None)
            response.raise_for_status()
            return response.json()

    def rag_patch_session_memory(self, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        with self._client() as client:
            response = client.patch(f"/v1/rag/session/{session_id}/memory", json=payload)
            response.raise_for_status()
            return response.json()

    def rag_compact_session_memory(self, session_id: str) -> dict[str, Any]:
        with self._client() as client:
            response = client.post(f"/v1/rag/session/{session_id}/memory/compact")
            response.raise_for_status()
            return response.json()

    def rag_clear_session_memory(self, session_id: str) -> dict[str, Any]:
        with self._client() as client:
            response = client.post(f"/v1/rag/session/{session_id}/memory/clear")
            response.raise_for_status()
            return response.json()

    def rag_lookup(self, question: str, session_id: str | None = None, scope: dict[str, Any] | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"question": question}
        if session_id:
            payload["session_id"] = session_id
        if scope:
            payload["scope"] = scope
        with self._client(timeout=30.0) as client:
            response = client.post("/v1/rag/lookup", json=payload)
            response.raise_for_status()
            return response.json()

    def matrix_catalog(self) -> dict[str, Any]:
        with self._client(timeout=30.0) as client:
            response = client.get("/v1/matrix/catalog")
            response.raise_for_status()
            return response.json()

    def matrix_list_reports(self, limit: int | None = None) -> list[dict[str, Any]]:
        with self._client(timeout=30.0) as client:
            response = client.get("/v1/matrix/reports", params={"limit": limit} if limit else None)
            response.raise_for_status()
            return response.json()["reports"]

    def matrix_get_report(self, report_id: str) -> dict[str, Any]:
        with self._client(timeout=30.0) as client:
            response = client.get(f"/v1/matrix/reports/{report_id}")
            response.raise_for_status()
            return response.json()["report"]

    def matrix_list_jobs(self, limit: int | None = None) -> list[dict[str, Any]]:
        with self._client(timeout=30.0) as client:
            response = client.get("/v1/matrix/jobs", params={"limit": limit} if limit else None)
            response.raise_for_status()
            return response.json()["jobs"]

    def matrix_get_job(self, job_id: str) -> dict[str, Any]:
        with self._client(timeout=30.0) as client:
            response = client.get(f"/v1/matrix/jobs/{job_id}")
            response.raise_for_status()
            return response.json()["job"]

    def matrix_start_job(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._client(timeout=30.0) as client:
            response = client.post("/v1/matrix/jobs", json=payload)
            response.raise_for_status()
            return response.json()["job"]

    def matrix_compare(self, current_report_id: str, baseline_report_id: str) -> dict[str, Any]:
        with self._client(timeout=30.0) as client:
            response = client.get(
                "/v1/matrix/compare",
                params={"current_report_id": current_report_id, "baseline_report_id": baseline_report_id},
            )
            response.raise_for_status()
            return response.json()["comparison"]

    def stream_chat(self, payload: dict[str, Any]) -> Iterator[dict[str, Any]]:
        with self._client(timeout=None) as client:
            with client.stream("POST", "/v1/chat/stream", json=payload) as response:
                response.raise_for_status()
                yield from iter_sse_events(response)

    def stream_approval(self, approval_id: str, decision: str) -> Iterator[dict[str, Any]]:
        with self._client(timeout=None) as client:
            with client.stream(
                "POST",
                "/v1/approvals/stream",
                json={"approval_id": approval_id, "decision": decision},
            ) as response:
                response.raise_for_status()
                yield from iter_sse_events(response)

    def stream_clarification(self, clarification_id: str, answer: str) -> Iterator[dict[str, Any]]:
        with self._client(timeout=None) as client:
            with client.stream(
                "POST",
                "/v1/clarifications/stream",
                json={"clarification_id": clarification_id, "answer": answer},
            ) as response:
                response.raise_for_status()
                yield from iter_sse_events(response)
