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
