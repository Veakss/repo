from __future__ import annotations

import json

import httpx
import mongomock
from fastapi.testclient import TestClient

from continue_better_py.backend_app import create_backend_app
from continue_better_py.store import MongoStore


def parse_sse_payloads(text: str) -> list[dict]:
    payloads: list[dict] = []
    for packet in text.split("\n\n"):
        if packet.startswith("data: "):
            payloads.append(json.loads(packet[6:]))
    return payloads


def encode_sse(events: list[dict]) -> str:
    return "".join(f"data: {json.dumps(event, ensure_ascii=False)}\n\n" for event in events)


def build_test_client(tmp_path, handler) -> tuple[TestClient, MongoStore]:
    store = MongoStore(
        client=mongomock.MongoClient(),
        database_name="continue_better_python_backend_test",
        artifacts_root=tmp_path,
    )
    workspace = tmp_path.joinpath("workspace")
    workspace.mkdir(parents=True, exist_ok=True)
    workspace.joinpath("notes.txt").write_text("hello workspace", encoding="utf-8")
    store.settings.workspace_root = str(workspace)
    transport = httpx.MockTransport(handler)
    return TestClient(create_backend_app(store=store, sidecar_transport=transport, sidecar_base_url="http://sidecar.test")), store


def test_backend_session_crud_and_run_queries(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/chat/stream":
            body = encode_sse(
                [
                    {"type": "run_state", "runId": "run-1", "state": "planning", "timestamp": "2026-03-11T12:00:00+00:00"},
                    {"type": "run_phase_changed", "runId": "run-1", "phase": "execute", "timestamp": "2026-03-11T12:00:01+00:00"},
                    {"type": "token", "token": "Hello"},
                    {"type": "token", "token": " world"},
                    {"type": "run_state", "runId": "run-1", "state": "completed", "timestamp": "2026-03-11T12:00:02+00:00"},
                    {"type": "done"},
                ]
            )
            return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})
        if request.url.path == "/v1/capabilities":
            return httpx.Response(200, json={"tools": {"files": True}})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"models": [{"id": "gemini"}]})
        raise AssertionError(f"Unexpected path: {request.url.path}")

    client, store = build_test_client(tmp_path, handler)

    created = client.post("/v1/sessions", json={"title": "Planning"}).json()
    session_id = created["session_id"]
    assert client.get(f"/v1/sessions/{session_id}").status_code == 200
    renamed = client.patch(f"/v1/sessions/{session_id}", json={"title": "Renamed Session"})
    assert renamed.status_code == 200
    assert renamed.json()["session"]["title"] == "Renamed Session"

    stream = client.post(
        "/v1/chat/stream",
        json={"session_id": session_id, "message": "Hello backend", "run_id": "run-1", "allow_writes": False},
    )
    assert stream.status_code == 200
    events = parse_sse_payloads(stream.text)
    assert any(event["type"] == "token" for event in events)
    assert any(event["type"] == "run_state" and event["state"] == "completed" for event in events)

    messages = client.get(f"/v1/sessions/{session_id}/messages").json()["messages"]
    assert [message["role"] for message in messages] == ["user", "assistant"]
    assert messages[-1]["content"] == "Hello world"

    runs = client.get("/v1/runs", params={"session_id": session_id}).json()["runs"]
    assert runs[0]["run_id"] == "run-1"
    run = client.get("/v1/runs/run-1/events").json()["run"]
    assert run["meta"]["state"] == "completed"
    assert len(run["events"]) == 6
    assert store.get_run("run-1")["meta"]["event_count"] == 6

    tree = client.get("/v1/fs/tree")
    assert tree.status_code == 200
    assert tree.json()["children"][0]["path"] == "notes.txt"
    file_payload = client.get("/v1/fs/read", params={"path": "notes.txt"})
    assert file_payload.status_code == 200
    assert file_payload.json()["content"] == "hello workspace"

    deleted = client.delete(f"/v1/sessions/{session_id}")
    assert deleted.status_code == 200
    assert client.get(f"/v1/sessions/{session_id}").status_code == 404


def test_backend_persists_approval_and_clarification_flows(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/chat/stream":
            body = encode_sse(
                [
                    {"type": "run_state", "runId": "run-approve", "state": "planning", "timestamp": "2026-03-11T12:10:00+00:00"},
                    {
                        "type": "approval_required",
                        "runId": "run-approve",
                        "actionId": "approval-1",
                        "name": "write_file",
                        "riskLevel": "risky",
                        "arguments": "{\"path\":\"app.py\"}",
                        "timestamp": "2026-03-11T12:10:01+00:00",
                    },
                    {"type": "done"},
                ]
            )
            return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})
        if request.url.path == "/v1/approvals/respond/stream":
            body = encode_sse(
                [
                    {"type": "approval_decision", "runId": "run-approve", "actionId": "approval-1", "decision": "approved", "timestamp": "2026-03-11T12:10:02+00:00"},
                    {
                        "type": "clarification_required",
                        "runId": "run-approve",
                        "clarificationId": "clar-1",
                        "question": "Which module should I edit?",
                        "questions": ["Which module should I edit?"],
                        "options": [{"label": "app.py"}, {"label": "main.py"}],
                        "timestamp": "2026-03-11T12:10:03+00:00",
                    },
                    {"type": "done"},
                ]
            )
            return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})
        if request.url.path == "/v1/clarifications/respond/stream":
            body = encode_sse(
                [
                    {"type": "token", "token": "Updated"},
                    {"type": "token", "token": " app.py"},
                    {"type": "run_state", "runId": "run-approve", "state": "completed", "timestamp": "2026-03-11T12:10:04+00:00"},
                    {"type": "done"},
                ]
            )
            return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})
        if request.url.path == "/v1/capabilities":
            return httpx.Response(200, json={"tools": {"files": True}})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"models": [{"id": "gemini"}]})
        raise AssertionError(f"Unexpected path: {request.url.path}")

    client, store = build_test_client(tmp_path, handler)
    session_id = client.post("/v1/sessions", json={"title": "Approval Session"}).json()["session_id"]

    first = client.post(
        "/v1/chat/stream",
        json={"session_id": session_id, "message": "Please write the file", "run_id": "run-approve"},
    )
    assert first.status_code == 200
    first_events = parse_sse_payloads(first.text)
    assert any(event["type"] == "approval_required" for event in first_events)
    assert store.get_approval("approval-1")["status"] == "pending"

    approval = client.post("/v1/approvals/stream", json={"approval_id": "approval-1", "decision": "approved"})
    assert approval.status_code == 200
    approval_events = parse_sse_payloads(approval.text)
    assert any(event["type"] == "clarification_required" for event in approval_events)
    assert store.get_approval("approval-1")["status"] == "approved"
    clarification_message = client.get(f"/v1/sessions/{session_id}/messages").json()["messages"][-1]["content"]
    assert "I need clarification before continuing." in clarification_message
    assert store.get_clarification("clar-1")["status"] == "pending"

    clarification = client.post("/v1/clarifications/stream", json={"clarification_id": "clar-1", "answer": "Use app.py"})
    assert clarification.status_code == 200
    clarification_events = parse_sse_payloads(clarification.text)
    assert clarification_events[0]["type"] == "clarification_answered"
    assert any(event["type"] == "run_state" and event["state"] == "completed" for event in clarification_events)
    assert store.get_clarification("clar-1")["status"] == "answered"

    run = client.get("/v1/runs/run-approve/events").json()["run"]
    assert run["meta"]["clarifications_answered"] == 1
    assert run["meta"]["state"] == "completed"
    messages = client.get(f"/v1/sessions/{session_id}/messages").json()["messages"]
    assert messages[-1]["content"] == "Updated app.py"


def test_backend_proxies_rag_routes(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/rag/profiles" and request.method == "GET":
            return httpx.Response(200, json={"profiles": ["default"]})
        if request.url.path == "/v1/rag/profiles" and request.method == "POST":
            return httpx.Response(200, json={"ok": True, "profiles": ["default", "team"]})
        if request.url.path.startswith("/v1/rag/session/") and request.url.path.endswith("/files/import"):
            return httpx.Response(200, json={"imported": True, "entry": {"id": "file-1"}})
        if request.url.path.startswith("/v1/rag/session/") and request.url.path.endswith("/files"):
            return httpx.Response(200, json={"scope": {"kind": "session", "id": "s1"}, "files": [{"id": "file-1", "status": "pending"}]})
        if request.url.path.startswith("/v1/rag/session/") and request.url.path.endswith("/index/jobs"):
            return httpx.Response(200, json={"job": {"job_id": "job-1", "status": "queued"}})
        if request.url.path == "/v1/rag/index/jobs":
            return httpx.Response(200, json={"jobs": [{"job_id": "job-1", "status": "done"}], "queue": {"processing": False, "queuedJobIds": [], "activeJobId": None}})
        if request.url.path.startswith("/v1/rag/session/") and request.url.path.endswith("/memory"):
            if request.method == "GET":
                return httpx.Response(200, json={"config": {"enabled": True, "thresholdPct": 0.9, "tokenBudget": 12000}, "summary": {"text": "", "entryCount": 0, "estimatedTokens": 0}, "entries": []})
            return httpx.Response(200, json={"config": {"enabled": False, "thresholdPct": 0.75, "tokenBudget": 4000}, "summary": {"text": "", "entryCount": 0, "estimatedTokens": 0}, "entries": []})
        if request.url.path == "/v1/rag/lookup":
            return httpx.Response(200, json={"status": "ok", "query": "phase 4", "hits": [{"snippet": "Phase 4 covers RAG parity.", "citation": {"path": "notes.txt"}}]})
        if request.url.path == "/v1/capabilities":
            return httpx.Response(200, json={"tools": {"files": True, "rag": True}})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"models": [{"id": "gemini"}]})
        raise AssertionError(f"Unexpected path: {request.url.path} {request.method}")

    client, _store = build_test_client(tmp_path, handler)
    session_id = client.post("/v1/sessions", json={"title": "RAG Session"}).json()["session_id"]

    profiles = client.get("/v1/rag/profiles")
    assert profiles.status_code == 200
    assert profiles.json()["profiles"] == ["default"]

    created = client.post("/v1/rag/profiles", json={"name": "team"})
    assert created.status_code == 200
    assert "team" in created.json()["profiles"]

    imported = client.post(f"/v1/rag/session/{session_id}/files/import", json={"path": "notes.txt"})
    assert imported.status_code == 200

    files = client.get(f"/v1/rag/session/{session_id}/files")
    assert files.status_code == 200
    assert files.json()["files"][0]["id"] == "file-1"

    enqueued = client.post(f"/v1/rag/session/{session_id}/index/jobs")
    assert enqueued.status_code == 200
    assert enqueued.json()["job"]["job_id"] == "job-1"

    jobs = client.get("/v1/rag/index/jobs")
    assert jobs.status_code == 200
    assert jobs.json()["jobs"][0]["status"] == "done"

    memory = client.get(f"/v1/rag/session/{session_id}/memory")
    assert memory.status_code == 200
    patched = client.patch(f"/v1/rag/session/{session_id}/memory", json={"enabled": False, "threshold_pct": 0.75, "token_budget": 4000})
    assert patched.status_code == 200
    assert patched.json()["config"]["enabled"] is False

    lookup = client.post("/v1/rag/lookup", json={"question": "phase 4", "session_id": session_id})
    assert lookup.status_code == 200
    assert lookup.json()["hits"][0]["citation"]["path"] == "notes.txt"
