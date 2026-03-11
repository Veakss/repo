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
