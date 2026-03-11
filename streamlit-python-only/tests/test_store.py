from __future__ import annotations

import json

import mongomock

from continue_better_py.store import MongoStore


def build_store(tmp_path):
    return MongoStore(
        client=mongomock.MongoClient(),
        database_name="continue_better_python_test",
        artifacts_root=tmp_path,
    )


def test_store_persists_sessions_runs_and_json_mirrors(tmp_path):
    store = build_store(tmp_path)
    session = store.create_session("Alpha")
    store.add_message(session["id"], "user", "hello")
    store.add_message(session["id"], "assistant", "hi there")

    store.start_run(
        "run-1",
        session["id"],
        meta_fields={"requested_model": "google/gemini-2.5-flash-lite-preview-09-2025"},
    )
    store.append_run_event("run-1", {"type": "run_state", "runId": "run-1", "state": "planning", "timestamp": "2026-03-11T10:00:00+00:00"})
    store.append_run_event("run-1", {"type": "run_phase_changed", "runId": "run-1", "phase": "execute", "timestamp": "2026-03-11T10:00:01+00:00"})
    store.append_run_event(
        "run-1",
        {
            "type": "tool_call",
            "runId": "run-1",
            "actionId": "action-1",
            "name": "write_file",
            "arguments": "{\"path\":\"test.txt\"}",
            "timestamp": "2026-03-11T10:00:02+00:00",
        },
    )
    store.append_run_event(
        "run-1",
        {
            "type": "approval_required",
            "runId": "run-1",
            "actionId": "approval-1",
            "name": "write_file",
            "arguments": "{\"path\":\"test.txt\"}",
            "riskLevel": "risky",
            "timestamp": "2026-03-11T10:00:03+00:00",
        },
    )
    store.append_run_event(
        "run-1",
        {
            "type": "approval_decision",
            "runId": "run-1",
            "actionId": "approval-1",
            "decision": "approved",
            "timestamp": "2026-03-11T10:00:04+00:00",
        },
    )
    store.append_run_event(
        "run-1",
        {
            "type": "clarification_required",
            "runId": "run-1",
            "clarificationId": "clar-1",
            "question": "Which file?",
            "questions": ["Which file?"],
            "options": [{"label": "main.py"}],
            "timestamp": "2026-03-11T10:00:05+00:00",
        },
    )
    store.record_clarification_answer("clar-1", "Use main.py")
    store.append_run_event(
        "run-1",
        {
            "type": "clarification_answered",
            "runId": "run-1",
            "clarificationId": "clar-1",
            "answer": "Use main.py",
            "timestamp": "2026-03-11T10:00:06+00:00",
        },
    )
    store.append_run_event("run-1", {"type": "run_state", "runId": "run-1", "state": "completed", "timestamp": "2026-03-11T10:00:07+00:00"})

    run = store.get_run("run-1")
    assert run is not None
    assert run["meta"]["state"] == "completed"
    assert run["meta"]["tool_calls"] == 1
    assert run["meta"]["approvals_required"] == 1
    assert run["meta"]["approvals_approved"] == 1
    assert run["meta"]["clarifications_required"] == 1
    assert run["meta"]["clarifications_answered"] == 1
    assert len(run["events"]) == 8

    approval = store.get_approval("approval-1")
    clarification = store.get_clarification("clar-1")
    assert approval is not None and approval["status"] == "approved"
    assert clarification is not None and clarification["status"] == "answered"

    listed_runs = store.list_runs_for_session(session["id"])
    assert [row["run_id"] for row in listed_runs] == ["run-1"]

    meta_path = tmp_path.joinpath("runs", "run-1.meta.json")
    events_path = tmp_path.joinpath("runs", "run-1.jsonl")
    assert meta_path.exists()
    assert events_path.exists()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    event_lines = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    assert meta["state"] == "completed"
    assert len(event_lines) == 8


def test_delete_session_cascades_run_state_and_artifacts(tmp_path):
    store = build_store(tmp_path)
    session = store.create_session("Delete Me")
    store.add_message(session["id"], "user", "hello")
    store.start_run("run-delete", session["id"])
    store.append_run_event("run-delete", {"type": "run_state", "runId": "run-delete", "state": "completed", "timestamp": "2026-03-11T10:00:00+00:00"})

    result = store.delete_session(session["id"])

    assert result["deleted_session_id"] == session["id"]
    assert store.get_session(session["id"]) is None
    assert store.get_run("run-delete") is None
    assert not tmp_path.joinpath("runs", "run-delete.meta.json").exists()
    assert not tmp_path.joinpath("runs", "run-delete.jsonl").exists()
