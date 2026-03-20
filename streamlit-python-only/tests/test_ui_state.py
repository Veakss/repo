from __future__ import annotations

from frontend.ui_state import (
    build_persistent_progress_messages,
    build_terminal_html,
    build_timeline_html,
    derive_pending_items,
    flatten_tree,
    merge_timeline,
)


def test_merge_timeline_deduplicates_events():
    existing = [{"type": "run_state", "runId": "r1", "state": "planning"}]
    new_events = [
        {"type": "run_state", "runId": "r1", "state": "planning"},
        {"type": "approval_required", "runId": "r1", "actionId": "a1", "name": "write_file"},
    ]
    merged = merge_timeline(existing, new_events)
    assert len(merged) == 2


def test_derive_pending_items_tracks_approval_and_clarification_state():
    timeline = [
        {"type": "approval_required", "runId": "r1", "actionId": "a1", "name": "write_file", "riskLevel": "risky", "arguments": "{}"},
        {"type": "clarification_required", "runId": "r1", "clarificationId": "c1", "question": "Which file?", "questions": ["Which file?"], "options": []},
    ]
    approvals, clarification = derive_pending_items(timeline)
    assert approvals[0]["approval_id"] == "a1"
    assert clarification is not None
    assert clarification["clarification_id"] == "c1"

    approvals, clarification = derive_pending_items(
        timeline
        + [
            {"type": "approval_decision", "runId": "r1", "actionId": "a1", "decision": "approved"},
            {"type": "clarification_answered", "runId": "r1", "clarificationId": "c1", "answer": "Use app.py"},
        ]
    )
    assert approvals == []
    assert clarification is None


def test_derive_pending_items_ignores_replayed_resolved_clarification():
    timeline = [
        {"type": "clarification_required", "runId": "r1", "clarificationId": "c1", "question": "Which file?", "questions": ["Which file?"], "options": []},
        {"type": "clarification_answered", "runId": "r1", "clarificationId": "c1", "answer": "Use app.py"},
        {"type": "run_state", "runId": "r1", "state": "completed"},
        {"type": "clarification_required", "runId": "r1", "clarificationId": "c1", "question": "Which file?", "questions": ["Which file?"], "options": []},
    ]
    approvals, clarification = derive_pending_items(timeline)
    assert approvals == []
    assert clarification is None


def test_flatten_tree_and_timeline_html():
    tree = {
        "root": "/tmp/demo",
        "children": [
            {"name": "src", "path": "src", "type": "directory", "children": [{"name": "app.py", "path": "src/app.py", "type": "file"}]},
            {"name": "README.md", "path": "README.md", "type": "file"},
        ],
    }
    flattened = flatten_tree(tree)
    assert flattened == ["src/app.py", "README.md"]

    html = build_timeline_html([{"type": "run_state", "runId": "r1", "state": "completed", "timestamp": "2026-03-11T12:00:00+00:00"}])
    assert "State -&gt; completed" in html

    terminal_html = build_terminal_html(
        [
            {"type": "terminal_opened", "runId": "r1", "terminalId": "t1", "command": "pwd", "cwd": "/tmp", "timestamp": "2026-03-11T12:00:00+00:00"},
            {"type": "terminal_exit", "runId": "r1", "terminalId": "t1", "exitCode": 0, "output": "/tmp", "timestamp": "2026-03-11T12:00:01+00:00"},
        ]
    )
    assert "pwd" in terminal_html
    assert "exit=0" in terminal_html


def test_timeline_html_renders_progress_and_run_step():
    html = build_timeline_html(
        [
            {"type": "assistant_progress", "runId": "r1", "stepIndex": 2, "summary": "Action terminée: `read_file` réussi.", "source": "runtime"},
            {"type": "run_step", "runId": "r1", "stepIndex": 2, "kind": "tool_result", "status": "ok", "summary": "Tool `read_file` succeeded."},
        ]
    )
    assert "Assistant progress #2" in html
    assert "Run step #2 (tool_result)" in html


def test_build_persistent_progress_messages_dedupes_and_keeps_order():
    timeline = [
        {"type": "assistant_progress", "runId": "r1", "stepIndex": 1, "summary": "Action en cours: appel de `web_search`."},
        {"type": "assistant_progress", "runId": "r1", "stepIndex": 1, "summary": "Action en cours: appel de `web_search`."},
        {"type": "assistant_progress", "runId": "r1", "stepIndex": 1, "summary": "Action terminée: `web_search` réussi."},
        {"type": "assistant_progress", "runId": "r1", "stepIndex": 5, "summary": "Action en cours: appel de `write_file`."},
        {"type": "assistant_progress", "runId": "r1", "stepIndex": 6, "summary": "Vérification: mise à jour du gap objectif/réalisation."},
    ]
    messages = build_persistent_progress_messages(timeline)
    assert len(messages) == 3
    assert messages[0]["kind"] == "progress"
    assert messages[0]["content"].startswith("**Étape 1**")
    assert "- Action en cours: appel de `web_search`." in messages[0]["content"]
    assert "- Action terminée: `web_search` réussi." in messages[0]["content"]
    assert messages[1]["content"].startswith("**Étape 2**")
    assert "- Action en cours: appel de `write_file`." in messages[1]["content"]
    assert messages[2]["content"].startswith("**Étape 3**")
    assert "- Vérification: mise à jour du gap objectif/réalisation." in messages[2]["content"]
