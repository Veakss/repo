from __future__ import annotations

from frontend.ui_state import build_terminal_html, build_timeline_html, derive_pending_items, flatten_tree, merge_timeline


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
