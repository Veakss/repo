from __future__ import annotations

from streamlit.testing.v1 import AppTest


def test_streamlit_app_renders_core_phase_three_panels():
    script = """
import streamlit as st
from frontend.app import run_app

class FakeClient:
    def capabilities(self):
        return {"webSearch": True, "rag": True, "interactiveTerminal": True, "clarification": True}

    def models(self):
        return {"models": [{"id": "google/gemini-2.5-flash-lite-preview-09-2025"}]}

    def list_sessions(self):
        return [{"id": "s1", "title": "Session One", "message_count": 2}]

    def create_session(self, title=None):
        return {"session_id": "s1"}

    def get_session_messages(self, session_id, limit=None):
        return [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]

    def list_runs(self, session_id):
        return [{"run_id": "run-1", "state": "awaiting_approval", "phase": "execute"}]

    def get_run(self, run_id):
        return {
            "run_id": "run-1",
            "events": [
                {"type": "run_state", "runId": "run-1", "state": "awaiting_approval", "timestamp": "2026-03-11T12:00:00+00:00"},
                {"type": "approval_required", "runId": "run-1", "actionId": "approval-1", "name": "write_file", "riskLevel": "risky", "arguments": "{}"},
            ],
        }

    def update_session(self, session_id, title):
        return {"id": session_id, "title": title}

    def delete_session(self, session_id):
        return {"ok": True}

    def fs_tree(self):
        return {"root": "/tmp", "children": [{"name": "notes.txt", "path": "notes.txt", "type": "file"}]}

    def fs_read(self, path):
        return {"path": path, "content": "hello workspace"}

    def stream_chat(self, payload):
        return iter([])

    def stream_approval(self, approval_id, decision):
        return iter([])

    def stream_clarification(self, clarification_id, answer):
        return iter([])

run_app(client_factory=lambda: FakeClient())
"""
    at = AppTest.from_string(script)
    at.run()

    assert any(item.value == "Continue Better" for item in at.title)
    assert any(metric.label == "Pending Approvals" and metric.value == "1" for metric in at.metric)
    assert any(button.label == "Approve" for button in at.button)
    assert any(button.label == "Reject" for button in at.button)
    assert any(tab.label == "Timeline" for tab in at.tabs)
    assert any(tab.label == "Files" for tab in at.tabs)
