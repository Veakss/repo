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

    def rag_list_profiles(self):
        return {"profiles": ["default"]}

    def rag_get_session_files(self, session_id):
        return {"files": [{"id": "file-1", "status": "done", "source_display_path": "notes.txt"}]}

    def rag_get_profile_files(self, profile_name):
        return {"files": []}

    def rag_list_index_jobs(self, limit=None):
        return {"jobs": [{"job_id": "job-1", "status": "done", "stage": "done", "scope_kind": "session", "scope_id": "s1", "progress": {"percent": 100}}]}

    def rag_get_session_memory(self, session_id, limit=None):
        return {"config": {"enabled": True, "thresholdPct": 0.9, "tokenBudget": 12000}, "summary": {"text": "", "entryCount": 0, "estimatedTokens": 0}, "entries": []}

    def rag_patch_session_memory(self, session_id, payload):
        return self.rag_get_session_memory(session_id)

    def rag_compact_session_memory(self, session_id):
        return self.rag_get_session_memory(session_id)

    def rag_clear_session_memory(self, session_id):
        return self.rag_get_session_memory(session_id)

    def rag_create_profile(self, name):
        return {"profiles": ["default", name]}

    def rag_rename_profile(self, profile_name, new_name):
        return {"profiles": [new_name]}

    def rag_delete_profile(self, profile_name):
        return {"profiles": []}

    def rag_import_session_file(self, session_id, path):
        return {"ok": True}

    def rag_import_profile_file(self, profile_name, path):
        return {"ok": True}

    def rag_enqueue_session_index_job(self, session_id):
        return {"job": {"job_id": "job-1"}}

    def rag_enqueue_profile_index_job(self, profile_name):
        return {"job": {"job_id": "job-2"}}

    def rag_delete_session_file(self, session_id, file_id):
        return {"ok": True}

    def rag_delete_profile_file(self, profile_name, file_id):
        return {"ok": True}

    def matrix_catalog(self):
        return {
            "scenarios": [{"id": "social_no_clarify"}, {"id": "read_file_phase_status"}],
            "profiles": [{"id": "baseline_current"}],
            "surfaces": [{"id": "backend_relay"}],
        }

    def matrix_list_reports(self, limit=None):
        return [{"report_id": "matrix-1", "generatedAt": "2026-03-11T12:00:00+00:00"}]

    def matrix_list_jobs(self, limit=None):
        return [{"job_id": "job-1", "status": "done", "progress": {"completed": 1, "total": 1, "label": "done"}, "report_id": "matrix-1"}]

    def matrix_get_report(self, report_id):
        return {
            "report_id": report_id,
            "models": ["google/gemini-2.5-flash-lite-preview-09-2025"],
            "profiles": ["baseline_current"],
            "surfaces": ["backend_relay"],
            "aggregate": {
                "byModelProfileSurface": {
                    "google/gemini-2.5-flash-lite-preview-09-2025::baseline_current::backend_relay": {
                        "runs": 1,
                        "passRate": 1.0,
                        "avgScore": 1.0,
                        "hard_fail": 0,
                    }
                }
            },
            "results": [
                {
                    "scenarioId": "social_no_clarify",
                    "model": "google/gemini-2.5-flash-lite-preview-09-2025",
                    "profile": "baseline_current",
                    "surface": "backend_relay",
                    "grade": {"overall": "pass", "score": 9, "maxScore": 9},
                    "summary": {"tools": [], "state": "completed", "latencyMs": 100},
                }
            ],
        }

    def matrix_start_job(self, payload):
        return {"job_id": "job-2", "status": "queued", "progress": {"completed": 0, "total": 1, "label": "queued"}}

    def matrix_compare(self, current_report_id, baseline_report_id):
        return {"summary": {"currentRuns": 1, "baselineRuns": 1}, "byModel": [{"key": "gemini"}]}

    def stream_chat(self, payload):
        return iter([])

    def stream_approval(self, approval_id, decision):
        return iter([])

    def stream_clarification(self, clarification_id, answer):
        return iter([])

run_app(client_factory=lambda: FakeClient())
"""
    at = AppTest.from_string(script, default_timeout=10)
    at.run(timeout=10)

    assert any(item.value == "Continue Better" for item in at.title)
    assert any(checkbox.label == "Web" for checkbox in at.checkbox)
    assert any(checkbox.label == "Clarify" for checkbox in at.checkbox)
    assert any(button.label == "Approve" for button in at.button)
    assert any(button.label == "Reject" for button in at.button)
    assert any(radio.label == "Inspector panel" for radio in at.radio)
    assert any(button.label == "New Session" for button in at.button)
