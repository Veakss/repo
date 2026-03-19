from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pymongo import MongoClient, ReturnDocument
from pymongo.errors import PyMongoError

from streamlit_python_only.settings import Settings, get_settings


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class MongoStore:
    def __init__(
        self,
        client: Any | None = None,
        settings: Settings | None = None,
        database_name: str | None = None,
        artifacts_root: Path | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.client = client or MongoClient(
            self.settings.mongodb_uri,
            serverSelectionTimeoutMS=2000,
            connectTimeoutMS=2000,
        )
        self.db = self.client[database_name or self.settings.mongodb_database]
        self.artifacts_root = (artifacts_root or self.settings.resolved_artifact_root).resolve()
        self.runs_root = self.artifacts_root.joinpath("runs")
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self._ensure_indexes()

    def _require_connection(self) -> None:
        try:
            self.client.admin.command("ping")
        except PyMongoError as exc:
            raise RuntimeError("MongoDB is unavailable. Start a local MongoDB instance or update MONGODB_URI.") from exc

    def _ensure_indexes(self) -> None:
        try:
            self.db.sessions.create_index("id", unique=True)
            self.db.messages.create_index([("session_id", 1), ("created_at", 1)])
            self.db.runs.create_index("run_id", unique=True)
            self.db.runs.create_index([("session_id", 1), ("updated_at", -1)])
            self.db.run_events.create_index([("run_id", 1), ("sequence", 1)], unique=True)
            self.db.pending_approvals.create_index("approval_id", unique=True)
            self.db.pending_clarifications.create_index("clarification_id", unique=True)
        except Exception:
            # Index creation should not prevent tests or offline bootstrapping.
            pass

    @staticmethod
    def _clean(document: dict[str, Any] | None) -> dict[str, Any] | None:
        if not document:
            return None
        cleaned = dict(document)
        cleaned.pop("_id", None)
        return cleaned

    def list_sessions(self) -> list[dict[str, Any]]:
        self._require_connection()
        rows = self.db.sessions.find({}, {"_id": 0}).sort("updated_at", -1)
        return list(rows)

    def create_session(self, title: str | None = None) -> dict[str, Any]:
        self._require_connection()
        session_id = str(uuid.uuid4())
        payload = {
            "id": session_id,
            "title": title or "New session",
            "title_source": "user" if title else "system",
            "created_at": now_iso(),
            "updated_at": now_iso(),
            "last_run_id": None,
            "message_count": 0,
        }
        self.db.sessions.insert_one(dict(payload))
        return payload

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        self._require_connection()
        return self._clean(self.db.sessions.find_one({"id": session_id}))

    def update_session_title(self, session_id: str, title: str) -> dict[str, Any]:
        self._require_connection()
        trimmed = title.strip()
        if not trimmed:
            raise ValueError("Session title cannot be empty")
        result = self.db.sessions.find_one_and_update(
            {"id": session_id},
            {"$set": {"title": trimmed, "title_source": "user", "updated_at": now_iso()}},
            return_document=ReturnDocument.AFTER,
        )
        if not result:
            raise KeyError(session_id)
        return self._clean(result) or {}

    def delete_session(self, session_id: str) -> dict[str, Any]:
        self._require_connection()
        session = self.get_session(session_id)
        if not session:
            raise KeyError(session_id)
        run_ids = [row["run_id"] for row in self.db.runs.find({"session_id": session_id}, {"_id": 0, "run_id": 1})]
        self.db.sessions.delete_one({"id": session_id})
        self.db.messages.delete_many({"session_id": session_id})
        if run_ids:
            self.db.runs.delete_many({"run_id": {"$in": run_ids}})
            self.db.run_events.delete_many({"run_id": {"$in": run_ids}})
            self.db.pending_approvals.delete_many({"run_id": {"$in": run_ids}})
            self.db.pending_clarifications.delete_many({"run_id": {"$in": run_ids}})
        for run_id in run_ids:
            self._events_path(run_id).unlink(missing_ok=True)
            self._meta_path(run_id).unlink(missing_ok=True)
        return {"ok": True, "deleted_session_id": session_id, "deleted_run_ids": run_ids}

    def add_message(self, session_id: str, role: str, content: str, run_id: str | None = None) -> dict[str, Any]:
        self._require_connection()
        if not self.get_session(session_id):
            raise KeyError(session_id)
        payload = {
            "id": str(uuid.uuid4()),
            "session_id": session_id,
            "role": role,
            "content": content,
            "run_id": run_id,
            "created_at": now_iso(),
        }
        self.db.messages.insert_one(dict(payload))
        self.db.sessions.update_one(
            {"id": session_id},
            {
                "$set": {"updated_at": payload["created_at"]},
                "$inc": {"message_count": 1},
            },
        )
        return payload

    def get_messages(self, session_id: str, limit: int | None = None) -> list[dict[str, Any]]:
        self._require_connection()
        cursor = self.db.messages.find({"session_id": session_id}).sort("created_at", 1)
        rows = [self._clean(row) or {} for row in cursor]
        return rows[-limit:] if limit else rows

    def start_run(self, run_id: str, session_id: str, meta_fields: dict[str, Any] | None = None) -> dict[str, Any]:
        self._require_connection()
        session = self.get_session(session_id)
        if not session:
            raise KeyError(session_id)
        existing = self._clean(self.db.runs.find_one({"run_id": run_id}))
        if existing:
            existing["session_id"] = session_id
            existing["updated_at"] = now_iso()
            if meta_fields:
                existing.update(meta_fields)
            self.db.runs.replace_one({"run_id": run_id}, existing, upsert=True)
            self._write_run_meta(existing)
            return existing

        ts = now_iso()
        payload = self._default_run_meta(run_id=run_id, session_id=session_id, created_at=ts)
        if meta_fields:
            payload.update(meta_fields)
        self.db.runs.insert_one(dict(payload))
        self.db.sessions.update_one(
            {"id": session_id},
            {"$set": {"updated_at": ts, "last_run_id": run_id}},
        )
        self._write_run_meta(payload)
        return payload

    def append_run_event(self, run_id: str, event: dict[str, Any]) -> dict[str, Any]:
        self._require_connection()
        run = self._clean(self.db.runs.find_one({"run_id": run_id}))
        if not run:
            run = self.start_run(run_id=run_id, session_id=str(event.get("sessionId") or ""))
        sequence = int(run.get("event_count", 0)) + 1
        event_doc = {
            "run_id": run_id,
            "session_id": run.get("session_id"),
            "sequence": sequence,
            "event": event,
            "created_at": event.get("timestamp") or now_iso(),
        }
        self.db.run_events.insert_one(dict(event_doc))
        updated = dict(run)
        updated["event_count"] = sequence
        self._apply_meta_update(updated, event)
        self.db.runs.replace_one({"run_id": run_id}, updated, upsert=True)
        if updated.get("session_id"):
            self.db.sessions.update_one(
                {"id": updated["session_id"]},
                {"$set": {"updated_at": updated.get("updated_at", now_iso()), "last_run_id": run_id}},
            )
        self._update_pending_state(updated, event)
        self._append_event_to_disk(run_id, event)
        self._write_run_meta(updated)
        return updated

    def list_runs_for_session(self, session_id: str) -> list[dict[str, Any]]:
        self._require_connection()
        cursor = self.db.runs.find({"session_id": session_id}).sort("updated_at", -1)
        return [self._clean(row) or {} for row in cursor]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        self._require_connection()
        meta = self._clean(self.db.runs.find_one({"run_id": run_id}))
        if not meta:
            return None
        events = self.get_run_events(run_id)
        return {"run_id": run_id, "session_id": meta.get("session_id"), "state": meta.get("state"), "phase": meta.get("phase"), "meta": meta, "events": events}

    def get_run_events(self, run_id: str) -> list[dict[str, Any]]:
        self._require_connection()
        cursor = self.db.run_events.find({"run_id": run_id}, {"_id": 0, "event": 1}).sort("sequence", 1)
        return [row["event"] for row in cursor]

    def get_approval(self, approval_id: str) -> dict[str, Any] | None:
        self._require_connection()
        return self._clean(self.db.pending_approvals.find_one({"approval_id": approval_id}))

    def get_clarification(self, clarification_id: str) -> dict[str, Any] | None:
        self._require_connection()
        return self._clean(self.db.pending_clarifications.find_one({"clarification_id": clarification_id}))

    def record_approval_decision(self, approval_id: str, decision: str) -> None:
        self._require_connection()
        self.db.pending_approvals.update_one(
            {"approval_id": approval_id},
            {"$set": {"status": decision, "decision": decision, "resolved_at": now_iso(), "updated_at": now_iso()}},
            upsert=True,
        )

    def record_clarification_answer(self, clarification_id: str, answer: str) -> None:
        self._require_connection()
        self.db.pending_clarifications.update_one(
            {"clarification_id": clarification_id},
            {"$set": {"status": "answered", "answer": answer, "resolved_at": now_iso(), "updated_at": now_iso()}},
            upsert=True,
        )

    def _default_run_meta(self, run_id: str, session_id: str, created_at: str | None = None) -> dict[str, Any]:
        ts = created_at or now_iso()
        return {
            "run_id": run_id,
            "session_id": session_id,
            "state": "planning",
            "phase": "planning",
            "created_at": ts,
            "updated_at": ts,
            "last_event_at": None,
            "event_count": 0,
            "attempt": 0,
            "retry_count": 0,
            "no_progress_count": 0,
            "diagnostic_count": 0,
            "terminal_ids": [],
            "last_tool_call": None,
            "tool_calls": 0,
            "tool_errors": 0,
            "approvals_required": 0,
            "approvals_approved": 0,
            "approvals_rejected": 0,
            "clarifications_required": 0,
            "clarifications_answered": 0,
            "active_approval_id": None,
            "active_clarification_id": None,
            "outcome": None,
            "failure_code": None,
            "run_trace": None,
        }

    def _apply_meta_update(self, meta: dict[str, Any], event: dict[str, Any]) -> None:
        event_type = event.get("type")
        ts = event.get("timestamp")
        if isinstance(ts, str) and ts:
            meta["last_event_at"] = ts
            meta["updated_at"] = ts
        else:
            meta["updated_at"] = now_iso()

        if event_type == "run_state":
            meta["state"] = event.get("state", meta.get("state", "unknown"))
        elif event_type == "run_phase_changed":
            phase = event.get("phase", meta.get("phase", "unknown"))
            meta["phase"] = phase
            if phase == "execute":
                meta["attempt"] = int(meta.get("attempt", 0)) + 1
            elif phase == "repair":
                meta["retry_count"] = int(meta.get("retry_count", 0)) + 1
        elif event_type == "run_diagnostic":
            meta["diagnostic_count"] = int(meta.get("diagnostic_count", 0)) + 1
            if event.get("code") == "no_progress":
                meta["no_progress_count"] = int(meta.get("no_progress_count", 0)) + 1
        elif event_type == "tool_call":
            meta["tool_calls"] = int(meta.get("tool_calls", 0)) + 1
            meta["last_tool_call"] = {
                "action_id": event.get("actionId"),
                "name": event.get("name"),
                "arguments": event.get("arguments"),
                "timestamp": ts,
            }
        elif event_type == "tool_result":
            if event.get("ok") is False:
                meta["tool_errors"] = int(meta.get("tool_errors", 0)) + 1
        elif event_type == "approval_required":
            meta["approvals_required"] = int(meta.get("approvals_required", 0)) + 1
            meta["active_approval_id"] = event.get("actionId")
        elif event_type == "approval_decision":
            decision = event.get("decision")
            meta["active_approval_id"] = None
            if decision == "approved":
                meta["approvals_approved"] = int(meta.get("approvals_approved", 0)) + 1
            elif decision == "rejected":
                meta["approvals_rejected"] = int(meta.get("approvals_rejected", 0)) + 1
        elif event_type == "clarification_required":
            meta["clarifications_required"] = int(meta.get("clarifications_required", 0)) + 1
            meta["active_clarification_id"] = event.get("clarificationId")
        elif event_type == "clarification_answered":
            meta["clarifications_answered"] = int(meta.get("clarifications_answered", 0)) + 1
            meta["active_clarification_id"] = None
        elif event_type == "done":
            trace = event.get("runTrace")
            if isinstance(trace, dict):
                outcome = trace.get("outcome")
                failure = trace.get("failure")
                meta["run_trace"] = trace
                if isinstance(outcome, str) and outcome:
                    meta["outcome"] = outcome
                if isinstance(failure, dict):
                    failure_code = failure.get("code")
                    if isinstance(failure_code, str) and failure_code:
                        meta["failure_code"] = failure_code

        if event_type in {"terminal_opened", "terminal_control_changed", "terminal_exit", "terminal_error"}:
            terminal_id = event.get("terminalId")
            if isinstance(terminal_id, str) and terminal_id:
                terminals = meta.get("terminal_ids")
                if not isinstance(terminals, list):
                    terminals = []
                    meta["terminal_ids"] = terminals
                if terminal_id not in terminals:
                    terminals.append(terminal_id)

    def _update_pending_state(self, run: dict[str, Any], event: dict[str, Any]) -> None:
        event_type = event.get("type")
        if event_type == "approval_required":
            approval_id = event.get("actionId")
            if isinstance(approval_id, str) and approval_id:
                self.db.pending_approvals.update_one(
                    {"approval_id": approval_id},
                    {
                        "$set": {
                            "approval_id": approval_id,
                            "run_id": run["run_id"],
                            "session_id": run.get("session_id"),
                            "status": "pending",
                            "tool_name": event.get("name"),
                            "arguments": event.get("arguments"),
                            "risk_level": event.get("riskLevel"),
                            "created_at": event.get("timestamp") or now_iso(),
                            "updated_at": event.get("timestamp") or now_iso(),
                        }
                    },
                    upsert=True,
                )
        elif event_type == "approval_decision":
            approval_id = event.get("actionId")
            if isinstance(approval_id, str) and approval_id:
                self.db.pending_approvals.update_one(
                    {"approval_id": approval_id},
                    {
                        "$set": {
                            "status": event.get("decision"),
                            "decision": event.get("decision"),
                            "resolved_at": event.get("timestamp") or now_iso(),
                            "updated_at": event.get("timestamp") or now_iso(),
                        }
                    },
                    upsert=True,
                )
        elif event_type == "clarification_required":
            clarification_id = event.get("clarificationId")
            if isinstance(clarification_id, str) and clarification_id:
                self.db.pending_clarifications.update_one(
                    {"clarification_id": clarification_id},
                    {
                        "$set": {
                            "clarification_id": clarification_id,
                            "run_id": run["run_id"],
                            "session_id": run.get("session_id"),
                            "status": "pending",
                            "question": event.get("question"),
                            "questions": event.get("questions", []),
                            "options": event.get("options", []),
                            "created_at": event.get("timestamp") or now_iso(),
                            "updated_at": event.get("timestamp") or now_iso(),
                        }
                    },
                    upsert=True,
                )
        elif event_type == "clarification_answered":
            clarification_id = event.get("clarificationId")
            if isinstance(clarification_id, str) and clarification_id:
                self.db.pending_clarifications.update_one(
                    {"clarification_id": clarification_id},
                    {
                        "$set": {
                            "status": "answered",
                            "answer": event.get("answer"),
                            "resolved_at": event.get("timestamp") or now_iso(),
                            "updated_at": event.get("timestamp") or now_iso(),
                        }
                    },
                    upsert=True,
                )

    def _events_path(self, run_id: str) -> Path:
        return self.runs_root.joinpath(f"{run_id}.jsonl")

    def _meta_path(self, run_id: str) -> Path:
        return self.runs_root.joinpath(f"{run_id}.meta.json")

    def _append_event_to_disk(self, run_id: str, event: dict[str, Any]) -> None:
        with self._events_path(run_id).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _write_run_meta(self, meta: dict[str, Any]) -> None:
        run_id = str(meta["run_id"])
        self._meta_path(run_id).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
