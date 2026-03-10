from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pymongo import MongoClient
from pymongo.errors import PyMongoError

from continue_better_py.settings import get_settings


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class MongoStore:
    def __init__(self) -> None:
        settings = get_settings()
        self.client = MongoClient(
            settings.mongodb_uri,
            serverSelectionTimeoutMS=2000,
            connectTimeoutMS=2000,
        )
        self.db = self.client[settings.mongodb_database]
        self.artifacts_root = settings.resolved_artifact_root
        self.artifacts_root.joinpath("runs").mkdir(parents=True, exist_ok=True)

    def _require_connection(self) -> None:
        try:
            self.client.admin.command("ping")
        except PyMongoError as exc:
            raise RuntimeError("MongoDB is unavailable. Start a local MongoDB instance or update MONGODB_URI.") from exc

    def list_sessions(self) -> list[dict[str, Any]]:
        self._require_connection()
        return list(self.db.sessions.find({}, {"_id": 0}).sort("updated_at", -1))

    def create_session(self, title: str | None = None) -> dict[str, Any]:
        self._require_connection()
        session_id = str(uuid.uuid4())
        payload = {
            "id": session_id,
            "title": title or "New session",
            "created_at": now_iso(),
            "updated_at": now_iso(),
        }
        self.db.sessions.insert_one(payload)
        return payload

    def add_message(self, session_id: str, role: str, content: str) -> None:
        self._require_connection()
        self.db.messages.insert_one(
            {
                "session_id": session_id,
                "role": role,
                "content": content,
                "created_at": now_iso(),
            }
        )
        self.db.sessions.update_one({"id": session_id}, {"$set": {"updated_at": now_iso()}}, upsert=True)

    def get_messages(self, session_id: str) -> list[dict[str, Any]]:
        self._require_connection()
        rows = self.db.messages.find({"session_id": session_id}, {"_id": 0}).sort("created_at", 1)
        return list(rows)

    def start_run(self, run_id: str, session_id: str) -> None:
        self._require_connection()
        self.db.runs.update_one(
            {"run_id": run_id},
            {"$set": {"run_id": run_id, "session_id": session_id, "created_at": now_iso(), "updated_at": now_iso()}},
            upsert=True,
        )

    def append_run_event(self, run_id: str, event: dict[str, Any]) -> None:
        self._require_connection()
        self.db.run_events.insert_one({"run_id": run_id, **event})
        self.db.runs.update_one({"run_id": run_id}, {"$set": {"updated_at": now_iso()}})
        out = self.artifacts_root.joinpath("runs", f"{run_id}.jsonl")
        with out.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
