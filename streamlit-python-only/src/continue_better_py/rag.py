from __future__ import annotations

import asyncio
import hashlib
import json
import re
import uuid
from pathlib import Path
from typing import Any, Literal

from pymongo import MongoClient, ReturnDocument
from pymongo.errors import PyMongoError

from continue_better_py.settings import Settings, get_settings

RAG_SUPPORTED_FILE_EXTENSIONS_V1 = [".md", ".txt", ".json", ".yml", ".yaml"]
RAG_SOURCE_PRIORITY_DEFAULT_V1 = ["session_docs", "session_memory", "profiles"]

ScopeKind = Literal["session", "profile"]


def now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def assert_safe_segment(label: str, value: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{label} is required")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", normalized):
        raise ValueError(f"{label} contains invalid characters")
    if normalized in {".", ".."}:
        raise ValueError(f"{label} is invalid")
    return normalized


def scope_key(scope_kind: ScopeKind, scope_id: str) -> str:
    return f"{scope_kind}:{scope_id}"


def chunk_text_with_lines(input_text: str, max_len: int = 800) -> list[dict[str, Any]]:
    lines = input_text.splitlines() or [input_text]
    chunks: list[dict[str, Any]] = []
    buffer: list[str] = []
    start_line = 1
    current_len = 0

    def flush() -> None:
        nonlocal buffer, current_len, start_line
        if not buffer:
            return
        chunks.append(
            {
                "text": "\n".join(buffer),
                "lineStart": start_line,
                "lineEnd": start_line + len(buffer) - 1,
            }
        )
        buffer = []
        current_len = 0

    for index, line in enumerate(lines, start=1):
        if not buffer:
            buffer = [line]
            start_line = index
            current_len = len(line)
            continue
        next_len = current_len + 1 + len(line)
        if next_len <= max_len:
            buffer.append(line)
            current_len = next_len
            continue
        flush()
        buffer = [line]
        start_line = index
        current_len = len(line)
    flush()
    return [chunk for chunk in chunks if chunk["text"].strip()]


def tokenize(value: str) -> list[str]:
    return list(
        {
            token
            for token in re.split(r"[^a-zA-Z0-9_àâçéèêëîïôûùüÿñæœ-]+", str(value or "").lower())
            if len(token.strip()) >= 2
        }
    )


def score_chunk(query: str, query_tokens: list[str], text: str) -> float:
    hay = text.lower()
    score = 0.0
    for token in query_tokens:
        if token in hay:
            score += 1.0
            score += min(3, max(0, hay.count(token) - 1)) * 0.2
    if query.lower() in hay:
        score += 2.0
    return score


def estimate_tokens(text: str) -> int:
    words = len([token for token in str(text or "").split() if token.strip()])
    return max(0, int(words * 1.35))


def normalize_source_priority(value: list[str] | None = None) -> list[str]:
    seeds = value or list(RAG_SOURCE_PRIORITY_DEFAULT_V1)
    out: list[str] = []
    for item in seeds:
        if item in RAG_SOURCE_PRIORITY_DEFAULT_V1 and item not in out:
            out.append(item)
    for default in RAG_SOURCE_PRIORITY_DEFAULT_V1:
        if default not in out:
            out.append(default)
    return out


def build_transparency_message(response: dict[str, Any]) -> str:
    status = response.get("status")
    if status == "no_hits":
        return "No relevant information was found in the active RAG sources."
    if status == "indexing":
        return "RAG sources are still indexing. Retry shortly or continue without RAG."
    if status == "disabled":
        return "Session memory is disabled and no active indexed RAG source is available."
    if status == "error":
        error = response.get("error", {})
        suffix = f" ({error.get('message')})" if error.get("message") else ""
        return f"RAG is unavailable right now{suffix}."
    return ""


def render_lookup_for_model(response: dict[str, Any]) -> str:
    if response.get("status") != "ok":
        return build_transparency_message(response)
    parts = ["RAG context:"]
    for hit in response.get("hits", [])[:5]:
        citation = hit.get("citation", {})
        path = citation.get("path", "unknown")
        line_start = citation.get("lineStart")
        line_end = citation.get("lineEnd")
        if line_start and line_end and line_start != line_end:
            label = f"{path}:{line_start}-{line_end}"
        elif line_start:
            label = f"{path}:{line_start}"
        else:
            label = path
        parts.append(f"- {label}\n{hit.get('snippet', '').strip()}")
    return "\n\n".join(parts)


class RagService:
    def __init__(
        self,
        settings: Settings | None = None,
        client: Any | None = None,
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
        self.rag_root = self.artifacts_root.joinpath("rag")
        self.rag_root.mkdir(parents=True, exist_ok=True)
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        try:
            self.db.rag_profiles.create_index("name", unique=True)
            self.db.rag_files.create_index([("scope_kind", 1), ("scope_id", 1), ("sha256", 1)], unique=True)
            self.db.rag_files.create_index([("scope_kind", 1), ("scope_id", 1), ("imported_at", -1)])
            self.db.rag_chunks.create_index([("scope_kind", 1), ("scope_id", 1)])
            self.db.rag_memory_entries.create_index([("session_id", 1), ("record_type", 1), ("ts", -1)])
            self.db.rag_index_jobs.create_index("job_id", unique=True)
            self.db.rag_index_jobs.create_index([("created_at", -1)])
        except Exception:
            pass

    def _require_connection(self) -> None:
        try:
            self.client.admin.command("ping")
        except PyMongoError as exc:
            raise RuntimeError("MongoDB is unavailable for RAG operations.") from exc

    @staticmethod
    def _clean(document: dict[str, Any] | None) -> dict[str, Any] | None:
        if not document:
            return None
        output = dict(document)
        output.pop("_id", None)
        return output

    def _scope_artifact_root(self, scope_kind: ScopeKind, scope_id: str) -> Path:
        return self.rag_root.joinpath(scope_kind, scope_id)

    def _write_scope_artifacts(self, scope_kind: ScopeKind, scope_id: str) -> None:
        root = self._scope_artifact_root(scope_kind, scope_id)
        root.mkdir(parents=True, exist_ok=True)
        files = [self._clean(row) or {} for row in self.db.rag_files.find({"scope_kind": scope_kind, "scope_id": scope_id}).sort("imported_at", -1)]
        chunks = [self._clean(row) or {} for row in self.db.rag_chunks.find({"scope_kind": scope_kind, "scope_id": scope_id}).sort("chunk_index", 1)]
        root.joinpath("files.json").write_text(json.dumps({"files": files}, ensure_ascii=False, indent=2), encoding="utf-8")
        with root.joinpath("chunks.jsonl").open("w", encoding="utf-8") as handle:
            for row in chunks:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _memory_config(self, session_id: str) -> dict[str, Any]:
        doc = self._clean(
            self.db.rag_memory_entries.find_one({"session_id": session_id, "record_type": "config"})
        )
        if doc:
            return {
                "enabled": bool(doc.get("enabled", True)),
                "thresholdPct": float(doc.get("thresholdPct", 0.9)),
                "tokenBudget": int(doc.get("tokenBudget", 12000)),
            }
        config = {"enabled": True, "thresholdPct": 0.9, "tokenBudget": 12000}
        self.db.rag_memory_entries.insert_one({"session_id": session_id, "record_type": "config", **config, "updated_at": now_iso()})
        return config

    def _memory_summary(self, session_id: str) -> dict[str, Any]:
        doc = self._clean(
            self.db.rag_memory_entries.find_one({"session_id": session_id, "record_type": "summary"})
        )
        if doc:
            return {
                "text": str(doc.get("text", "")),
                "updatedAt": doc.get("updatedAt") or doc.get("updated_at") or now_iso(),
                "lastCompactedAt": doc.get("lastCompactedAt"),
                "estimatedTokens": int(doc.get("estimatedTokens", 0)),
                "entryCount": int(doc.get("entryCount", 0)),
            }
        summary = {"text": "", "updatedAt": now_iso(), "estimatedTokens": 0, "entryCount": 0}
        self.db.rag_memory_entries.insert_one({"session_id": session_id, "record_type": "summary", **summary})
        return summary

    def list_profiles(self) -> list[str]:
        self._require_connection()
        rows = self.db.rag_profiles.find({}, {"_id": 0, "name": 1}).sort("name", 1)
        return [row["name"] for row in rows]

    def create_profile(self, name: str) -> list[str]:
        self._require_connection()
        safe_name = assert_safe_segment("profileName", name)
        now = now_iso()
        self.db.rag_profiles.update_one(
            {"name": safe_name},
            {"$setOnInsert": {"name": safe_name, "created_at": now, "updated_at": now}},
            upsert=True,
        )
        return self.list_profiles()

    def rename_profile(self, old_name: str, new_name: str) -> list[str]:
        self._require_connection()
        source = assert_safe_segment("profileName", old_name)
        target = assert_safe_segment("profileName", new_name)
        if not self.db.rag_profiles.find_one({"name": source}):
            raise KeyError(source)
        if self.db.rag_profiles.find_one({"name": target}):
            raise ValueError("Profile already exists")
        self.db.rag_profiles.update_one({"name": source}, {"$set": {"name": target, "updated_at": now_iso()}})
        self.db.rag_files.update_many({"scope_kind": "profile", "scope_id": source}, {"$set": {"scope_id": target}})
        self.db.rag_chunks.update_many({"scope_kind": "profile", "scope_id": source}, {"$set": {"scope_id": target}})
        self.db.rag_index_jobs.update_many({"scope_kind": "profile", "scope_id": source}, {"$set": {"scope_id": target}})
        old_root = self._scope_artifact_root("profile", source)
        new_root = self._scope_artifact_root("profile", target)
        if old_root.exists():
            new_root.parent.mkdir(parents=True, exist_ok=True)
            old_root.rename(new_root)
        return self.list_profiles()

    def delete_profile(self, name: str) -> list[str]:
        self._require_connection()
        safe_name = assert_safe_segment("profileName", name)
        self.db.rag_profiles.delete_one({"name": safe_name})
        self.db.rag_files.delete_many({"scope_kind": "profile", "scope_id": safe_name})
        self.db.rag_chunks.delete_many({"scope_kind": "profile", "scope_id": safe_name})
        self.db.rag_index_jobs.delete_many({"scope_kind": "profile", "scope_id": safe_name})
        root = self._scope_artifact_root("profile", safe_name)
        if root.exists():
            import shutil

            shutil.rmtree(root, ignore_errors=True)
        return self.list_profiles()

    def _ensure_profile_scope(self, scope_kind: ScopeKind, scope_id: str) -> None:
        if scope_kind == "profile":
            self.create_profile(scope_id)

    def import_file(self, scope_kind: ScopeKind, scope_id: str, source_path: str, display_path: str | None = None) -> dict[str, Any]:
        self._require_connection()
        safe_scope = assert_safe_segment("scopeId", scope_id)
        self._ensure_profile_scope(scope_kind, safe_scope)
        source = Path(source_path).resolve()
        if not source.exists() or not source.is_file():
            raise FileNotFoundError(source_path)
        extension = source.suffix.lower()
        if extension not in RAG_SUPPORTED_FILE_EXTENSIONS_V1:
            raise ValueError(f"Unsupported RAG file extension: {extension}")
        content = source.read_text(encoding="utf-8")
        sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
        existing = self._clean(
            self.db.rag_files.find_one({"scope_kind": scope_kind, "scope_id": safe_scope, "sha256": sha})
        )
        if existing:
            return {"imported": False, "deduped": True, "entry": existing, "manifest": self.list_files(scope_kind, safe_scope)}
        entry = {
            "id": str(uuid.uuid4()),
            "scope_kind": scope_kind,
            "scope_id": safe_scope,
            "original_name": source.name,
            "source_display_path": display_path or source.name,
            "sha256": sha,
            "size_bytes": len(content.encode("utf-8")),
            "extension": extension,
            "status": "pending",
            "imported_at": now_iso(),
            "content": content,
            "error": None,
        }
        self.db.rag_files.insert_one(dict(entry))
        self._write_scope_artifacts(scope_kind, safe_scope)
        return {"imported": True, "deduped": False, "entry": entry, "manifest": self.list_files(scope_kind, safe_scope)}

    def list_files(self, scope_kind: ScopeKind, scope_id: str) -> dict[str, Any]:
        self._require_connection()
        safe_scope = assert_safe_segment("scopeId", scope_id)
        rows = [self._clean(row) or {} for row in self.db.rag_files.find({"scope_kind": scope_kind, "scope_id": safe_scope}).sort("imported_at", -1)]
        return {
            "scope": {"kind": scope_kind, "id": safe_scope},
            "files": rows,
            "stats": {
                "fileCount": len(rows),
                "totalBytes": sum(int(row.get("size_bytes", 0)) for row in rows),
            },
        }

    def delete_file(self, scope_kind: ScopeKind, scope_id: str, file_id: str) -> dict[str, Any]:
        self._require_connection()
        safe_scope = assert_safe_segment("scopeId", scope_id)
        self.db.rag_files.delete_one({"scope_kind": scope_kind, "scope_id": safe_scope, "id": file_id})
        self.db.rag_chunks.delete_many({"scope_kind": scope_kind, "scope_id": safe_scope, "file_id": file_id})
        self._write_scope_artifacts(scope_kind, safe_scope)
        return {"deleted": True, **self.list_files(scope_kind, safe_scope)}

    def _update_file_status(self, scope_kind: ScopeKind, scope_id: str, file_id: str, status: str, error: str | None = None) -> None:
        self.db.rag_files.update_one(
            {"scope_kind": scope_kind, "scope_id": scope_id, "id": file_id},
            {"$set": {"status": status, "error": error, "updated_at": now_iso()}},
        )

    def enqueue_index_job(self, scope_kind: ScopeKind, scope_id: str) -> dict[str, Any]:
        self._require_connection()
        safe_scope = assert_safe_segment("scopeId", scope_id)
        job = {
            "job_id": str(uuid.uuid4()),
            "scope_kind": scope_kind,
            "scope_id": safe_scope,
            "status": "queued",
            "stage": "queued",
            "created_at": now_iso(),
            "started_at": None,
            "finished_at": None,
            "progress": {
                "totalFiles": 0,
                "processedFiles": 0,
                "doneFiles": 0,
                "failedFiles": 0,
                "percent": 0,
                "currentFileId": None,
                "currentFileName": None,
                "currentStage": None,
            },
            "files": [],
            "error": None,
        }
        self.db.rag_index_jobs.insert_one(dict(job))
        try:
            task = asyncio.get_running_loop().create_task(self.process_job(job["job_id"]))
            self._tasks[job["job_id"]] = task
        except RuntimeError:
            asyncio.run(self.process_job(job["job_id"]))
        return self.get_job(job["job_id"]) or job

    async def process_job(self, job_id: str) -> None:
        await asyncio.sleep(0)
        job = self.get_job(job_id)
        if not job:
            return
        scope_kind = job["scope_kind"]
        scope_id = job["scope_id"]
        files = [self._clean(row) or {} for row in self.db.rag_files.find({"scope_kind": scope_kind, "scope_id": scope_id}).sort("imported_at", 1)]
        self.db.rag_index_jobs.update_one(
            {"job_id": job_id},
            {
                "$set": {
                    "status": "running",
                    "stage": "hashing",
                    "started_at": now_iso(),
                    "progress.totalFiles": len(files),
                    "files": [{"fileId": row["id"], "originalName": row["original_name"], "status": row["status"], "error": row.get("error")} for row in files],
                }
            },
        )
        self.db.rag_chunks.delete_many({"scope_kind": scope_kind, "scope_id": scope_id})
        processed = done_count = failed_count = 0
        for file_row in files:
            current_file = file_row["id"]
            current_name = file_row["original_name"]
            try:
                self.db.rag_index_jobs.update_one(
                    {"job_id": job_id},
                    {"$set": {"stage": "chunking", "progress.currentFileId": current_file, "progress.currentFileName": current_name, "progress.currentStage": "chunking"}},
                )
                self._update_file_status(scope_kind, scope_id, current_file, "chunking")
                chunks = chunk_text_with_lines(str(file_row.get("content", "")))
                self.db.rag_index_jobs.update_one(
                    {"job_id": job_id},
                    {"$set": {"stage": "embedding", "progress.currentStage": "embedding"}},
                )
                self._update_file_status(scope_kind, scope_id, current_file, "embedding")
                for index, chunk in enumerate(chunks):
                    self.db.rag_chunks.insert_one(
                        {
                            "chunk_id": f"{current_file}:{index}",
                            "file_id": current_file,
                            "scope_kind": scope_kind,
                            "scope_id": scope_id,
                            "source": "session_docs" if scope_kind == "session" else "profiles",
                            "path": file_row.get("source_display_path") or file_row.get("original_name"),
                            "line_start": chunk["lineStart"],
                            "line_end": chunk["lineEnd"],
                            "text": chunk["text"],
                            "chunk_index": index,
                            "embedding": [],
                            "created_at": now_iso(),
                        }
                    )
                self._update_file_status(scope_kind, scope_id, current_file, "done")
                done_count += 1
            except Exception as exc:
                self._update_file_status(scope_kind, scope_id, current_file, "failed", str(exc))
                failed_count += 1
            processed += 1
            percent = 100 if not files else int((processed / len(files)) * 100)
            self.db.rag_index_jobs.update_one(
                {"job_id": job_id},
                {
                    "$set": {
                        "progress.processedFiles": processed,
                        "progress.doneFiles": done_count,
                        "progress.failedFiles": failed_count,
                        "progress.percent": percent,
                    }
                },
            )
        final_status = "failed" if failed_count and not done_count else "done"
        self.db.rag_index_jobs.update_one(
            {"job_id": job_id},
            {
                "$set": {
                    "status": final_status,
                    "stage": "failed" if final_status == "failed" else "done",
                    "finished_at": now_iso(),
                    "progress.currentFileId": None,
                    "progress.currentFileName": None,
                    "progress.currentStage": None,
                }
            },
        )
        self._write_scope_artifacts(scope_kind, scope_id)

    def list_jobs(self, limit: int = 50) -> dict[str, Any]:
        self._require_connection()
        rows = [self._clean(row) or {} for row in self.db.rag_index_jobs.find({}).sort("created_at", -1).limit(max(1, limit))]
        running = next((row["job_id"] for row in rows if row.get("status") == "running"), None)
        queued = [row["job_id"] for row in rows if row.get("status") == "queued"]
        return {"jobs": rows, "queue": {"activeJobId": running, "queuedJobIds": queued, "processing": bool(running)}}

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        self._require_connection()
        return self._clean(self.db.rag_index_jobs.find_one({"job_id": job_id}))

    def retry_job(self, job_id: str) -> dict[str, Any]:
        self._require_connection()
        job = self.get_job(job_id)
        if not job:
            raise KeyError(job_id)
        return self.enqueue_index_job(job["scope_kind"], job["scope_id"])

    def get_memory_state(self, session_id: str, limit: int = 80) -> dict[str, Any]:
        self._require_connection()
        safe_session = assert_safe_segment("sessionId", session_id)
        config = self._memory_config(safe_session)
        summary = self._memory_summary(safe_session)
        cursor = self.db.rag_memory_entries.find({"session_id": safe_session, "record_type": "entry"}).sort("ts", -1).limit(max(1, limit))
        entries = []
        for row in cursor:
            clean = self._clean(row) or {}
            clean.pop("record_type", None)
            clean.pop("session_id", None)
            entries.append(clean)
        return {"config": config, "summary": summary, "entries": entries}

    def update_memory_config(self, session_id: str, patch: dict[str, Any]) -> dict[str, Any]:
        self._require_connection()
        safe_session = assert_safe_segment("sessionId", session_id)
        current = self._memory_config(safe_session)
        next_config = {
            "enabled": bool(patch["enabled"]) if "enabled" in patch else current["enabled"],
            "thresholdPct": max(0.5, min(0.99, float(patch.get("thresholdPct", current["thresholdPct"])))),
            "tokenBudget": max(1000, min(200000, int(patch.get("tokenBudget", current["tokenBudget"])))),
            "updated_at": now_iso(),
        }
        self.db.rag_memory_entries.find_one_and_update(
            {"session_id": safe_session, "record_type": "config"},
            {"$set": next_config, "$setOnInsert": {"session_id": safe_session, "record_type": "config"}},
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )
        return self.get_memory_state(safe_session)

    def clear_memory(self, session_id: str) -> dict[str, Any]:
        self._require_connection()
        safe_session = assert_safe_segment("sessionId", session_id)
        self.db.rag_memory_entries.delete_many({"session_id": safe_session, "record_type": "entry"})
        summary = {"text": "", "updatedAt": now_iso(), "estimatedTokens": 0, "entryCount": 0}
        self.db.rag_memory_entries.find_one_and_update(
            {"session_id": safe_session, "record_type": "summary"},
            {"$set": summary, "$setOnInsert": {"session_id": safe_session, "record_type": "summary"}},
            upsert=True,
        )
        return self.get_memory_state(safe_session)

    def compact_memory(self, session_id: str, reason: str = "manual") -> dict[str, Any]:
        self._require_connection()
        safe_session = assert_safe_segment("sessionId", session_id)
        state = self.get_memory_state(safe_session, limit=200)
        entries = state["entries"]
        bullets: list[str] = []
        for entry in entries[:60]:
            content = str(entry.get("content", "")).strip().replace("\n", " ")
            if not content:
                continue
            line = f"- [{entry.get('kind', 'context')}] {content[:220]}"
            if line not in bullets:
                bullets.append(line)
            if len(bullets) >= 18:
                break
        summary_text = "\n".join([f"{reason.title()} compaction triggered."] + bullets) if bullets else ""
        retained = list(reversed(entries[:20]))
        self.db.rag_memory_entries.delete_many({"session_id": safe_session, "record_type": "entry"})
        if retained:
            self.db.rag_memory_entries.insert_many(
                [
                    {"session_id": safe_session, "record_type": "entry", **entry}
                    for entry in retained
                ]
            )
        summary = {
            "text": summary_text,
            "updatedAt": now_iso(),
            "lastCompactedAt": now_iso(),
            "estimatedTokens": estimate_tokens(summary_text + "\n" + "\n".join(entry["content"] for entry in retained if entry.get("content"))),
            "entryCount": len(retained),
        }
        self.db.rag_memory_entries.find_one_and_update(
            {"session_id": safe_session, "record_type": "summary"},
            {"$set": summary, "$setOnInsert": {"session_id": safe_session, "record_type": "summary"}},
            upsert=True,
        )
        return self.get_memory_state(safe_session)

    def append_memory_turn(self, session_id: str, prompt: str, answer: str) -> dict[str, Any]:
        self._require_connection()
        safe_session = assert_safe_segment("sessionId", session_id)
        config = self._memory_config(safe_session)
        if not config["enabled"]:
            return self.get_memory_state(safe_session)
        text = str(answer or "").strip()
        if not text:
            return self.get_memory_state(safe_session)
        combined = f"{prompt}\n{text}".lower()
        kind = "context"
        if any(token in combined for token in ["decision", "validated", "chosen", "will use"]):
            kind = "decision"
        elif any(token in combined for token in ["must", "constraint", "required", "policy"]):
            kind = "constraint"
        elif any(token in combined for token in ["todo", "next step", "remaining"]):
            kind = "todo"
        elif any(token in combined for token in ["phase", "profile", "version", "is ", "are "]):
            kind = "fact"
        entry = {
            "id": str(uuid.uuid4()),
            "ts": now_iso(),
            "kind": kind,
            "content": text[:1200] + ("…" if len(text) > 1200 else ""),
            "confidence": 0.55,
            "source": prompt[:300],
        }
        self.db.rag_memory_entries.insert_one({"session_id": safe_session, "record_type": "entry", **entry})
        entries_state = self.get_memory_state(safe_session, limit=300)
        total_tokens = estimate_tokens(entries_state["summary"]["text"] + "\n" + "\n".join(row["content"] for row in entries_state["entries"]))
        self.db.rag_memory_entries.find_one_and_update(
            {"session_id": safe_session, "record_type": "summary"},
            {
                "$set": {
                    "text": entries_state["summary"]["text"],
                    "updatedAt": now_iso(),
                    "estimatedTokens": total_tokens,
                    "entryCount": len(entries_state["entries"]),
                },
                "$setOnInsert": {"session_id": safe_session, "record_type": "summary"},
            },
            upsert=True,
        )
        if total_tokens >= int(config["tokenBudget"] * config["thresholdPct"]):
            return self.compact_memory(safe_session, reason="auto")
        return self.get_memory_state(safe_session)

    def lookup(self, question: str, session_id: str | None = None, scope: dict[str, Any] | None = None) -> dict[str, Any]:
        self._require_connection()
        query = str(question or "").strip()
        scope = scope or {}
        source_priority = normalize_source_priority(scope.get("source_priority"))
        explicit_profiles = scope.get("profiles")
        selected_profiles = explicit_profiles or self.list_profiles()
        hits: list[dict[str, Any]] = []
        indexing = False
        disabled_memory = False
        query_tokens = tokenize(query)

        if "session_memory" in source_priority and scope.get("session_memory", True) and session_id:
            memory_state = self.get_memory_state(session_id, limit=80)
            if not memory_state["config"]["enabled"]:
                disabled_memory = True
            else:
                memory_docs: list[tuple[str, str]] = []
                if memory_state["summary"]["text"].strip():
                    memory_docs.append(("session_memory:summary", memory_state["summary"]["text"]))
                memory_docs.extend((f"session_memory:entry:{entry['id']}", entry["content"]) for entry in memory_state["entries"])
                for path, text in memory_docs:
                    score = score_chunk(query, query_tokens, text)
                    if score > 0:
                        hits.append(
                            {
                                "source": "session_memory",
                                "chunkId": path,
                                "score": score,
                                "snippet": text[:800],
                                "citation": {"path": path},
                            }
                        )

        selected_scopes: list[tuple[str, ScopeKind, str, str | None]] = []
        for source in source_priority:
            if source == "session_docs" and scope.get("session_docs", True) and session_id:
                selected_scopes.append((source, "session", assert_safe_segment("sessionId", session_id), None))
            elif source == "profiles":
                for profile in selected_profiles:
                    selected_scopes.append((source, "profile", assert_safe_segment("profileName", profile), profile))

        for source, scope_kind, scope_id, profile in selected_scopes:
            files = [self._clean(row) or {} for row in self.db.rag_files.find({"scope_kind": scope_kind, "scope_id": scope_id})]
            if files and any(row.get("status") != "done" for row in files):
                indexing = True
            rows = self.db.rag_chunks.find({"scope_kind": scope_kind, "scope_id": scope_id})
            for row in rows:
                clean = self._clean(row) or {}
                text = str(clean.get("text", ""))
                score = score_chunk(query, query_tokens, text)
                if score <= 0:
                    continue
                hits.append(
                    {
                        "source": source,
                        "profile": profile,
                        "chunkId": clean.get("chunk_id"),
                        "score": score,
                        "snippet": text[:800],
                        "citation": {
                            "path": clean.get("path", "unknown"),
                            "lineStart": clean.get("line_start"),
                            "lineEnd": clean.get("line_end"),
                        },
                    }
                )

        order = {source: index for index, source in enumerate(source_priority)}
        hits.sort(key=lambda item: (order.get(item["source"], 999), -item["score"]))
        trimmed = hits[:8]
        if trimmed:
            status = "ok"
        elif indexing:
            status = "indexing"
        elif disabled_memory and not selected_scopes:
            status = "disabled"
        else:
            status = "no_hits"
        response = {
            "status": status,
            "query": query,
            "scope": {
                "sourcePriority": source_priority,
                "sessionDocs": scope.get("session_docs"),
                "sessionMemory": scope.get("session_memory"),
                "profiles": explicit_profiles,
            },
            "hits": trimmed,
            "meta": {"sourcePriority": source_priority, "totalHits": len(trimmed), **({"indexing": {"active": True}} if indexing else {})},
        }
        if status != "ok":
            response["message"] = build_transparency_message(response)
        return response

    def lookup_for_model(self, question: str, session_id: str | None = None, scope: dict[str, Any] | None = None) -> str:
        return render_lookup_for_model(self.lookup(question=question, session_id=session_id, scope=scope))
