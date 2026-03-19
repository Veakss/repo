from __future__ import annotations

import mongomock

from streamlit_python_only.rag import RagService


def build_service(tmp_path):
    return RagService(
        client=mongomock.MongoClient(),
        database_name="streamlit_python_only_rag_test",
        artifacts_root=tmp_path,
    )


def test_rag_profile_file_index_and_lookup_flow(tmp_path):
    service = build_service(tmp_path)
    source = tmp_path.joinpath("plan.md")
    source.write_text("Phase 4 focuses on Session Docs and Profiles.\nRAG should provide citations.", encoding="utf-8")

    profiles = service.create_profile("default")
    assert profiles == ["default"]

    imported = service.import_file("profile", "default", str(source), "docs/plan.md")
    assert imported["imported"] is True
    assert imported["entry"]["status"] == "pending"

    job = service.enqueue_index_job("profile", "default")
    assert job["status"] in {"done", "queued", "running"}
    final_job = service.get_job(job["job_id"])
    assert final_job is not None
    assert final_job["status"] == "done"

    files = service.list_files("profile", "default")
    assert files["files"][0]["status"] == "done"

    lookup = service.lookup("What does phase 4 focus on?", scope={"profiles": ["default"]})
    assert lookup["status"] == "ok"
    assert lookup["hits"][0]["citation"]["path"] == "docs/plan.md"
    assert lookup["meta"]["strongHitCount"] >= 1
    assert "phase" in lookup["meta"]["queryTokens"]
    assert "docs/plan.md" in lookup["meta"]["topDocumentPaths"]


def test_rag_session_memory_lifecycle(tmp_path):
    service = build_service(tmp_path)
    state = service.get_memory_state("session-1", limit=20)
    assert state["config"]["enabled"] is True
    assert state["summary"]["entryCount"] == 0

    service.append_memory_turn("session-1", "What is the plan?", "The current plan is to build phase 4 next.")
    service.append_memory_turn("session-1", "Any constraint?", "A constraint is to keep Mongo canonical.")
    updated = service.get_memory_state("session-1", limit=20)
    assert len(updated["entries"]) == 2

    patched = service.update_memory_config("session-1", {"enabled": False, "thresholdPct": 0.75, "tokenBudget": 4000})
    assert patched["config"]["enabled"] is False

    compacted = service.compact_memory("session-1")
    assert "compaction" in compacted["summary"]["text"].lower()

    cleared = service.clear_memory("session-1")
    assert cleared["entries"] == []
    assert cleared["summary"]["entryCount"] == 0


def test_rag_session_lookup_reuses_followup_context(tmp_path):
    service = build_service(tmp_path)
    source = tmp_path.joinpath("roadmap_status.md")
    source.write_text(
        "# Matrix Fixture\nCurrent checkpoint: AURORA_PHASE4\nPhase 4 status: very advanced and close to closure.\n",
        encoding="utf-8",
    )

    service.import_file("session", "s1", str(source), "matrix/roadmap_status.md")
    service.enqueue_index_job("session", "s1")

    first = service.lookup("What checkpoint is recorded in the indexed document?", session_id="s1")
    assert first["status"] == "ok"
    assert first["meta"]["followupContextUsed"] is False

    second = service.lookup("And what is the Phase 4 status?", session_id="s1")
    assert second["status"] == "ok"
    assert second["meta"]["followupContextUsed"] is True
    assert second["meta"]["strongHitCount"] >= 1
    assert "matrix/roadmap_status.md" in second["meta"]["topDocumentPaths"]
    assert any("status" in hit["matchedTerms"] for hit in second["hits"])


def test_rag_lookup_dedupes_nearby_chunks_and_reports_indexing(tmp_path):
    service = build_service(tmp_path)
    source = tmp_path.joinpath("repeated.md")
    source.write_text(
        "\n".join(
            [
                "Phase 4 status: very advanced and close to closure.",
                "Phase 4 status: very advanced and close to closure.",
                "Phase 4 status: very advanced and close to closure.",
                "Phase 4 status: very advanced and close to closure.",
            ]
        ),
        encoding="utf-8",
    )

    service.import_file("session", "s1", str(source), "docs/repeated.md")
    indexing = service.lookup("What is the Phase 4 status?", session_id="s1")
    assert indexing["status"] == "indexing"

    service.enqueue_index_job("session", "s1")
    indexed = service.lookup("What is the Phase 4 status?", session_id="s1")
    assert indexed["status"] == "ok"
    assert len(indexed["hits"]) <= 2
