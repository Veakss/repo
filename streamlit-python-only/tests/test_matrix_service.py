from __future__ import annotations

import json
from pathlib import Path

import httpx
import mongomock

from continue_better_py.matrix import MatrixService
from continue_better_py.store import MongoStore


def encode_sse(events: list[dict]) -> str:
    return "".join(f"data: {json.dumps(event, ensure_ascii=False)}\n\n" for event in events)


def build_service(tmp_path: Path) -> MatrixService:
    store = MongoStore(
        client=mongomock.MongoClient(),
        database_name="continue_better_python_matrix_test",
        artifacts_root=tmp_path,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/sessions" and request.method == "POST":
            return httpx.Response(200, json={"session_id": "matrix-session-1"})
        if request.url.path == "/v1/chat/stream":
            payload = json.loads(request.content.decode("utf-8"))
            prompt = payload["message"]
            if "food application" in prompt:
                return httpx.Response(
                    200,
                    text=encode_sse(
                        [
                            {"type": "run_state", "runId": "run-clar", "state": "awaiting_clarification", "timestamp": "2026-03-11T12:00:00+00:00"},
                            {
                                "type": "clarification_required",
                                "runId": "run-clar",
                                "clarificationId": "clar-1",
                                "question": "Which platform?",
                                "questions": ["Which platform?"],
                                "options": [{"label": "iPhone"}],
                                "timestamp": "2026-03-11T12:00:01+00:00",
                            },
                            {"type": "done"},
                        ]
                    ),
                    headers={"content-type": "text/event-stream"},
                )
            if "Create the file artifacts/matrix_sandbox/hello.txt" in prompt:
                return httpx.Response(
                    200,
                    text=encode_sse(
                        [
                            {"type": "run_state", "runId": "run-file", "state": "awaiting_approval", "timestamp": "2026-03-11T12:01:00+00:00"},
                            {
                                "type": "tool_call",
                                "runId": "run-file",
                                "actionId": "call-1",
                                "name": "write_file",
                                "arguments": "{\"path\":\"artifacts/matrix_sandbox/hello.txt\",\"content\":\"bonjour-matrice\"}",
                                "timestamp": "2026-03-11T12:01:01+00:00",
                            },
                            {
                                "type": "approval_required",
                                "runId": "run-file",
                                "actionId": "approval-1",
                                "name": "write_file",
                                "riskLevel": "risky",
                                "arguments": "{\"path\":\"artifacts/matrix_sandbox/hello.txt\"}",
                                "timestamp": "2026-03-11T12:01:02+00:00",
                            },
                            {"type": "done"},
                        ]
                    ),
                    headers={"content-type": "text/event-stream"},
                )
            if "Read matrix_fixtures/roadmap_status.md" in prompt:
                return httpx.Response(
                    200,
                    text=encode_sse(
                        [
                            {"type": "run_state", "runId": "run-read", "state": "planning", "timestamp": "2026-03-11T12:02:00+00:00"},
                            {
                                "type": "tool_call",
                                "runId": "run-read",
                                "actionId": "call-2",
                                "name": "read_file",
                                "arguments": "{\"path\":\"matrix_fixtures/roadmap_status.md\"}",
                                "timestamp": "2026-03-11T12:02:01+00:00",
                            },
                            {"type": "token", "token": "AURORA_PHASE4"},
                            {"type": "run_state", "runId": "run-read", "state": "completed", "timestamp": "2026-03-11T12:02:02+00:00"},
                            {"type": "done"},
                        ]
                    ),
                    headers={"content-type": "text/event-stream"},
                )
            return httpx.Response(
                200,
                text=encode_sse(
                    [
                        {"type": "run_state", "runId": "run-social", "state": "planning", "timestamp": "2026-03-11T11:59:00+00:00"},
                        {"type": "token", "token": "Salut"},
                        {"type": "run_state", "runId": "run-social", "state": "completed", "timestamp": "2026-03-11T11:59:01+00:00"},
                        {"type": "done"},
                    ]
                ),
                headers={"content-type": "text/event-stream"},
            )
        if request.url.path == "/v1/approvals/stream":
            matrix_file = Path(__file__).resolve().parents[1].joinpath("artifacts", "matrix_sandbox", "hello.txt")
            matrix_file.parent.mkdir(parents=True, exist_ok=True)
            matrix_file.write_text("bonjour-matrice", encoding="utf-8")
            return httpx.Response(
                200,
                text=encode_sse(
                    [
                        {"type": "approval_decision", "runId": "run-file", "actionId": "approval-1", "decision": "approved", "timestamp": "2026-03-11T12:01:03+00:00"},
                        {"type": "tool_result", "runId": "run-file", "name": "write_file", "ok": True, "timestamp": "2026-03-11T12:01:04+00:00"},
                        {
                            "type": "tool_call",
                            "runId": "run-file",
                            "actionId": "call-3",
                            "name": "read_file",
                            "arguments": "{\"path\":\"artifacts/matrix_sandbox/hello.txt\"}",
                            "timestamp": "2026-03-11T12:01:05+00:00",
                        },
                        {"type": "token", "token": "bonjour-matrice"},
                        {"type": "run_state", "runId": "run-file", "state": "completed", "timestamp": "2026-03-11T12:01:06+00:00"},
                        {"type": "done"},
                    ]
                ),
                headers={"content-type": "text/event-stream"},
            )
        if request.url.path == "/v1/clarifications/stream":
            return httpx.Response(
                200,
                text=encode_sse(
                    [
                        {"type": "clarification_answered", "runId": "run-clar", "clarificationId": "clar-1", "answer": "For iPhone", "timestamp": "2026-03-11T12:00:02+00:00"},
                        {"type": "token", "token": "iPhone MVP with restaurants and favorites"},
                        {"type": "run_state", "runId": "run-clar", "state": "completed", "timestamp": "2026-03-11T12:00:03+00:00"},
                        {"type": "done"},
                    ]
                ),
                headers={"content-type": "text/event-stream"},
            )
        raise AssertionError(f"Unexpected path {request.method} {request.url.path}")

    transport = httpx.MockTransport(handler)
    return MatrixService(
        store=store,
        backend_base_url="http://backend.test",
        sidecar_base_url="http://sidecar.test",
        backend_transport=transport,
        sidecar_transport=transport,
    )


def test_matrix_service_runs_and_persists_report(tmp_path: Path):
    service = build_service(tmp_path)

    report = service.run_matrix(
        {
            "models": ["test-model"],
            "scenario_ids": [
                "social_no_clarify",
                "clarification_resume_food_app_multiturn",
                "files_write_then_readback",
                "read_file_phase_status",
            ],
            "profiles": ["baseline_current"],
            "surfaces": ["backend_relay"],
            "repeat": 1,
        }
    )

    assert report["report_id"].startswith("matrix-")
    assert len(report["results"]) == 4
    assert report["aggregate"]["bySurface"]["backend_relay"]["runs"] == 4
    assert any(row["scenarioId"] == "files_write_then_readback" and row["grade"]["overall"] == "pass" for row in report["results"])
    assert service.get_report(report["report_id"]) is not None
    assert tmp_path.joinpath("matrix", f"{report['report_id']}.json").exists()


def test_matrix_service_compare_returns_summary(tmp_path: Path):
    service = build_service(tmp_path)
    report_a = service.run_matrix(
        {
            "models": ["test-model"],
            "scenario_ids": ["social_no_clarify"],
            "profiles": ["baseline_current"],
            "surfaces": ["backend_relay"],
        }
    )
    report_b = service.run_matrix(
        {
            "models": ["test-model"],
            "scenario_ids": ["read_file_phase_status"],
            "profiles": ["baseline_current"],
            "surfaces": ["backend_relay"],
        }
    )

    comparison = service.compare(report_b["report_id"], report_a["report_id"])
    assert comparison["summary"]["currentRuns"] == 1
    assert comparison["summary"]["baselineRuns"] == 1
    assert isinstance(comparison["byModel"], list)
