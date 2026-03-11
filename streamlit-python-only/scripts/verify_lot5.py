from __future__ import annotations

import json
from pathlib import Path

import httpx
import mongomock

from continue_better_py.matrix import MatrixService
from continue_better_py.store import MongoStore


def encode_sse(events: list[dict]) -> str:
    return "".join(f"data: {json.dumps(event, ensure_ascii=False)}\n\n" for event in events)


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    artifact_root = project_root.joinpath("artifacts", "verify_lot5")
    artifact_root.mkdir(parents=True, exist_ok=True)

    store = MongoStore(
        client=mongomock.MongoClient(),
        database_name="continue_better_python_verify_lot5",
        artifacts_root=artifact_root,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/sessions" and request.method == "POST":
            return httpx.Response(200, json={"session_id": "verify-session"})
        if request.url.path == "/v1/chat/stream":
            payload = json.loads(request.content.decode("utf-8"))
            prompt = payload["message"]
            if "Read matrix_fixtures/roadmap_status.md" in prompt:
                return httpx.Response(
                    200,
                    text=encode_sse(
                        [
                            {"type": "run_state", "runId": "verify-run", "state": "planning", "timestamp": "2026-03-11T13:00:00+00:00"},
                            {"type": "tool_call", "runId": "verify-run", "actionId": "call-1", "name": "read_file", "arguments": "{\"path\":\"matrix_fixtures/roadmap_status.md\"}", "timestamp": "2026-03-11T13:00:01+00:00"},
                            {"type": "token", "token": "AURORA_PHASE4"},
                            {"type": "run_state", "runId": "verify-run", "state": "completed", "timestamp": "2026-03-11T13:00:02+00:00"},
                            {"type": "done"},
                        ]
                    ),
                    headers={"content-type": "text/event-stream"},
                )
            return httpx.Response(
                200,
                text=encode_sse(
                    [
                        {"type": "run_state", "runId": "verify-social", "state": "planning", "timestamp": "2026-03-11T12:59:00+00:00"},
                        {"type": "token", "token": "Salut"},
                        {"type": "run_state", "runId": "verify-social", "state": "completed", "timestamp": "2026-03-11T12:59:01+00:00"},
                        {"type": "done"},
                    ]
                ),
                headers={"content-type": "text/event-stream"},
            )
        raise RuntimeError(f"Unexpected mock path: {request.method} {request.url.path}")

    transport = httpx.MockTransport(handler)
    service = MatrixService(
        store=store,
        backend_base_url="http://backend.test",
        sidecar_base_url="http://sidecar.test",
        backend_transport=transport,
        sidecar_transport=transport,
    )

    report = service.run_matrix(
        {
            "models": ["verify-model"],
            "scenario_ids": ["social_no_clarify", "read_file_phase_status"],
            "profiles": ["baseline_current"],
            "surfaces": ["backend_relay"],
            "repeat": 1,
        }
    )
    comparison = service.compare(report["report_id"], report["report_id"])
    persisted = service.get_report(report["report_id"])
    checks = {
        "report_saved": persisted is not None,
        "results_count": len(report["results"]) == 2,
        "aggregate_runs": report["aggregate"]["bySurface"]["backend_relay"]["runs"] == 2,
        "comparison_runs": comparison["summary"]["currentRuns"] == 2,
        "artifact_written": artifact_root.joinpath("matrix", f"{report['report_id']}.json").exists(),
    }
    print(json.dumps({"checks": checks, "report_id": report["report_id"]}, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
