from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi.testclient import TestClient

from continue_better_py.backend_app import create_backend_app
from continue_better_py.sidecar_app import create_sidecar_app
from continue_better_py.store import MongoStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_sse_payloads(text: str) -> list[dict]:
    payloads: list[dict] = []
    for packet in text.split("\n\n"):
        if packet.startswith("data: "):
            payloads.append(json.loads(packet[6:]))
    return payloads


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    database_name = f"continue_better_python_live_{int(time.time())}"
    artifacts_root = PROJECT_ROOT.joinpath("artifacts", "live_mongo_checks")
    artifacts_root.mkdir(parents=True, exist_ok=True)

    store = MongoStore(database_name=database_name, artifacts_root=artifacts_root)
    sidecar_transport = httpx.ASGITransport(app=create_sidecar_app())
    app = create_backend_app(store=store, sidecar_transport=sidecar_transport, sidecar_base_url="http://sidecar.test")

    summary: dict[str, object] = {"database": database_name, "checks": []}

    try:
        with TestClient(app) as client:
            summary["checks"].append(("health", client.get("/v1/health").status_code == 200))
            summary["checks"].append(("models", client.get("/v1/models").status_code == 200))
            summary["checks"].append(("capabilities", client.get("/v1/capabilities").status_code == 200))

            created = client.post("/v1/sessions", json={"title": "Live Mongo Session"})
            summary["checks"].append(("create_session", created.status_code == 200))
            session_id = created.json()["session_id"]

            response = client.post(
                "/v1/chat/stream",
                json={
                    "session_id": session_id,
                    "message": "Reply with exactly: mongo live ok",
                    "run_id": "lot2-live-mongo",
                    "allow_writes": False,
                },
            )
            summary["checks"].append(("chat_stream_http", response.status_code == 200))
            events = parse_sse_payloads(response.text)
            summary["checks"].append(("chat_has_done", any(event.get("type") == "done" for event in events)))
            summary["checks"].append(("chat_has_token", any(event.get("type") == "token" for event in events)))
            summary["checks"].append(
                (
                    "chat_completed",
                    any(event.get("type") == "run_state" and event.get("state") == "completed" for event in events),
                )
            )

            runs = client.get("/v1/runs", params={"session_id": session_id})
            summary["checks"].append(("runs_list", runs.status_code == 200 and len(runs.json()["runs"]) == 1))

            run = client.get("/v1/runs/lot2-live-mongo/events")
            summary["checks"].append(("run_detail", run.status_code == 200 and len(run.json()["run"]["events"]) >= 3))

            messages = client.get(f"/v1/sessions/{session_id}/messages")
            body = messages.json()["messages"]
            summary["checks"].append(("message_persisted", messages.status_code == 200 and len(body) >= 2))
            summary["checks"].append(
                (
                    "assistant_reply_persisted",
                    any(message["role"] == "assistant" and "mongo" in message["content"].lower() for message in body),
                )
            )

            meta_path = artifacts_root.joinpath("runs", "lot2-live-mongo.meta.json")
            jsonl_path = artifacts_root.joinpath("runs", "lot2-live-mongo.jsonl")
            summary["checks"].append(("meta_json_written", meta_path.exists()))
            summary["checks"].append(("events_jsonl_written", jsonl_path.exists()))
    finally:
        store.client.drop_database(database_name)

    print(json.dumps(summary, indent=2))
    return 0 if all(ok for _, ok in summary["checks"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
