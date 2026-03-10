from __future__ import annotations

import json
from datetime import datetime, timezone


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def run_state(run_id: str, state: str) -> dict:
    return {"type": "run_state", "runId": run_id, "state": state, "timestamp": now_iso()}


def run_phase(run_id: str, phase: str, detail: str | None = None) -> dict:
    event = {"type": "run_phase_changed", "runId": run_id, "phase": phase, "timestamp": now_iso()}
    if detail:
        event["detail"] = detail
    return event


def token(value: str) -> dict:
    return {"type": "token", "token": value}


def done() -> dict:
    return {"type": "done"}
