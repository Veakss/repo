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


def run_diagnostic(run_id: str, code: str, message: str, level: str = "info") -> dict:
    return {
        "type": "run_diagnostic",
        "runId": run_id,
        "code": code,
        "level": level,
        "message": message,
        "timestamp": now_iso(),
    }


def token(value: str) -> dict:
    return {"type": "token", "token": value}


def approval_required(run_id: str, approval_id: str, name: str, arguments: str, risk_level: str) -> dict:
    return {
        "type": "approval_required",
        "runId": run_id,
        "actionId": approval_id,
        "name": name,
        "riskLevel": risk_level,
        "arguments": arguments,
        "timestamp": now_iso(),
    }


def approval_decision(run_id: str, approval_id: str, decision: str, reason: str | None = None) -> dict:
    event = {
        "type": "approval_decision",
        "runId": run_id,
        "actionId": approval_id,
        "decision": decision,
        "timestamp": now_iso(),
    }
    if reason:
        event["reason"] = reason
    return event


def clarification_required(
    run_id: str,
    clarification_id: str,
    question: str,
    options: list[dict[str, str]] | None = None,
) -> dict:
    return {
        "type": "clarification_required",
        "runId": run_id,
        "clarificationId": clarification_id,
        "question": question,
        "questions": [question],
        "options": options or [],
        "allowFreeText": True,
        "timestamp": now_iso(),
    }


def error_event(message: str) -> dict:
    return {"type": "error", "error": message, "timestamp": now_iso()}


def terminal_opened(run_id: str, terminal_id: str, command: str, cwd: str) -> dict:
    return {
        "type": "terminal_opened",
        "runId": run_id,
        "terminalId": terminal_id,
        "command": command,
        "cwd": cwd,
        "timestamp": now_iso(),
    }


def terminal_data(run_id: str | None, terminal_id: str, chunk: str, stream: str = "pty") -> dict:
    event = {
        "type": "terminal_data",
        "terminalId": terminal_id,
        "chunk": chunk,
        "stream": stream,
        "timestamp": now_iso(),
    }
    if run_id:
        event["runId"] = run_id
    return event


def terminal_control_changed(run_id: str | None, terminal_id: str, owner: str, reason: str | None = None) -> dict:
    event = {
        "type": "terminal_control_changed",
        "terminalId": terminal_id,
        "owner": owner,
        "timestamp": now_iso(),
    }
    if run_id:
        event["runId"] = run_id
    if reason:
        event["reason"] = reason
    return event


def terminal_exit(run_id: str, terminal_id: str, exit_code: int, output: str) -> dict:
    return {
        "type": "terminal_exit",
        "runId": run_id,
        "terminalId": terminal_id,
        "exitCode": exit_code,
        "output": output,
        "timestamp": now_iso(),
    }


def terminal_error(run_id: str, terminal_id: str, message: str) -> dict:
    return {
        "type": "terminal_error",
        "runId": run_id,
        "terminalId": terminal_id,
        "message": message,
        "timestamp": now_iso(),
    }


def done() -> dict:
    return {"type": "done"}
