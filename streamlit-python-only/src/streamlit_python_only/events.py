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


def run_diagnostic(run_id: str, code: str, message: str, level: str = "info", data: dict | None = None) -> dict:
    event = {
        "type": "run_diagnostic",
        "runId": run_id,
        "code": code,
        "level": level,
        "message": message,
        "timestamp": now_iso(),
    }
    if data:
        event["data"] = data
    return event


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


def terminal_opened(
    run_id: str | None,
    terminal_id: str,
    command: str | None,
    cwd: str,
    *,
    session_id: str | None = None,
    shell: str | None = None,
    backend: str | None = None,
) -> dict:
    event = {
        "type": "terminal_opened",
        "terminalId": terminal_id,
        "command": command or "",
        "cwd": cwd,
        "timestamp": now_iso(),
    }
    if run_id:
        event["runId"] = run_id
    if session_id:
        event["sessionId"] = session_id
    if shell:
        event["shell"] = shell
    if backend:
        event["backend"] = backend
    return event


def terminal_data(
    run_id: str | None,
    terminal_id: str,
    chunk: str,
    stream: str = "pty",
    *,
    session_id: str | None = None,
) -> dict:
    event = {
        "type": "terminal_data",
        "terminalId": terminal_id,
        "chunk": chunk,
        "stream": stream,
        "timestamp": now_iso(),
    }
    if run_id:
        event["runId"] = run_id
    if session_id:
        event["sessionId"] = session_id
    return event


def terminal_input(
    run_id: str | None,
    terminal_id: str,
    *,
    source: str,
    data: str,
    input_kind: str = "text",
    session_id: str | None = None,
) -> dict:
    preview = str(data or "")[-4000:]
    event = {
        "type": "terminal_input",
        "terminalId": terminal_id,
        "source": source,
        "inputKind": input_kind,
        "data": preview,
        "byteLength": len((data or "").encode("utf-8")),
        "timestamp": now_iso(),
    }
    if run_id:
        event["runId"] = run_id
    if session_id:
        event["sessionId"] = session_id
    return event


def terminal_resized(
    run_id: str | None,
    terminal_id: str,
    cols: int,
    rows: int,
    *,
    session_id: str | None = None,
) -> dict:
    event = {
        "type": "terminal_resized",
        "terminalId": terminal_id,
        "cols": cols,
        "rows": rows,
        "timestamp": now_iso(),
    }
    if run_id:
        event["runId"] = run_id
    if session_id:
        event["sessionId"] = session_id
    return event


def terminal_control_changed(
    run_id: str | None,
    terminal_id: str,
    owner: str,
    reason: str | None = None,
    *,
    session_id: str | None = None,
) -> dict:
    event = {
        "type": "terminal_control_changed",
        "terminalId": terminal_id,
        "owner": owner,
        "timestamp": now_iso(),
    }
    if run_id:
        event["runId"] = run_id
    if session_id:
        event["sessionId"] = session_id
    if reason:
        event["reason"] = reason
    return event


def terminal_closed(
    run_id: str | None,
    terminal_id: str,
    *,
    reason: str = "closed",
    by: str | None = None,
    session_id: str | None = None,
) -> dict:
    event = {
        "type": "terminal_closed",
        "terminalId": terminal_id,
        "reason": reason,
        "timestamp": now_iso(),
    }
    if run_id:
        event["runId"] = run_id
    if session_id:
        event["sessionId"] = session_id
    if by:
        event["by"] = by
    return event


def terminal_exit(
    run_id: str | None,
    terminal_id: str,
    exit_code: int | None,
    output: str,
    *,
    signal_value: int | None = None,
    session_id: str | None = None,
) -> dict:
    event = {
        "type": "terminal_exit",
        "terminalId": terminal_id,
        "exitCode": exit_code,
        "output": output,
        "timestamp": now_iso(),
    }
    if run_id:
        event["runId"] = run_id
    if session_id:
        event["sessionId"] = session_id
    if signal_value is not None:
        event["signal"] = signal_value
    return event


def terminal_error(
    run_id: str | None,
    terminal_id: str,
    message: str,
    *,
    session_id: str | None = None,
) -> dict:
    event = {
        "type": "terminal_error",
        "terminalId": terminal_id,
        "error": message,
        "message": message,
        "timestamp": now_iso(),
    }
    if run_id:
        event["runId"] = run_id
    if session_id:
        event["sessionId"] = session_id
    return event


def done() -> dict:
    return {"type": "done"}
