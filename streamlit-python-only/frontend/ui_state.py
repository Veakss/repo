from __future__ import annotations

import html
import json
from typing import Any


def merge_timeline(existing: list[dict[str, Any]], new_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged = list(existing)
    known = {json.dumps(event, sort_keys=True, ensure_ascii=False) for event in existing}
    for event in new_events:
        key = json.dumps(event, sort_keys=True, ensure_ascii=False)
        if key not in known:
            merged.append(event)
            known.add(key)
    return merged


def derive_pending_items(timeline: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    approvals: dict[str, dict[str, Any]] = {}
    clarification: dict[str, Any] | None = None
    for event in timeline:
        if event.get("type") == "approval_required":
            approvals[str(event["actionId"])] = {
                "approval_id": event["actionId"],
                "run_id": event.get("runId"),
                "name": event.get("name"),
                "risk_level": event.get("riskLevel", "moderate"),
                "arguments": event.get("arguments", ""),
            }
        elif event.get("type") == "approval_decision":
            approvals.pop(str(event.get("actionId")), None)
        elif event.get("type") == "clarification_required":
            clarification = {
                "clarification_id": event["clarificationId"],
                "run_id": event.get("runId"),
                "question": event.get("question") or "",
                "questions": event.get("questions") or [],
                "options": event.get("options") or [],
            }
        elif event.get("type") == "clarification_answered":
            if clarification and clarification.get("clarification_id") == event.get("clarificationId"):
                clarification = None
        elif event.get("type") == "run_state" and event.get("state") not in {"awaiting_approval", "awaiting_clarification"}:
            if clarification and clarification.get("run_id") == event.get("runId"):
                clarification = None
    return list(approvals.values()), clarification


def flatten_tree(tree: dict[str, Any]) -> list[str]:
    output: list[str] = []

    def walk(nodes: list[dict[str, Any]]) -> None:
        for node in nodes:
            if node.get("type") == "file":
                output.append(str(node["path"]))
            else:
                walk(node.get("children", []))

    walk(tree.get("children", []))
    return output


def format_event_label(event: dict[str, Any]) -> str:
    event_type = event.get("type", "event")
    if event_type == "run_state":
        return f"State -> {event.get('state', 'unknown')}"
    if event_type == "run_phase_changed":
        detail = event.get("detail")
        return f"Phase -> {event.get('phase', 'unknown')}" + (f" ({detail})" if detail else "")
    if event_type == "approval_required":
        return f"Approval required for {event.get('name', 'tool')}"
    if event_type == "approval_decision":
        return f"Approval {event.get('decision', 'unknown')}"
    if event_type == "clarification_required":
        return f"Clarification needed: {event.get('question', '')}"
    if event_type == "tool_call":
        return f"Tool call: {event.get('name', 'unknown')}"
    if event_type == "tool_result":
        return f"Tool result: {event.get('name', 'unknown')}"
    if event_type == "error":
        return f"Error: {event.get('error', '')}"
    if event_type == "token":
        return "Streaming response"
    return event_type.replace("_", " ").title()


def build_timeline_html(timeline: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for event in reversed(timeline[-40:]):
        tone = "default"
        if event.get("type") == "error":
            tone = "error"
        elif event.get("type") in {"approval_required", "clarification_required"}:
            tone = "warn"
        elif event.get("type") == "run_state" and event.get("state") == "completed":
            tone = "success"
        label = html.escape(format_event_label(event))
        timestamp = html.escape(str(event.get("timestamp") or ""))
        payload = html.escape(json.dumps(event, ensure_ascii=False, indent=2))
        blocks.append(
            f"""
            <div class="cb-timeline-card cb-tone-{tone}">
              <div class="cb-timeline-head">
                <span>{label}</span>
                <span>{timestamp}</span>
              </div>
              <pre>{payload}</pre>
            </div>
            """
        )
    if not blocks:
        blocks.append('<div class="cb-empty-card">No run events yet.</div>')
    return f'<div class="cb-timeline-wrap">{"".join(blocks)}</div>'


def build_terminal_html(timeline: list[dict[str, Any]]) -> str:
    by_terminal: dict[str, dict[str, Any]] = {}
    for event in timeline:
        terminal_id = str(event.get("terminalId") or "")
        if not terminal_id:
            continue
        entry = by_terminal.setdefault(terminal_id, {"terminalId": terminal_id})
        if event.get("type") == "terminal_opened":
            entry["command"] = event.get("command")
            entry["cwd"] = event.get("cwd")
            entry["openedAt"] = event.get("timestamp")
        elif event.get("type") == "terminal_exit":
            entry["exitCode"] = event.get("exitCode")
            entry["output"] = event.get("output")
            entry["closedAt"] = event.get("timestamp")
        elif event.get("type") == "terminal_error":
            entry["error"] = event.get("message")
            entry["closedAt"] = event.get("timestamp")
    blocks: list[str] = []
    for terminal in reversed(list(by_terminal.values())[-20:]):
        command = html.escape(str(terminal.get("command") or "terminal"))
        cwd = html.escape(str(terminal.get("cwd") or ""))
        opened = html.escape(str(terminal.get("openedAt") or ""))
        output = html.escape(str(terminal.get("output") or terminal.get("error") or "(no output)"))
        meta = []
        if cwd:
            meta.append(cwd)
        if terminal.get("exitCode") is not None:
            meta.append(f"exit={terminal['exitCode']}")
        elif terminal.get("error"):
            meta.append("error")
        meta_text = " | ".join(meta)
        blocks.append(
            f"""
            <div class="cb-timeline-card">
              <div class="cb-timeline-head">
                <span>{command}</span>
                <span>{opened}</span>
              </div>
              <div class="cb-terminal-meta">{html.escape(meta_text)}</div>
              <pre>{output}</pre>
            </div>
            """
        )
    if not blocks:
        blocks.append('<div class="cb-empty-card">No terminal activity yet.</div>')
    return f'<div class="cb-timeline-wrap">{"".join(blocks)}</div>'
