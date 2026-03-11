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


def build_terminal_component_html(
    *,
    backend_url: str,
    workspace_root: str,
    session_id: str | None,
    run_id: str | None,
    initial_terminal_id: str | None,
) -> str:
    config = {
        "backendUrl": backend_url.rstrip("/"),
        "workspaceRoot": workspace_root,
        "sessionId": session_id,
        "runId": run_id,
        "initialTerminalId": initial_terminal_id,
    }
    payload = html.escape(json.dumps(config, ensure_ascii=False))
    return f"""
    <div class="cb-terminal-shell">
      <div class="cb-terminal-toolbar">
        <div class="cb-terminal-title">
          <span>Interactive Terminal</span>
          <span id="cb-terminal-mode" class="cb-terminal-pill">Idle</span>
        </div>
        <div class="cb-terminal-actions">
          <button id="cb-terminal-open">Open</button>
          <button id="cb-terminal-reconnect">Reconnect</button>
          <button id="cb-terminal-interrupt">Ctrl+C</button>
          <button id="cb-terminal-agent">Agent</button>
          <button id="cb-terminal-user">User</button>
          <button id="cb-terminal-close">Close</button>
        </div>
      </div>
      <div id="cb-terminal-meta" class="cb-terminal-meta-row">No terminal session yet.</div>
      <div id="cb-terminal-host" class="cb-terminal-host"></div>
      <pre id="cb-terminal-fallback" class="cb-terminal-fallback"></pre>
      <script id="cb-terminal-config" type="application/json">{payload}</script>
    </div>
    <style>
      .cb-terminal-shell {{
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        background: rgba(7,12,18,0.78);
        backdrop-filter: blur(10px);
        overflow: hidden;
      }}
      .cb-terminal-toolbar {{
        display: flex;
        justify-content: space-between;
        gap: 10px;
        padding: 10px 12px;
        border-bottom: 1px solid rgba(255,255,255,0.08);
        align-items: center;
        flex-wrap: wrap;
      }}
      .cb-terminal-title {{
        display: flex;
        gap: 8px;
        align-items: center;
        color: #e5edf5;
        font-size: 12px;
        letter-spacing: 0.04em;
        text-transform: uppercase;
      }}
      .cb-terminal-pill {{
        border: 1px solid rgba(148,163,184,0.18);
        border-radius: 999px;
        padding: 2px 8px;
        font-size: 10px;
        color: #9bd7ff;
      }}
      .cb-terminal-actions {{
        display: flex;
        gap: 6px;
        flex-wrap: wrap;
      }}
      .cb-terminal-actions button {{
        background: rgba(255,255,255,0.04);
        color: #f8fbff;
        border: 1px solid rgba(148,163,184,0.18);
        border-radius: 10px;
        padding: 6px 10px;
        cursor: pointer;
        font-size: 12px;
      }}
      .cb-terminal-meta-row {{
        padding: 8px 12px;
        font-size: 12px;
        color: rgba(226,232,240,0.78);
        border-bottom: 1px solid rgba(255,255,255,0.05);
      }}
      .cb-terminal-host {{
        height: 360px;
        background: #05070d;
      }}
      .cb-terminal-fallback {{
        display: none;
        margin: 0;
        height: 360px;
        overflow: auto;
        background: #05070d;
        color: #dbeafe;
        padding: 14px;
        white-space: pre-wrap;
      }}
    </style>
    <script>
      const cfg = JSON.parse(document.getElementById("cb-terminal-config").textContent);
      const host = document.getElementById("cb-terminal-host");
      const fallback = document.getElementById("cb-terminal-fallback");
      const modeEl = document.getElementById("cb-terminal-mode");
      const metaEl = document.getElementById("cb-terminal-meta");
      const state = {{ terminal: null, controller: null, term: null, xtermLoaded: false }};

      function setMode(label, tone) {{
        modeEl.textContent = label;
        modeEl.style.color = tone || "#9bd7ff";
      }}

      function setMeta(text) {{
        metaEl.textContent = text;
      }}

      function appendFallback(text) {{
        fallback.style.display = "block";
        fallback.textContent += text;
        fallback.scrollTop = fallback.scrollHeight;
      }}

      function renderSnapshot(snapshot) {{
        if (!snapshot) return;
        state.terminal = snapshot;
        setMode(snapshot.alive ? (snapshot.owner === "agent" ? "Agent control" : "User control") : "Exited", snapshot.alive ? (snapshot.owner === "agent" ? "#fbbf24" : "#86efac") : "#fca5a5");
        setMeta(`${{snapshot.cwd}} • ${{snapshot.shell || "shell"}} • ${{snapshot.backend || "pty"}}`);
        if (state.term) {{
          state.term.reset();
          if (snapshot.tail) state.term.write(snapshot.tail);
        }} else {{
          fallback.textContent = snapshot.tail || "";
        }}
      }}

      function handleEvent(event) {{
        if (event.type === "terminal_opened") {{
          if (state.terminal) state.terminal.alive = true;
          setMode("Opened", "#9bd7ff");
          setMeta(`${{event.cwd || ""}} • ${{event.shell || "shell"}}`);
        }} else if (event.type === "terminal_data") {{
          if (state.term) state.term.write(event.chunk || "");
          else appendFallback(event.chunk || "");
        }} else if (event.type === "terminal_control_changed") {{
          if (state.terminal) state.terminal.owner = event.owner;
          setMode(event.owner === "agent" ? "Agent control" : "User control", event.owner === "agent" ? "#fbbf24" : "#86efac");
        }} else if (event.type === "terminal_exit") {{
          if (state.terminal) state.terminal.alive = false;
          setMode(`Exited (${{event.exitCode ?? "?"}})`, "#fca5a5");
        }} else if (event.type === "terminal_error") {{
          setMode("Error", "#fca5a5");
          appendFallback("\\n[terminal error] " + (event.error || event.message || "unknown error") + "\\n");
        }}
      }}

      async function fetchJson(path, init) {{
        const response = await fetch(cfg.backendUrl + path, {{
          ...init,
          headers: {{ "Content-Type": "application/json", ...(init && init.headers ? init.headers : {{}}) }},
        }});
        if (!response.ok) throw new Error(await response.text() || `${{response.status}}`);
        return await response.json();
      }}

      async function connectStream() {{
        if (!state.terminal || !state.terminal.terminalId) return;
        if (state.controller) state.controller.abort();
        const controller = new AbortController();
        state.controller = controller;
        setMode("Streaming", "#9bd7ff");
        const response = await fetch(cfg.backendUrl + `/v1/terminals/${{encodeURIComponent(state.terminal.terminalId)}}/stream`, {{ signal: controller.signal }});
        if (!response.ok || !response.body) throw new Error(await response.text() || "stream failed");
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        while (true) {{
          const {{ done, value }} = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, {{ stream: true }});
          const parts = buffer.split("\\n\\n");
          buffer = parts.pop() || "";
          for (const part of parts) {{
            if (!part.startsWith("data: ")) continue;
            const event = JSON.parse(part.slice(6).trim());
            handleEvent(event);
            if (event.type === "terminal_exit") return;
          }}
        }}
      }}

      async function ensureXterm() {{
        if (window.Terminal) return true;
        const css = document.createElement("link");
        css.rel = "stylesheet";
        css.href = "https://cdn.jsdelivr.net/npm/xterm@5.5.0/css/xterm.min.css";
        document.head.appendChild(css);
        await new Promise((resolve, reject) => {{
          const script = document.createElement("script");
          script.src = "https://cdn.jsdelivr.net/npm/xterm@5.5.0/lib/xterm.min.js";
          script.onload = resolve;
          script.onerror = reject;
          document.head.appendChild(script);
        }}).catch(() => false);
        return Boolean(window.Terminal);
      }}

      async function ensureTerminalUi() {{
        const ok = await ensureXterm();
        if (!ok || !window.Terminal) {{
          host.style.display = "none";
          fallback.style.display = "block";
          return;
        }}
        if (state.term) return;
        state.term = new window.Terminal({{
          cursorBlink: true,
          convertEol: true,
          fontSize: 12,
          fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace",
          theme: {{
            background: "#05070d",
            foreground: "#dbeafe",
            cursor: "#93c5fd",
          }},
        }});
        state.term.open(host);
        state.term.onData((data) => {{
          if (!state.terminal || !state.terminal.alive) return;
          if (state.terminal.owner === "agent") {{
            if (data === "\\u0003") void interruptTerminal();
            return;
          }}
          void writeTerminal(data, "user");
        }});
      }}

      async function openTerminal() {{
        await ensureTerminalUi();
        const result = await fetchJson("/v1/terminals", {{
          method: "POST",
          body: JSON.stringify({{
            session_id: cfg.sessionId,
            run_id: cfg.runId,
            workspace_root: cfg.workspaceRoot,
            owner: "user",
          }}),
        }});
        state.terminal = result.terminal;
        renderSnapshot(result.terminal);
        for (const event of result.events || []) handleEvent(event);
        await connectStream();
      }}

      async function loadExisting() {{
        await ensureTerminalUi();
        if (!cfg.initialTerminalId) return;
        try {{
          const result = await fetchJson(`/v1/terminals/${{encodeURIComponent(cfg.initialTerminalId)}}`);
          state.terminal = result.terminal;
          renderSnapshot(result.terminal);
          await connectStream();
        }} catch (_error) {{
          setMeta("No reusable terminal session found.");
        }}
      }}

      async function writeTerminal(data, source) {{
        if (!state.terminal) return;
        await fetchJson(`/v1/terminals/${{encodeURIComponent(state.terminal.terminalId)}}/write`, {{
          method: "POST",
          body: JSON.stringify({{ data, source: source || "user" }}),
        }});
      }}

      async function interruptTerminal() {{
        if (!state.terminal) return;
        await fetchJson(`/v1/terminals/${{encodeURIComponent(state.terminal.terminalId)}}/interrupt`, {{
          method: "POST",
          body: JSON.stringify({{ source: "user" }}),
        }});
      }}

      async function setControl(owner) {{
        if (!state.terminal) return;
        const result = await fetchJson(`/v1/terminals/${{encodeURIComponent(state.terminal.terminalId)}}/control`, {{
          method: "POST",
          body: JSON.stringify({{ owner, reason: owner === "agent" ? "user_granted_agent_control" : "user_requested" }}),
        }});
        state.terminal = result.terminal;
        for (const event of result.events || []) handleEvent(event);
      }}

      async function closeTerminal() {{
        if (!state.terminal) return;
        const result = await fetchJson(`/v1/terminals/${{encodeURIComponent(state.terminal.terminalId)}}/close`, {{ method: "POST" }});
        state.terminal = result.terminal;
        for (const event of result.events || []) handleEvent(event);
      }}

      document.getElementById("cb-terminal-open").addEventListener("click", () => void openTerminal());
      document.getElementById("cb-terminal-reconnect").addEventListener("click", () => void connectStream());
      document.getElementById("cb-terminal-interrupt").addEventListener("click", () => void interruptTerminal());
      document.getElementById("cb-terminal-agent").addEventListener("click", () => void setControl("agent"));
      document.getElementById("cb-terminal-user").addEventListener("click", () => void setControl("user"));
      document.getElementById("cb-terminal-close").addEventListener("click", () => void closeTerminal());
      void loadExisting();
    </script>
    """
