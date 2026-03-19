from __future__ import annotations

import json
import re
import shlex
import subprocess
import uuid

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from streamlit_python_only.terminal_manager import TerminalManager, classify_terminal_command
from streamlit_python_only.tooling_shared import ToolMetadata, resolve_workspace_path


class TerminalInput(BaseModel):
    command: str = Field(description="Command to execute without shell chaining.")
    cwd: str | None = Field(default=None, description="Optional working directory relative to the workspace root.")
    timeout_sec: int = Field(default=20, ge=1, le=120, description="Maximum execution time in seconds.")


class OpenTerminalInput(BaseModel):
    cwd: str | None = Field(default=None, description="Optional working directory relative to the workspace root.")
    shell: str | None = Field(default=None, description="Optional shell executable path.")
    cols: int | None = Field(default=120, ge=40, le=240, description="Terminal width in columns.")
    rows: int | None = Field(default=36, ge=10, le=120, description="Terminal height in rows.")


class TerminalWriteToolInput(BaseModel):
    terminalId: str | None = Field(default=None, description="Existing terminal identifier.")
    data: str = Field(description="Text to write into the terminal.")


class TerminalInterruptToolInput(BaseModel):
    terminalId: str | None = Field(default=None, description="Existing terminal identifier.")


class TerminalControlToolInput(BaseModel):
    terminalId: str | None = Field(default=None, description="Existing terminal identifier.")


class TerminalSnapshotInput(BaseModel):
    terminalId: str | None = Field(default=None, description="Existing terminal identifier.")


class TerminalWaitInput(BaseModel):
    terminalId: str | None = Field(default=None, description="Existing terminal identifier.")
    pattern: str = Field(description="Text or regex to wait for in terminal output.")
    timeoutMs: int = Field(default=30_000, ge=100, le=300_000, description="How long to wait before returning.")
    regex: bool = Field(default=False, description="Interpret pattern as a regular expression.")


class TerminalCloseInput(BaseModel):
    terminalId: str | None = Field(default=None, description="Existing terminal identifier.")


def build_terminal_tool_definitions(
    workspace_root: str,
    session_id: str | None,
    run_id: str | None,
    terminal_manager: TerminalManager,
) -> list[ToolMetadata]:
    def resolve_tool_terminal_id(terminal_id: str | None, *, require_alive: bool = True) -> dict[str, str | None]:
        return terminal_manager.resolve_terminal_reference(
            terminal_id,
            run_id=run_id,
            session_id=session_id,
            require_alive=require_alive,
        )

    def run_terminal(command: str, cwd: str | None = None, timeout_sec: int = 20) -> str:
        terminal_id = str(uuid.uuid4())
        requested = str(command or "").strip()
        policy = classify_terminal_command(requested)
        target_cwd = resolve_workspace_path(workspace_root, cwd or ".")
        if policy["blocked"] or re.search(r"&&|\|\||`|\$\(|[|<>]|;(?=\s*\S)", requested):
            return json.dumps({"terminalId": terminal_id, "command": requested, "cwd": str(target_cwd), "blocked": True, "exitCode": 126, "stdout": "", "stderr": f"Blocked by terminal policy: {policy['reason']}"}, ensure_ascii=False)
        try:
            args = shlex.split(requested)
        except ValueError as exc:
            return json.dumps({"terminalId": terminal_id, "command": requested, "cwd": str(target_cwd), "blocked": True, "exitCode": 126, "stdout": "", "stderr": f"Command parsing failed: {exc}"}, ensure_ascii=False)
        if not args:
            return json.dumps({"terminalId": terminal_id, "command": requested, "cwd": str(target_cwd), "blocked": True, "exitCode": 126, "stdout": "", "stderr": "No command provided."}, ensure_ascii=False)
        try:
            completed = subprocess.run(args, cwd=target_cwd, capture_output=True, text=True, timeout=max(1, int(timeout_sec)), check=False)
            return json.dumps({"terminalId": terminal_id, "command": requested, "cwd": str(target_cwd), "blocked": False, "exitCode": int(completed.returncode), "stdout": completed.stdout[-8000:], "stderr": completed.stderr[-4000:]}, ensure_ascii=False)
        except subprocess.TimeoutExpired:
            return json.dumps({"terminalId": terminal_id, "command": requested, "cwd": str(target_cwd), "blocked": False, "exitCode": 124, "stdout": "", "stderr": "Command timed out."}, ensure_ascii=False)

    def open_terminal(cwd: str | None = None, shell: str | None = None, cols: int | None = 120, rows: int | None = 36) -> str:
        created = terminal_manager.create_terminal(workspace_root=workspace_root, session_id=session_id, run_id=run_id, cwd=cwd, shell=shell, cols=cols, rows=rows, owner="agent")
        return json.dumps({"kind": "terminal_tool", "tool": "open_terminal", "terminal": created["terminal"], "events": created["events"], "resolution": {"terminalId": created["terminal"]["terminalId"], "strategy": "created"}}, ensure_ascii=False)

    def terminal_write(terminalId: str | None, data: str) -> str:
        normalized = data.strip().lower()
        if any(normalized == cmd or normalized.startswith(f"{cmd} ") for cmd in {"nano", "vim", "vi", "nvim", "less", "more", "top", "htop"}):
            raise ValueError("Interactive terminal editors are blocked in agent terminal mode. Use write_file/read_file tools instead.")
        resolution = resolve_tool_terminal_id(terminalId)
        terminal_id = str(resolution["terminalId"] or "")
        updated = terminal_manager.write_terminal(terminal_id, data, "agent")
        return json.dumps({"kind": "terminal_tool", "tool": "terminal_write", "terminal": updated["terminal"], "events": updated.get("events", []), "preview": updated["terminal"].get("tail", "")[-1000:], "resolution": resolution}, ensure_ascii=False)

    def terminal_interrupt(terminalId: str | None) -> str:
        resolution = resolve_tool_terminal_id(terminalId)
        terminal_id = str(resolution["terminalId"] or "")
        updated = terminal_manager.interrupt_terminal(terminal_id, "agent")
        return json.dumps({"kind": "terminal_tool", "tool": "terminal_interrupt", "terminal": updated["terminal"], "events": updated.get("events", []), "interrupted": True, "resolution": resolution}, ensure_ascii=False)

    def terminal_release_control(terminalId: str | None) -> str:
        resolution = resolve_tool_terminal_id(terminalId)
        terminal_id = str(resolution["terminalId"] or "")
        updated = terminal_manager.set_control(terminal_id, "user", "agent_release")
        return json.dumps({"kind": "terminal_tool", "tool": "terminal_release_control", "terminal": updated["terminal"], "events": updated.get("events", []), "resolution": resolution}, ensure_ascii=False)

    def terminal_request_control(terminalId: str | None) -> str:
        resolution = resolve_tool_terminal_id(terminalId)
        terminal_id = str(resolution["terminalId"] or "")
        updated = terminal_manager.set_control(terminal_id, "agent", "agent_request")
        return json.dumps({"kind": "terminal_tool", "tool": "terminal_request_control", "terminal": updated["terminal"], "events": updated.get("events", []), "resolution": resolution}, ensure_ascii=False)

    def terminal_snapshot(terminalId: str | None) -> str:
        resolution = resolve_tool_terminal_id(terminalId)
        terminal_id = str(resolution["terminalId"] or "")
        snapshot = terminal_manager.terminal_snapshot(terminal_id)
        return json.dumps({"kind": "terminal_tool", "tool": "terminal_snapshot", "terminal": terminal_manager.get_terminal(terminal_id)["terminal"], "events": [], "snapshot": snapshot, "resolution": resolution}, ensure_ascii=False)

    def terminal_wait_for_output(terminalId: str | None, pattern: str, timeoutMs: int = 30_000, regex: bool = False) -> str:
        resolution = resolve_tool_terminal_id(terminalId)
        terminal_id = str(resolution["terminalId"] or "")
        waited = terminal_manager.wait_for_output(terminal_id, pattern=pattern, timeout_ms=timeoutMs, regex=regex)
        return json.dumps({"kind": "terminal_tool", "tool": "terminal_wait_for_output", "terminal": waited["terminal"], "events": [], "matched": waited["matched"], "tail": str(waited["tail"])[-8000:], "resolution": resolution}, ensure_ascii=False)

    def terminal_close(terminalId: str | None) -> str:
        resolution = resolve_tool_terminal_id(terminalId, require_alive=False)
        terminal_id = str(resolution["terminalId"] or "")
        updated = terminal_manager.close_terminal(terminal_id, "agent_closed")
        return json.dumps({"kind": "terminal_tool", "tool": "terminal_close", "terminal": updated["terminal"], "events": updated.get("events", []), "closed": True, "resolution": resolution}, ensure_ascii=False)

    specs = [
        (StructuredTool.from_function(run_terminal, name="run_terminal", description="Run a single safe terminal command inside the workspace root without shell chaining.", args_schema=TerminalInput), "safe"),
        (StructuredTool.from_function(open_terminal, name="open_terminal", description="Open an interactive PTY terminal in the workspace and take agent control.", args_schema=OpenTerminalInput), "moderate"),
        (StructuredTool.from_function(terminal_write, name="terminal_write", description="Write text to an interactive terminal already controlled by the agent.", args_schema=TerminalWriteToolInput), "risky"),
        (StructuredTool.from_function(terminal_interrupt, name="terminal_interrupt", description="Send Ctrl+C to the interactive terminal.", args_schema=TerminalInterruptToolInput), "moderate"),
        (StructuredTool.from_function(terminal_release_control, name="terminal_release_control", description="Release terminal control back to the user.", args_schema=TerminalControlToolInput), "safe"),
        (StructuredTool.from_function(terminal_request_control, name="terminal_request_control", description="Take terminal control from the user.", args_schema=TerminalControlToolInput), "moderate"),
        (StructuredTool.from_function(terminal_snapshot, name="terminal_snapshot", description="Read recent interactive terminal output for reasoning.", args_schema=TerminalSnapshotInput), "safe"),
        (StructuredTool.from_function(terminal_wait_for_output, name="terminal_wait_for_output", description="Wait until terminal output contains a pattern or regex, then return the recent tail.", args_schema=TerminalWaitInput), "safe"),
        (StructuredTool.from_function(terminal_close, name="terminal_close", description="Close an interactive terminal session.", args_schema=TerminalCloseInput), "moderate"),
    ]
    return [
        ToolMetadata(
            tool=tool,
            risk_level=risk_level,
            module_id="terminal",
            supports_thales=True,
            supports_native_tools=True,
            supports_textual_replay=True,
            returns_evidence_kind="terminal",
        )
        for tool, risk_level in specs
    ]
