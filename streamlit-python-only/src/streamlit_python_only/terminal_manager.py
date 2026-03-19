from __future__ import annotations

import os
import queue
import re
import shlex
import signal
import struct
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, BinaryIO

from streamlit_python_only.events import (
    now_iso,
    terminal_closed,
    terminal_control_changed,
    terminal_data,
    terminal_error,
    terminal_exit,
    terminal_input,
    terminal_opened,
    terminal_resized,
)

try:
    import fcntl
    import termios
except ImportError:  # pragma: no cover - non-POSIX fallback is intentionally limited
    fcntl = None
    termios = None


TerminalOwner = str

TERMINAL_ID_PLACEHOLDERS = {
    "",
    "__FROM_TOOL__",
    "__ACTIVE_TERMINAL__",
    "$ACTIVE_TERMINAL",
    "ACTIVE_TERMINAL",
    "active",
    "current",
}


def _is_path_inside_root(root: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(root)
        return True
    except ValueError:
        return candidate == root


def resolve_terminal_cwd(workspace_root: str, input_path: str | None) -> Path:
    root = Path(workspace_root).resolve()
    safe_path = (input_path or ".").strip() or "."
    if safe_path.startswith("/workspace/"):
        safe_path = safe_path[len("/workspace/") :]
    elif safe_path in {"/workspace", "workspace"}:
        safe_path = "."
    elif os.path.isabs(safe_path):
        candidate = Path(safe_path).resolve()
        if not _is_path_inside_root(root, candidate):
            safe_path = "."
    resolved = root.joinpath(safe_path).resolve()
    if not _is_path_inside_root(root, resolved):
        raise ValueError("Terminal cwd escapes workspace root")
    return resolved


def _default_terminal_shell() -> str:
    if os.name == "nt":
        return os.environ.get("COMSPEC") or "C:\\Windows\\System32\\cmd.exe"
    return os.environ.get("SHELL") or "/bin/bash"


def _shell_spawn_args(shell: str) -> list[str]:
    shell_name = Path(shell).name.lower()
    if shell_name in {"powershell.exe", "powershell", "pwsh.exe", "pwsh"}:
        return [shell, "-NoLogo"]
    if shell_name in {"cmd.exe", "cmd"}:
        return [shell, "/Q", "/K"]
    if shell_name in {"bash", "zsh", "sh"}:
        return [shell, "-i"]
    return [shell]


def _clamp_terminal_size(cols: int | None, rows: int | None) -> tuple[int, int]:
    return max(40, min(cols or 120, 240)), max(10, min(rows or 36, 120))


@dataclass(slots=True)
class TerminalSnapshot:
    terminal_id: str
    run_id: str | None
    session_id: str | None
    cwd: str
    shell: str
    owner: TerminalOwner
    alive: bool
    created_at: str
    updated_at: str
    tail: str
    backend: str = "pty"
    cols: int = 120
    rows: int = 36

    def as_dict(self) -> dict[str, Any]:
        return {
            "terminalId": self.terminal_id,
            "runId": self.run_id,
            "sessionId": self.session_id,
            "cwd": self.cwd,
            "shell": self.shell,
            "owner": self.owner,
            "alive": self.alive,
            "createdAt": self.created_at,
            "updatedAt": self.updated_at,
            "tail": self.tail,
            "backend": self.backend,
            "cols": self.cols,
            "rows": self.rows,
        }


@dataclass(slots=True)
class TerminalRecord:
    terminal_id: str
    run_id: str | None
    session_id: str | None
    cwd: str
    shell: str
    owner: TerminalOwner
    created_at: str
    updated_at: str
    process: subprocess.Popen[bytes]
    cols: int
    rows: int
    master_fd: int | None = None
    alive: bool = True
    tail: str = ""
    backend: str = "pty"
    subscribers: list[queue.Queue[dict[str, Any]]] = field(default_factory=list)
    lock: threading.RLock = field(default_factory=threading.RLock)
    condition: threading.Condition = field(init=False)

    def __post_init__(self) -> None:
        self.condition = threading.Condition(self.lock)

    def snapshot(self) -> TerminalSnapshot:
        return TerminalSnapshot(
            terminal_id=self.terminal_id,
            run_id=self.run_id,
            session_id=self.session_id,
            cwd=self.cwd,
            shell=self.shell,
            owner=self.owner,
            alive=self.alive,
            created_at=self.created_at,
            updated_at=self.updated_at,
            tail=self.tail,
            backend=self.backend,
            cols=self.cols,
            rows=self.rows,
        )


def tokenize_command_head(command_input: str) -> list[str]:
    try:
        return shlex.split(command_input, posix=True)
    except ValueError:
        return command_input.strip().split()


def classify_terminal_command(command_input: str) -> dict[str, Any]:
    original_command = str(command_input or "").strip()
    if not original_command:
        return {"riskLevel": "risky", "blocked": True, "reason": "Empty command is not allowed.", "commandName": ""}
    command = re.sub(r"(?:;\s*)+$", "", original_command).strip() or original_command
    if re.search(r"&&|\|\||`|\$\(", command) or re.search(r";(?=\s*\S)", command):
        tokens = tokenize_command_head(command)
        return {
            "riskLevel": "risky",
            "blocked": False,
            "reason": "Shell chaining/substitution operators detected; require approval under cautious policies.",
            "commandName": (tokens[0].lower() if tokens else ""),
        }
    tokens = tokenize_command_head(command)
    base = (tokens[0] if tokens else "").lower()
    if not base:
        return {"riskLevel": "risky", "blocked": True, "reason": "Unable to parse command.", "commandName": ""}
    blocked_commands = {
        "sudo",
        "su",
        "doas",
        "rm",
        "dd",
        "mkfs",
        "fdisk",
        "diskutil",
        "shutdown",
        "reboot",
        "halt",
        "poweroff",
        "launchctl",
        "pkill",
        "killall",
        "chown",
        "chmod",
    }
    if base in blocked_commands:
        return {
            "riskLevel": "risky",
            "blocked": False,
            "reason": f"Command '{base}' is destructive or privileged; require approval under cautious policies.",
            "commandName": base,
        }
    blocked_interactive = {"vim", "vi", "nano", "emacs", "pico", "less", "more", "top", "htop"}
    if base in blocked_interactive:
        editor_hint = (
            "Use read_file or non-interactive inspection commands instead."
            if base in {"less", "more", "top", "htop"}
            else "Use write_file instead of an interactive terminal editor."
        )
        return {
            "riskLevel": "risky",
            "blocked": True,
            "reason": f"Command '{base}' is blocked by terminal policy (interactive session not supported). {editor_hint}",
            "commandName": base,
        }
    if base == "git":
        sub = (tokens[1] if len(tokens) > 1 else "").lower()
        safe_git = {"status", "log", "diff", "show", "branch", "rev-parse"}
        blocked_git = {"reset", "clean", "checkout", "restore", "rebase", "push"}
        if sub in blocked_git:
            return {
                "riskLevel": "risky",
                "blocked": False,
                "reason": f"git {sub} may modify repository state; require approval under cautious policies.",
                "commandName": "git",
            }
        if sub in safe_git:
            return {"riskLevel": "safe", "blocked": False, "reason": f"Read-only git subcommand '{sub}'.", "commandName": "git"}
        return {"riskLevel": "risky", "blocked": False, "reason": f"git {sub or '(unknown)'} may modify repository state.", "commandName": "git"}
    safe_commands = {
        "pwd",
        "ls",
        "cat",
        "head",
        "tail",
        "wc",
        "grep",
        "rg",
        "find",
        "stat",
        "file",
        "echo",
        "printf",
        "cut",
        "sort",
        "uniq",
        "du",
        "which",
        "env",
        "ps",
        "date",
    }
    if base in safe_commands:
        return {"riskLevel": "safe", "blocked": False, "reason": f"Read/inspect command '{base}'.", "commandName": base}
    moderate_commands = {"mkdir", "touch", "cp", "mv"}
    if base in moderate_commands:
        return {"riskLevel": "moderate", "blocked": False, "reason": f"Filesystem mutation command '{base}'.", "commandName": base}
    risky_commands = {"bash", "sh", "zsh", "python", "python3", "node", "npm", "pnpm", "yarn", "npx", "make", "cmake", "cargo", "go", "pip", "pip3", "curl", "wget"}
    if base in risky_commands or re.search(r"[|<>]", command):
        return {
            "riskLevel": "risky",
            "blocked": False,
            "reason": f"Execution/network/package command '{base}'." if base in risky_commands else "Shell piping/redirection detected; requires approval under cautious policies.",
            "commandName": base,
        }
    return {"riskLevel": "risky", "blocked": False, "reason": f"Unclassified command '{base}' defaults to risky.", "commandName": base}


class TerminalManager:
    def __init__(self) -> None:
        self._terminals: dict[str, TerminalRecord] = {}
        self._active_by_run: dict[str, str] = {}
        self._active_by_session: dict[str, str] = {}
        self._lock = threading.RLock()

    def _emit(self, record: TerminalRecord, event: dict[str, Any]) -> dict[str, Any]:
        with record.lock:
            record.updated_at = str(event.get("timestamp") or now_iso())
            for subscriber in list(record.subscribers):
                subscriber.put(event)
            record.condition.notify_all()
        return event

    def _append_tail(self, record: TerminalRecord, chunk: str) -> None:
        with record.lock:
            record.tail = (record.tail + chunk)[-20_000:]
            record.updated_at = now_iso()
            record.condition.notify_all()

    def _mark_active(self, record: TerminalRecord) -> None:
        with self._lock:
            if record.run_id:
                self._active_by_run[record.run_id] = record.terminal_id
            if record.session_id:
                self._active_by_session[record.session_id] = record.terminal_id

    def _clear_active(self, record: TerminalRecord) -> None:
        with self._lock:
            if record.run_id and self._active_by_run.get(record.run_id) == record.terminal_id:
                replacement = self._latest_terminal_locked(run_id=record.run_id, alive_only=True, exclude_terminal_id=record.terminal_id)
                if replacement:
                    self._active_by_run[record.run_id] = replacement.terminal_id
                else:
                    self._active_by_run.pop(record.run_id, None)
            if record.session_id and self._active_by_session.get(record.session_id) == record.terminal_id:
                replacement = self._latest_terminal_locked(session_id=record.session_id, alive_only=True, exclude_terminal_id=record.terminal_id)
                if replacement:
                    self._active_by_session[record.session_id] = replacement.terminal_id
                else:
                    self._active_by_session.pop(record.session_id, None)

    def _iter_filtered_locked(
        self,
        *,
        run_id: str | None = None,
        session_id: str | None = None,
        alive_only: bool = False,
        exclude_terminal_id: str | None = None,
    ) -> list[TerminalRecord]:
        records = list(self._terminals.values())
        if run_id is not None:
            records = [record for record in records if record.run_id == run_id]
        if session_id is not None:
            records = [record for record in records if record.session_id == session_id]
        if alive_only:
            records = [record for record in records if record.alive]
        if exclude_terminal_id:
            records = [record for record in records if record.terminal_id != exclude_terminal_id]
        return sorted(records, key=lambda record: (record.updated_at, record.created_at, record.terminal_id))

    def _latest_terminal_locked(
        self,
        *,
        run_id: str | None = None,
        session_id: str | None = None,
        alive_only: bool = False,
        exclude_terminal_id: str | None = None,
    ) -> TerminalRecord | None:
        candidates = self._iter_filtered_locked(
            run_id=run_id,
            session_id=session_id,
            alive_only=alive_only,
            exclude_terminal_id=exclude_terminal_id,
        )
        return candidates[-1] if candidates else None

    def _read_stream(self, record: TerminalRecord, stream: BinaryIO | None, stream_name: str) -> None:
        if stream is None:
            return
        try:
            while True:
                data = stream.read(4096)
                if not data:
                    break
                chunk = data.decode("utf-8", errors="replace")
                self._append_tail(record, chunk)
                self._emit(
                    record,
                    terminal_data(record.run_id, record.terminal_id, chunk[-8000:], stream=stream_name, session_id=record.session_id),
                )
        finally:
            with record.lock:
                record.condition.notify_all()

    def _start_reader_threads(self, record: TerminalRecord) -> None:
        def pty_reader() -> None:
            try:
                while True:
                    try:
                        data = os.read(record.master_fd or -1, 4096)
                    except OSError:
                        break
                    if not data:
                        break
                    chunk = data.decode("utf-8", errors="replace")
                    self._append_tail(record, chunk)
                    self._emit(
                        record,
                        terminal_data(record.run_id, record.terminal_id, chunk[-8000:], stream="pty", session_id=record.session_id),
                    )
            finally:
                with record.lock:
                    record.condition.notify_all()

        def waiter() -> None:
            exit_code = None
            try:
                exit_code = record.process.wait()
            finally:
                with record.lock:
                    record.alive = False
                    record.condition.notify_all()
                self._emit(
                    record,
                    terminal_exit(record.run_id, record.terminal_id, exit_code, "", session_id=record.session_id),
                )
                self._clear_active(record)
                if record.master_fd is not None:
                    try:
                        os.close(record.master_fd)
                    except OSError:
                        pass

        if record.backend == "pty" and record.master_fd is not None:
            threading.Thread(target=pty_reader, daemon=True, name=f"terminal-reader-{record.terminal_id}").start()
        else:
            threading.Thread(
                target=self._read_stream,
                args=(record, record.process.stdout, "stdout"),
                daemon=True,
                name=f"terminal-stdout-{record.terminal_id}",
            ).start()
            threading.Thread(
                target=self._read_stream,
                args=(record, record.process.stderr, "stderr"),
                daemon=True,
                name=f"terminal-stderr-{record.terminal_id}",
            ).start()
        threading.Thread(target=waiter, daemon=True, name=f"terminal-waiter-{record.terminal_id}").start()

    def _spawn_pty_process(
        self,
        *,
        shell: str,
        cwd: Path,
        cols: int,
        rows: int,
    ) -> tuple[subprocess.Popen[bytes], int, str]:
        if not hasattr(os, "openpty"):
            raise OSError("os.openpty is unavailable on this platform")
        master_fd, slave_fd = os.openpty()
        try:
            if fcntl is not None and termios is not None:
                winsz = struct.pack("HHHH", rows, cols, 0, 0)
                fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, winsz)
            process = subprocess.Popen(
                _shell_spawn_args(shell),
                cwd=str(cwd),
                stdin=slave_fd,
                stdout=slave_fd,
                stderr=slave_fd,
                start_new_session=True,
                close_fds=True,
                env={**os.environ, "TERM": os.environ.get("TERM", "xterm-256color")},
            )
        finally:
            os.close(slave_fd)
        return process, master_fd, "pty"

    def _spawn_pipe_process(
        self,
        *,
        shell: str,
        cwd: Path,
    ) -> tuple[subprocess.Popen[bytes], None, str]:
        process = subprocess.Popen(
            _shell_spawn_args(shell),
            cwd=str(cwd),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=(os.name != "nt"),
            close_fds=(os.name != "nt"),
            env={
                **os.environ,
                "TERM": os.environ.get("TERM", "xterm-256color"),
                "PROMPT": os.environ.get("PROMPT", "$P$G"),
            },
        )
        return process, None, "pipe"

    def create_terminal(
        self,
        *,
        workspace_root: str,
        session_id: str | None = None,
        run_id: str | None = None,
        cwd: str | None = None,
        shell: str | None = None,
        cols: int | None = None,
        rows: int | None = None,
        owner: TerminalOwner = "agent",
    ) -> dict[str, Any]:
        resolved_cwd = resolve_terminal_cwd(workspace_root, cwd)
        selected_shell = shell or _default_terminal_shell()
        terminal_id = str(uuid.uuid4())
        created_at = now_iso()
        resolved_cols, resolved_rows = _clamp_terminal_size(cols, rows)
        emitted_events: list[dict[str, Any]] = []

        try:
            process, master_fd, backend = self._spawn_pty_process(
                shell=selected_shell,
                cwd=resolved_cwd,
                cols=resolved_cols,
                rows=resolved_rows,
            )
            fallback_reason = None
        except Exception as exc:
            process, master_fd, backend = self._spawn_pipe_process(shell=selected_shell, cwd=resolved_cwd)
            fallback_reason = str(exc)

        record = TerminalRecord(
            terminal_id=terminal_id,
            run_id=run_id,
            session_id=session_id,
            cwd=str(resolved_cwd),
            shell=selected_shell,
            owner=owner,
            created_at=created_at,
            updated_at=created_at,
            process=process,
            cols=resolved_cols,
            rows=resolved_rows,
            master_fd=master_fd,
            backend=backend,
        )
        with self._lock:
            self._terminals[terminal_id] = record
        self._mark_active(record)
        self._start_reader_threads(record)

        opened = self._emit(
            record,
            terminal_opened(
                run_id,
                terminal_id,
                command="",
                cwd=str(resolved_cwd),
                session_id=session_id,
                shell=selected_shell,
                backend=backend,
            )
            | {"timestamp": created_at},
        )
        emitted_events.append(opened)
        owner_event = self._emit(
            record,
            terminal_control_changed(run_id, terminal_id, owner, reason="terminal_created", session_id=session_id)
            | {"timestamp": created_at},
        )
        emitted_events.append(owner_event)
        if fallback_reason and os.environ.get("TERMINAL_EMIT_PTY_FALLBACK_WARNING") == "1":
            emitted_events.append(
                self._emit(
                    record,
                    terminal_error(
                        run_id,
                        terminal_id,
                        f"PTY unavailable, fallback to pipe terminal: {fallback_reason}",
                        session_id=session_id,
                    )
                    | {"timestamp": created_at},
                )
            )
        return {"terminal": record.snapshot().as_dict(), "events": emitted_events}

    def get_terminal(self, terminal_id: str) -> dict[str, Any]:
        record = self._require_terminal(terminal_id)
        self._mark_active(record)
        return {"terminal": record.snapshot().as_dict()}

    def list_terminals(
        self,
        *,
        run_id: str | None = None,
        session_id: str | None = None,
        alive_only: bool = False,
    ) -> dict[str, Any]:
        with self._lock:
            records = self._iter_filtered_locked(run_id=run_id, session_id=session_id, alive_only=alive_only)
            active_terminal_id = None
            if run_id:
                active_terminal_id = self._active_by_run.get(run_id)
            if not active_terminal_id and session_id:
                active_terminal_id = self._active_by_session.get(session_id)
            if active_terminal_id and active_terminal_id not in {record.terminal_id for record in records}:
                active_terminal_id = None
        return {
            "terminals": [record.snapshot().as_dict() for record in records],
            "activeTerminalId": active_terminal_id,
        }

    def list_by_run(self, run_id: str) -> list[dict[str, Any]]:
        return self.list_terminals(run_id=run_id)["terminals"]

    def list_by_session(self, session_id: str, *, alive_only: bool = False) -> list[dict[str, Any]]:
        return self.list_terminals(session_id=session_id, alive_only=alive_only)["terminals"]

    def _require_terminal(self, terminal_id: str) -> TerminalRecord:
        with self._lock:
            record = self._terminals.get(terminal_id)
        if not record:
            raise KeyError("Terminal not found")
        return record

    def resolve_terminal_reference(
        self,
        terminal_id: str | None,
        *,
        run_id: str | None = None,
        session_id: str | None = None,
        require_alive: bool = True,
    ) -> dict[str, Any]:
        requested = str(terminal_id or "").strip()
        if requested and requested not in TERMINAL_ID_PLACEHOLDERS:
            record = self._require_terminal(requested)
            if require_alive and not record.alive:
                raise RuntimeError("Referenced terminal has already exited")
            self._mark_active(record)
            return {
                "terminalId": record.terminal_id,
                "strategy": "explicit",
                "requestedTerminalId": terminal_id,
            }

        with self._lock:
            candidate: TerminalRecord | None = None
            strategy = ""
            if run_id:
                active_terminal_id = self._active_by_run.get(run_id)
                active_record = self._terminals.get(active_terminal_id) if active_terminal_id else None
                if active_record and (not require_alive or active_record.alive):
                    candidate = active_record
                    strategy = "active_run_terminal"
            if candidate is None and session_id:
                active_terminal_id = self._active_by_session.get(session_id)
                active_record = self._terminals.get(active_terminal_id) if active_terminal_id else None
                if active_record and (not require_alive or active_record.alive):
                    candidate = active_record
                    strategy = "active_session_terminal"
            if candidate is None and run_id:
                run_candidates = self._iter_filtered_locked(run_id=run_id, alive_only=require_alive)
                if len(run_candidates) == 1:
                    candidate = run_candidates[0]
                    strategy = "single_run_terminal"
                elif len(run_candidates) > 1:
                    raise ValueError("terminalId is required because multiple terminals are active for this run")
            if candidate is None and session_id:
                session_candidates = self._iter_filtered_locked(session_id=session_id, alive_only=require_alive)
                if len(session_candidates) == 1:
                    candidate = session_candidates[0]
                    strategy = "single_session_terminal"
                elif len(session_candidates) > 1:
                    raise ValueError("terminalId is required because multiple terminals are active for this session")
            if candidate is None:
                all_candidates = self._iter_filtered_locked(alive_only=require_alive)
                if len(all_candidates) == 1:
                    candidate = all_candidates[0]
                    strategy = "single_terminal"

        if candidate is None:
            raise ValueError("terminalId is required because no matching active terminal was found")
        self._mark_active(candidate)
        return {
            "terminalId": candidate.terminal_id,
            "strategy": strategy,
            "requestedTerminalId": terminal_id,
        }

    def write_terminal(self, terminal_id: str, data: str, source: TerminalOwner = "user") -> dict[str, Any]:
        record = self._require_terminal(terminal_id)
        payload = data or ""
        input_kind = "interrupt" if payload == "\u0003" else "text"
        with record.lock:
            if not record.alive:
                raise RuntimeError("Terminal has already exited")
            if source == "user" and record.owner == "agent" and input_kind != "interrupt":
                raise PermissionError("Terminal is under agent control. Only Ctrl+C is allowed.")
            if record.backend == "pty" and record.master_fd is not None:
                os.write(record.master_fd, payload.encode("utf-8"))
            else:
                if record.process.stdin is None:
                    raise RuntimeError("Terminal stdin is unavailable")
                record.process.stdin.write(payload.encode("utf-8"))
                record.process.stdin.flush()
            record.updated_at = now_iso()
        self._mark_active(record)
        event = self._emit(
            record,
            terminal_input(
                record.run_id,
                terminal_id,
                source=source,
                data=payload,
                session_id=record.session_id,
                input_kind=input_kind,
            ),
        )
        return {"terminal": record.snapshot().as_dict(), "events": [event]}

    def interrupt_terminal(self, terminal_id: str, source: TerminalOwner = "user") -> dict[str, Any]:
        return self.write_terminal(terminal_id, "\u0003", source)

    def resize_terminal(self, terminal_id: str, cols: int, rows: int) -> dict[str, Any]:
        record = self._require_terminal(terminal_id)
        resolved_cols, resolved_rows = _clamp_terminal_size(cols, rows)
        with record.lock:
            if not record.alive:
                raise RuntimeError("Terminal has already exited")
            if record.backend == "pty" and record.master_fd is not None and fcntl is not None and termios is not None:
                winsz = struct.pack("HHHH", resolved_rows, resolved_cols, 0, 0)
                fcntl.ioctl(record.master_fd, termios.TIOCSWINSZ, winsz)
            record.cols = resolved_cols
            record.rows = resolved_rows
            record.updated_at = now_iso()
        self._mark_active(record)
        event = self._emit(
            record,
            terminal_resized(record.run_id, terminal_id, resolved_cols, resolved_rows, session_id=record.session_id),
        )
        return {"terminal": record.snapshot().as_dict(), "events": [event]}

    def set_control(self, terminal_id: str, owner: TerminalOwner, reason: str | None = None) -> dict[str, Any]:
        record = self._require_terminal(terminal_id)
        with record.lock:
            record.owner = owner
        self._mark_active(record)
        event = self._emit(
            record,
            terminal_control_changed(record.run_id, terminal_id, owner, reason=reason, session_id=record.session_id),
        )
        return {"terminal": record.snapshot().as_dict(), "event": event, "events": [event]}

    def close_terminal(self, terminal_id: str, reason: str = "closed") -> dict[str, Any]:
        record = self._require_terminal(terminal_id)
        events: list[dict[str, Any]] = []
        with record.lock:
            if record.alive:
                try:
                    if os.name == "nt":
                        record.process.terminate()
                    else:
                        record.process.send_signal(signal.SIGTERM)
                except Exception as exc:
                    events.append(
                        self._emit(
                            record,
                            terminal_error(record.run_id, terminal_id, str(exc), session_id=record.session_id),
                        )
                    )
                record.alive = False
                closed_event = self._emit(
                    record,
                    terminal_closed(record.run_id, terminal_id, reason=reason, by="system", session_id=record.session_id),
                )
                control_event = self._emit(
                    record,
                    terminal_control_changed(record.run_id, terminal_id, "user", reason=reason, session_id=record.session_id),
                )
                events.extend([closed_event, control_event])
        self._clear_active(record)
        return {"terminal": record.snapshot().as_dict(), "events": events}

    def close_run_terminals(self, run_id: str, reason: str = "run_cancelled") -> dict[str, Any]:
        with self._lock:
            terminal_ids = [record.terminal_id for record in self._terminals.values() if record.run_id == run_id and record.alive]
        all_events: list[dict[str, Any]] = []
        snapshots: list[dict[str, Any]] = []
        for terminal_id in terminal_ids:
            result = self.close_terminal(terminal_id, reason=reason)
            all_events.extend(result.get("events", []))
            snapshots.append(result["terminal"])
        return {"terminals": snapshots, "events": all_events}

    def terminal_snapshot(self, terminal_id: str) -> dict[str, Any]:
        record = self._require_terminal(terminal_id)
        self._mark_active(record)
        return {
            "terminalId": record.terminal_id,
            "owner": record.owner,
            "alive": record.alive,
            "cwd": record.cwd,
            "tail": record.tail[-8000:],
            "cols": record.cols,
            "rows": record.rows,
            "backend": record.backend,
        }

    def wait_for_output(self, terminal_id: str, *, pattern: str, timeout_ms: int = 30_000, regex: bool = False) -> dict[str, Any]:
        record = self._require_terminal(terminal_id)
        timeout_sec = max(0.1, min(timeout_ms / 1000.0, 300.0))
        matcher = None
        if pattern:
            if regex:
                compiled = re.compile(pattern, re.MULTILINE)
                matcher = lambda text: bool(compiled.search(text))
            else:
                matcher = lambda text: pattern in text
        deadline = time.monotonic() + timeout_sec
        tail = ""
        with record.lock:
            while True:
                tail = record.tail
                if matcher and matcher(tail):
                    break
                if not record.alive or time.monotonic() >= deadline:
                    break
                record.condition.wait(timeout=max(0.05, deadline - time.monotonic()))
        self._mark_active(record)
        return {
            "matched": bool(matcher(tail)) if matcher else False,
            "tail": tail,
            "terminal": record.snapshot().as_dict(),
        }

    def stream_subscription(self, terminal_id: str) -> tuple[queue.Queue[dict[str, Any]], list[dict[str, Any]], TerminalRecord]:
        record = self._require_terminal(terminal_id)
        self._mark_active(record)
        subscription: queue.Queue[dict[str, Any]] = queue.Queue()
        with record.lock:
            record.subscribers.append(subscription)
            initial: list[dict[str, Any]] = []
            if record.tail:
                initial.append(
                    terminal_data(record.run_id, record.terminal_id, record.tail, stream=record.backend, session_id=record.session_id)
                )
            initial.append(
                terminal_control_changed(record.run_id, record.terminal_id, record.owner, reason="stream_attached", session_id=record.session_id)
            )
        return subscription, initial, record

    def remove_subscription(self, record: TerminalRecord, subscription: queue.Queue[dict[str, Any]]) -> None:
        with record.lock:
            if subscription in record.subscribers:
                record.subscribers.remove(subscription)


@lru_cache(maxsize=1)
def get_terminal_manager() -> TerminalManager:
    return TerminalManager()
