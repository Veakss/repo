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
from typing import Any

from continue_better_py.events import now_iso

try:
    import fcntl
    import termios
except ImportError:  # pragma: no cover - non-POSIX fallback is intentionally limited
    fcntl = None
    termios = None


TerminalOwner = str


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
    master_fd: int
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

    def _start_reader_threads(self, record: TerminalRecord) -> None:
        def reader() -> None:
            try:
                while True:
                    try:
                        data = os.read(record.master_fd, 4096)
                    except OSError:
                        break
                    if not data:
                        break
                    chunk = data.decode("utf-8", errors="replace")
                    self._append_tail(record, chunk)
                    self._emit(
                        record,
                        {
                            "type": "terminal_data",
                            "terminalId": record.terminal_id,
                            "runId": record.run_id,
                            "chunk": chunk[-8000:],
                            "stream": "pty",
                            "timestamp": now_iso(),
                        },
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
                    {
                        "type": "terminal_exit",
                        "terminalId": record.terminal_id,
                        "runId": record.run_id,
                        "exitCode": exit_code,
                        "timestamp": now_iso(),
                    },
                )
                try:
                    os.close(record.master_fd)
                except OSError:
                    pass

        threading.Thread(target=reader, daemon=True, name=f"terminal-reader-{record.terminal_id}").start()
        threading.Thread(target=waiter, daemon=True, name=f"terminal-waiter-{record.terminal_id}").start()

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
        selected_shell = shell or os.environ.get("SHELL") or "/bin/zsh"
        terminal_id = str(uuid.uuid4())
        created_at = now_iso()
        master_fd, slave_fd = os.openpty()
        try:
            if fcntl is not None and termios is not None:
                winsz = struct.pack("HHHH", max(10, min(rows or 36, 120)), max(40, min(cols or 120, 240)), 0, 0)
                fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, winsz)
            process = subprocess.Popen(
                [selected_shell, "-i"],
                cwd=str(resolved_cwd),
                stdin=slave_fd,
                stdout=slave_fd,
                stderr=slave_fd,
                start_new_session=True,
                close_fds=True,
                env={**os.environ, "TERM": os.environ.get("TERM", "xterm-256color")},
            )
        finally:
            os.close(slave_fd)
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
            master_fd=master_fd,
        )
        with self._lock:
            self._terminals[terminal_id] = record
        self._start_reader_threads(record)
        opened = self._emit(
            record,
            {
                "type": "terminal_opened",
                "terminalId": terminal_id,
                "runId": run_id,
                "sessionId": session_id,
                "cwd": str(resolved_cwd),
                "shell": selected_shell,
                "backend": "pty",
                "timestamp": created_at,
            },
        )
        owner_event = self._emit(
            record,
            {
                "type": "terminal_control_changed",
                "terminalId": terminal_id,
                "runId": run_id,
                "owner": owner,
                "reason": "terminal_created",
                "timestamp": created_at,
            },
        )
        return {"terminal": record.snapshot().as_dict(), "events": [opened, owner_event]}

    def get_terminal(self, terminal_id: str) -> dict[str, Any]:
        record = self._require_terminal(terminal_id)
        return {"terminal": record.snapshot().as_dict()}

    def list_by_run(self, run_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return [record.snapshot().as_dict() for record in self._terminals.values() if record.run_id == run_id]

    def _require_terminal(self, terminal_id: str) -> TerminalRecord:
        with self._lock:
            record = self._terminals.get(terminal_id)
        if not record:
            raise KeyError("Terminal not found")
        return record

    def write_terminal(self, terminal_id: str, data: str, source: TerminalOwner = "user") -> dict[str, Any]:
        record = self._require_terminal(terminal_id)
        payload = data or ""
        with record.lock:
            if not record.alive:
                raise RuntimeError("Terminal has already exited")
            if source == "user" and record.owner == "agent" and payload != "\u0003":
                raise PermissionError("Terminal is under agent control. Only Ctrl+C is allowed.")
            os.write(record.master_fd, payload.encode("utf-8"))
            record.updated_at = now_iso()
        return {"terminal": record.snapshot().as_dict(), "events": []}

    def interrupt_terminal(self, terminal_id: str, source: TerminalOwner = "user") -> dict[str, Any]:
        return self.write_terminal(terminal_id, "\u0003", source)

    def resize_terminal(self, terminal_id: str, cols: int, rows: int) -> dict[str, Any]:
        record = self._require_terminal(terminal_id)
        with record.lock:
            if not record.alive:
                raise RuntimeError("Terminal has already exited")
            if fcntl is not None and termios is not None:
                winsz = struct.pack("HHHH", max(10, min(rows, 120)), max(40, min(cols, 240)), 0, 0)
                fcntl.ioctl(record.master_fd, termios.TIOCSWINSZ, winsz)
            record.updated_at = now_iso()
        return {"terminal": record.snapshot().as_dict(), "events": []}

    def set_control(self, terminal_id: str, owner: TerminalOwner, reason: str | None = None) -> dict[str, Any]:
        record = self._require_terminal(terminal_id)
        with record.lock:
            record.owner = owner
        event = self._emit(
            record,
            {
                "type": "terminal_control_changed",
                "terminalId": terminal_id,
                "runId": record.run_id,
                "owner": owner,
                "reason": reason,
                "timestamp": now_iso(),
            },
        )
        return {"terminal": record.snapshot().as_dict(), "event": event, "events": [event]}

    def close_terminal(self, terminal_id: str, reason: str = "closed") -> dict[str, Any]:
        record = self._require_terminal(terminal_id)
        events: list[dict[str, Any]] = []
        with record.lock:
            if record.alive:
                try:
                    record.process.send_signal(signal.SIGTERM)
                except Exception as exc:
                    events.append(
                        self._emit(
                            record,
                            {
                                "type": "terminal_error",
                                "terminalId": terminal_id,
                                "runId": record.run_id,
                                "error": str(exc),
                                "timestamp": now_iso(),
                            },
                        )
                    )
                event = self._emit(
                    record,
                    {
                        "type": "terminal_control_changed",
                        "terminalId": terminal_id,
                        "runId": record.run_id,
                        "owner": "user",
                        "reason": reason,
                        "timestamp": now_iso(),
                    },
                )
                events.append(event)
        return {"terminal": record.snapshot().as_dict(), "events": events}

    def terminal_snapshot(self, terminal_id: str) -> dict[str, Any]:
        record = self._require_terminal(terminal_id)
        return {
            "terminalId": record.terminal_id,
            "owner": record.owner,
            "alive": record.alive,
            "cwd": record.cwd,
            "tail": record.tail[-8000:],
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
        with record.lock:
            while True:
                tail = record.tail
                if matcher and matcher(tail):
                    break
                if not record.alive or time.monotonic() >= deadline:
                    break
                record.condition.wait(timeout=max(0.05, deadline - time.monotonic()))
        return {
            "matched": bool(matcher(tail)) if matcher else False,
            "tail": tail,
            "terminal": record.snapshot().as_dict(),
        }

    def stream_subscription(self, terminal_id: str) -> tuple[queue.Queue[dict[str, Any]], list[dict[str, Any]], TerminalRecord]:
        record = self._require_terminal(terminal_id)
        subscription: queue.Queue[dict[str, Any]] = queue.Queue()
        with record.lock:
            record.subscribers.append(subscription)
            initial: list[dict[str, Any]] = []
            if record.tail:
                initial.append(
                    {
                        "type": "terminal_data",
                        "terminalId": record.terminal_id,
                        "runId": record.run_id,
                        "chunk": record.tail,
                        "stream": "pty",
                        "timestamp": now_iso(),
                    }
                )
            initial.append(
                {
                    "type": "terminal_control_changed",
                    "terminalId": record.terminal_id,
                    "runId": record.run_id,
                    "owner": record.owner,
                    "reason": "stream_attached",
                    "timestamp": now_iso(),
                }
            )
        return subscription, initial, record

    def remove_subscription(self, record: TerminalRecord, subscription: queue.Queue[dict[str, Any]]) -> None:
        with record.lock:
            if subscription in record.subscribers:
                record.subscribers.remove(subscription)


@lru_cache(maxsize=1)
def get_terminal_manager() -> TerminalManager:
    return TerminalManager()
