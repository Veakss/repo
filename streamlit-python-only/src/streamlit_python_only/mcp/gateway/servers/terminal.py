from __future__ import annotations

import shlex
import subprocess
from typing import Any

from streamlit_python_only.mcp.gateway.registry import GatewayRegistry


def register_terminal_tools(registry: GatewayRegistry) -> None:
    def run_terminal(arguments: dict[str, Any]) -> dict[str, Any]:
        command = str(arguments.get("command") or "").strip()
        cwd = str(arguments.get("cwd") or ".").strip()
        timeout_sec = int(arguments.get("timeout_sec") or 20)
        if not command:
            return {"status": "error", "payload": {"error": "Missing command"}, "evidence_kind": "terminal"}
        if any(token in command for token in ["&&", "||", "|", ";", "$(", "`"]):
            return {"status": "error", "payload": {"error": "Blocked by terminal safety policy"}, "evidence_kind": "terminal"}
        try:
            args = shlex.split(command)
        except Exception as exc:
            return {"status": "error", "payload": {"error": f"Invalid command: {exc}"}, "evidence_kind": "terminal"}
        try:
            completed = subprocess.run(
                args,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=max(1, timeout_sec),
                check=False,
            )
        except Exception as exc:
            return {"status": "error", "payload": {"error": str(exc)}, "evidence_kind": "terminal"}
        output = "\n".join([str(completed.stdout or "").strip(), str(completed.stderr or "").strip()]).strip()
        return {
            "status": "ok" if int(completed.returncode) == 0 else "error",
            "payload": {
                "command": command,
                "cwd": cwd,
                "exitCode": int(completed.returncode),
                "output": output,
            },
            "evidence_kind": "terminal",
        }

    registry.register("run_terminal", run_terminal)
