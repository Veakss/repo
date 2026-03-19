from __future__ import annotations

from pathlib import Path
from typing import Any

from streamlit_python_only.mcp.gateway.registry import GatewayRegistry


def register_file_tools(registry: GatewayRegistry, workspace_root: str) -> None:
    root = Path(workspace_root).resolve()

    def resolve_path(path_value: str) -> Path:
        target = root.joinpath(path_value).resolve()
        if target != root and root not in target.parents:
            raise ValueError("Path escapes workspace root")
        return target

    def read_file(arguments: dict[str, Any]) -> dict[str, Any]:
        path_value = str(arguments.get("path") or "")
        target = resolve_path(path_value)
        if not target.exists() or not target.is_file():
            return {"status": "error", "payload": {"error": f"File not found: {path_value}"}}
        return {"status": "ok", "payload": {"path": path_value, "content": target.read_text(encoding="utf-8")}, "evidence_kind": "file"}

    def list_directory(arguments: dict[str, Any]) -> dict[str, Any]:
        path_value = str(arguments.get("path") or ".")
        target = resolve_path(path_value)
        if not target.exists() or not target.is_dir():
            return {"status": "error", "payload": {"error": f"Directory not found: {path_value}"}}
        entries = sorted(item.name for item in target.iterdir())
        return {"status": "ok", "payload": {"path": path_value, "entries": entries}, "evidence_kind": "directory"}

    def write_file(arguments: dict[str, Any]) -> dict[str, Any]:
        path_value = str(arguments.get("path") or "")
        content = str(arguments.get("content") or "")
        target = resolve_path(path_value)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return {"status": "ok", "payload": {"path": path_value, "written": True}, "evidence_kind": "write"}

    registry.register("read_file", read_file)
    registry.register("list_directory", list_directory)
    registry.register("write_file", write_file)
