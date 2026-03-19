from __future__ import annotations

import subprocess
from pathlib import Path

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from streamlit_python_only.tooling_shared import ToolMetadata, resolve_workspace_path


class ListDirectoryInput(BaseModel):
    path: str = Field(default=".", description="Path relative to the workspace root.")


class ReadFileInput(BaseModel):
    path: str = Field(description="Path relative to the workspace root.")


class WriteFileInput(BaseModel):
    path: str = Field(description="Path relative to the workspace root.")
    content: str = Field(description="UTF-8 file content to write.")


class OpenUrlInput(BaseModel):
    url: str = Field(description="HTTP or HTTPS URL to open.")


class OpenPathInput(BaseModel):
    path: str = Field(description="Path relative to the workspace root.")


def build_core_tool_definitions(workspace_root: str) -> list[ToolMetadata]:
    def list_directory(path: str = ".") -> str:
        target = resolve_workspace_path(workspace_root, path)
        if not target.exists():
            return f"Path not found: {path}"
        if target.is_file():
            return f"File: {path}"
        entries = []
        for item in sorted(target.iterdir(), key=lambda entry: (entry.is_file(), entry.name.lower())):
            kind = "file" if item.is_file() else "dir"
            entries.append(f"{kind}\t{item.relative_to(Path(workspace_root))}")
        return "\n".join(entries) or "(empty directory)"

    def read_file(path: str) -> str:
        target = resolve_workspace_path(workspace_root, path)
        if not target.exists() or not target.is_file():
            return f"File not found: {path}"
        return target.read_text(encoding="utf-8")

    def write_file(path: str, content: str) -> str:
        target = resolve_workspace_path(workspace_root, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Wrote {path}"

    def open_url(url: str) -> str:
        trimmed = str(url or "").strip()
        if not trimmed.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        try:
            subprocess.run(["open", trimmed], check=False, capture_output=True, text=True, timeout=10)
        except Exception:
            pass
        return f"Opened URL: {trimmed}"

    def open_file(path: str) -> str:
        target = resolve_workspace_path(workspace_root, path)
        if not target.exists():
            return f"File not found: {path}"
        try:
            subprocess.run(["open", str(target)], check=False, capture_output=True, text=True, timeout=10)
        except Exception:
            pass
        return f"Opened file: {path}"

    def open_resource(path: str) -> str:
        return open_file(path)

    return [
        ToolMetadata(
            tool=StructuredTool.from_function(list_directory, name="list_directory", description="List files and folders relative to the workspace root.", args_schema=ListDirectoryInput),
            risk_level="safe",
            module_id="files",
            supports_thales=True,
            supports_native_tools=True,
            supports_textual_replay=True,
            returns_evidence_kind="directory",
        ),
        ToolMetadata(
            tool=StructuredTool.from_function(read_file, name="read_file", description="Read a UTF-8 file from the workspace root.", args_schema=ReadFileInput),
            risk_level="safe",
            module_id="files",
            supports_thales=True,
            supports_native_tools=True,
            supports_textual_replay=True,
            returns_evidence_kind="file",
        ),
        ToolMetadata(
            tool=StructuredTool.from_function(write_file, name="write_file", description="Write a UTF-8 file inside the workspace root.", args_schema=WriteFileInput),
            risk_level="risky",
            module_id="files",
            supports_thales=True,
            supports_native_tools=True,
            supports_textual_replay=True,
            returns_evidence_kind="write",
        ),
        ToolMetadata(
            tool=StructuredTool.from_function(open_url, name="open_url", description="Open an external URL in the operating system browser.", args_schema=OpenUrlInput),
            risk_level="safe",
            module_id="app_actions",
            supports_thales=True,
            supports_native_tools=True,
            supports_textual_replay=True,
            returns_evidence_kind="app_action",
        ),
        ToolMetadata(
            tool=StructuredTool.from_function(open_file, name="open_file", description="Open a local file in the operating system default app.", args_schema=OpenPathInput),
            risk_level="safe",
            module_id="app_actions",
            supports_thales=True,
            supports_native_tools=True,
            supports_textual_replay=True,
            returns_evidence_kind="app_action",
        ),
        ToolMetadata(
            tool=StructuredTool.from_function(open_resource, name="open_resource", description="Open a local resource in the operating system default app.", args_schema=OpenPathInput),
            risk_level="safe",
            module_id="app_actions",
            supports_thales=True,
            supports_native_tools=True,
            supports_textual_replay=True,
            returns_evidence_kind="app_action",
        ),
    ]
