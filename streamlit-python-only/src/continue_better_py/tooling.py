from __future__ import annotations

from pathlib import Path
from typing import Callable

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class ListDirectoryInput(BaseModel):
    path: str = Field(default=".", description="Path relative to the workspace root.")


class ReadFileInput(BaseModel):
    path: str = Field(description="Path relative to the workspace root.")


class WriteFileInput(BaseModel):
    path: str = Field(description="Path relative to the workspace root.")
    content: str = Field(description="UTF-8 file content to write.")


def resolve_workspace_path(workspace_root: str, relative_path: str) -> Path:
    root = Path(workspace_root).resolve()
    target = root.joinpath(relative_path).resolve()
    if target != root and root not in target.parents:
        raise ValueError("Path escapes workspace root")
    return target


def build_tools(workspace_root: str) -> list[StructuredTool]:
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

    return [
        StructuredTool.from_function(
            func=list_directory,
            name="list_directory",
            description="List files and folders relative to the workspace root.",
            args_schema=ListDirectoryInput,
        ),
        StructuredTool.from_function(
            func=read_file,
            name="read_file",
            description="Read a UTF-8 file from the workspace root.",
            args_schema=ReadFileInput,
        ),
        StructuredTool.from_function(
            func=write_file,
            name="write_file",
            description="Write a UTF-8 file inside the workspace root.",
            args_schema=WriteFileInput,
        ),
    ]
