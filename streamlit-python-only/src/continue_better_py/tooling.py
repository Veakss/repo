from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from continue_better_py.rag import RagService


class ListDirectoryInput(BaseModel):
    path: str = Field(default=".", description="Path relative to the workspace root.")


class ReadFileInput(BaseModel):
    path: str = Field(description="Path relative to the workspace root.")


class WriteFileInput(BaseModel):
    path: str = Field(description="Path relative to the workspace root.")
    content: str = Field(description="UTF-8 file content to write.")


class ClarificationInput(BaseModel):
    question: str = Field(description="Clarification question for the user.")
    option_a: str | None = Field(default=None, description="First suggested answer.")
    option_b: str | None = Field(default=None, description="Second suggested answer.")
    option_c: str | None = Field(default=None, description="Third suggested answer.")


class RagLookupInput(BaseModel):
    question: str = Field(description="The question to answer using the active RAG sources.")
    profiles: str | None = Field(default=None, description="Optional comma-separated profile names to restrict profile retrieval.")


@dataclass(slots=True)
class ToolDefinition:
    tool: StructuredTool
    risk_level: str
    module_id: str


def resolve_workspace_path(workspace_root: str, relative_path: str) -> Path:
    root = Path(workspace_root).resolve()
    target = root.joinpath(relative_path).resolve()
    if target != root and root not in target.parents:
        raise ValueError("Path escapes workspace root")
    return target


def build_tool_definitions(workspace_root: str, session_id: str | None = None, rag_service: RagService | None = None) -> list[ToolDefinition]:
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

    def request_clarification(
        question: str,
        option_a: str | None = None,
        option_b: str | None = None,
        option_c: str | None = None,
    ) -> str:
        options = [value for value in [option_a, option_b, option_c] if value]
        if options:
            rendered = "\n".join(f"- {item}" for item in options)
            return f"{question}\n{rendered}"
        return question

    def rag_lookup(question: str, profiles: str | None = None) -> str:
        rag = rag_service or RagService()
        profile_list = [item.strip() for item in (profiles or "").split(",") if item.strip()] or None
        return rag.lookup_for_model(
            question=question,
            session_id=session_id,
            scope={"profiles": profile_list} if profile_list else None,
        )

    definitions = [
        ToolDefinition(
            tool=StructuredTool.from_function(
                func=list_directory,
                name="list_directory",
                description="List files and folders relative to the workspace root.",
                args_schema=ListDirectoryInput,
            ),
            risk_level="safe",
            module_id="files",
        ),
        ToolDefinition(
            tool=StructuredTool.from_function(
                func=read_file,
                name="read_file",
                description="Read a UTF-8 file from the workspace root.",
                args_schema=ReadFileInput,
            ),
            risk_level="safe",
            module_id="files",
        ),
        ToolDefinition(
            tool=StructuredTool.from_function(
                func=write_file,
                name="write_file",
                description="Write a UTF-8 file inside the workspace root.",
                args_schema=WriteFileInput,
            ),
            risk_level="risky",
            module_id="files",
        ),
        ToolDefinition(
            tool=StructuredTool.from_function(
                func=request_clarification,
                name="request_clarification",
                description="Ask the user for clarification when the request is ambiguous or risky.",
                args_schema=ClarificationInput,
            ),
            risk_level="safe",
            module_id="clarification",
        ),
    ]
    if session_id:
        definitions.append(
            ToolDefinition(
                tool=StructuredTool.from_function(
                    func=rag_lookup,
                    name="rag_lookup",
                    description="Search Session Docs, Profiles, and Session Memory for relevant context with citations and transparency.",
                    args_schema=RagLookupInput,
                ),
                risk_level="safe",
                module_id="rag",
            )
        )
    return definitions


def build_tools(workspace_root: str, session_id: str | None = None) -> list[StructuredTool]:
    return [definition.tool for definition in build_tool_definitions(workspace_root, session_id=session_id)]


def build_tool_lookup(workspace_root: str, session_id: str | None = None) -> dict[str, ToolDefinition]:
    return {definition.tool.name: definition for definition in build_tool_definitions(workspace_root, session_id=session_id)}
