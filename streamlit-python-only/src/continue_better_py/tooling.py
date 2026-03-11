from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import shlex
import subprocess
import uuid
from urllib.parse import quote_plus

import httpx
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


class TerminalInput(BaseModel):
    command: str = Field(description="Command to execute without shell chaining.")
    cwd: str | None = Field(default=None, description="Optional working directory relative to the workspace root.")
    timeout_sec: int = Field(default=20, ge=1, le=120, description="Maximum execution time in seconds.")


class OpenUrlInput(BaseModel):
    url: str = Field(description="HTTP or HTTPS URL to open.")


class OpenPathInput(BaseModel):
    path: str = Field(description="Path relative to the workspace root.")


class WebSearchInput(BaseModel):
    query: str = Field(description="Search query.")
    limit: int = Field(default=5, ge=1, le=10, description="Maximum number of search results to return.")


class SessionMemoryUpsertInput(BaseModel):
    content: str = Field(description="Important session memory to store.")
    kind: str = Field(default="preference", description="Short memory kind label.")


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
    rag = rag_service or RagService()

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
        profile_list = [item.strip() for item in (profiles or "").split(",") if item.strip()] or None
        return rag.lookup_for_model(
            question=question,
            session_id=session_id,
            scope={"profiles": profile_list} if profile_list else None,
        )

    def run_terminal(command: str, cwd: str | None = None, timeout_sec: int = 20) -> str:
        terminal_id = str(uuid.uuid4())
        requested = str(command or "").strip()
        blocked_tokens = ["&&", "||", ";", "|", ">", "<", "`", "$(", "sudo ", " rm ", " mv ", " chmod ", " chown "]
        if any(token in f" {requested} " for token in blocked_tokens) or requested.startswith(("rm ", "mv ", "chmod ", "chown ")):
            return json.dumps(
                {
                    "terminalId": terminal_id,
                    "command": requested,
                    "cwd": str(resolve_workspace_path(workspace_root, cwd or ".")),
                    "blocked": True,
                    "exitCode": 126,
                    "stdout": "",
                    "stderr": "Blocked by terminal policy: use a single safe command without chaining or destructive operations.",
                },
                ensure_ascii=False,
            )
        target_cwd = resolve_workspace_path(workspace_root, cwd or ".")
        try:
            args = shlex.split(requested)
        except ValueError as exc:
            return json.dumps(
                {
                    "terminalId": terminal_id,
                    "command": requested,
                    "cwd": str(target_cwd),
                    "blocked": True,
                    "exitCode": 126,
                    "stdout": "",
                    "stderr": f"Command parsing failed: {exc}",
                },
                ensure_ascii=False,
            )
        if not args:
            return json.dumps(
                {
                    "terminalId": terminal_id,
                    "command": requested,
                    "cwd": str(target_cwd),
                    "blocked": True,
                    "exitCode": 126,
                    "stdout": "",
                    "stderr": "No command provided.",
                },
                ensure_ascii=False,
            )
        try:
            completed = subprocess.run(
                args,
                cwd=target_cwd,
                capture_output=True,
                text=True,
                timeout=max(1, int(timeout_sec)),
                check=False,
            )
            stdout = completed.stdout[-8000:]
            stderr = completed.stderr[-4000:]
            return json.dumps(
                {
                    "terminalId": terminal_id,
                    "command": requested,
                    "cwd": str(target_cwd),
                    "blocked": False,
                    "exitCode": int(completed.returncode),
                    "stdout": stdout,
                    "stderr": stderr,
                },
                ensure_ascii=False,
            )
        except subprocess.TimeoutExpired:
            return json.dumps(
                {
                    "terminalId": terminal_id,
                    "command": requested,
                    "cwd": str(target_cwd),
                    "blocked": False,
                    "exitCode": 124,
                    "stdout": "",
                    "stderr": "Command timed out.",
                },
                ensure_ascii=False,
            )

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

    def web_search(query: str, limit: int = 5) -> str:
        search_url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"
        response = httpx.get(search_url, timeout=15.0, follow_redirects=True, headers={"User-Agent": "ContinueBetterPython/1.0"})
        response.raise_for_status()
        matches = []
        for url, title in re.findall(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', response.text, flags=re.IGNORECASE):
            clean_title = re.sub(r"<[^>]+>", "", title)
            matches.append(f"- {clean_title.strip()}: {url}")
            if len(matches) >= limit:
                break
        if not matches:
            return "No search results found."
        return "Web search results:\n" + "\n".join(matches)

    def session_memory_upsert(content: str, kind: str = "preference") -> str:
        if not session_id:
            return "Session memory is unavailable without a session context."
        state = rag.append_memory_note(session_id=session_id, content=content, kind=kind)
        return f"Stored session memory ({kind}). Entry count: {state['summary']['entryCount']}"

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
        ToolDefinition(
            tool=StructuredTool.from_function(
                func=run_terminal,
                name="run_terminal",
                description="Run a single safe terminal command inside the workspace root without shell chaining.",
                args_schema=TerminalInput,
            ),
            risk_level="safe",
            module_id="terminal",
        ),
        ToolDefinition(
            tool=StructuredTool.from_function(
                func=open_url,
                name="open_url",
                description="Open an external URL in the operating system browser.",
                args_schema=OpenUrlInput,
            ),
            risk_level="safe",
            module_id="app_actions",
        ),
        ToolDefinition(
            tool=StructuredTool.from_function(
                func=open_file,
                name="open_file",
                description="Open a local file in the operating system default app.",
                args_schema=OpenPathInput,
            ),
            risk_level="safe",
            module_id="app_actions",
        ),
        ToolDefinition(
            tool=StructuredTool.from_function(
                func=open_resource,
                name="open_resource",
                description="Open a local resource in the operating system default app.",
                args_schema=OpenPathInput,
            ),
            risk_level="safe",
            module_id="app_actions",
        ),
        ToolDefinition(
            tool=StructuredTool.from_function(
                func=web_search,
                name="web_search",
                description="Search the web and return compact source links.",
                args_schema=WebSearchInput,
            ),
            risk_level="safe",
            module_id="web",
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
        definitions.append(
            ToolDefinition(
                tool=StructuredTool.from_function(
                    func=session_memory_upsert,
                    name="session_memory_upsert",
                    description="Store an explicit fact or preference into the current session memory.",
                    args_schema=SessionMemoryUpsertInput,
                ),
                risk_level="safe",
                module_id="memory",
            )
        )
    return definitions


def build_tools(workspace_root: str, session_id: str | None = None) -> list[StructuredTool]:
    return [definition.tool for definition in build_tool_definitions(workspace_root, session_id=session_id)]


def build_tool_lookup(workspace_root: str, session_id: str | None = None) -> dict[str, ToolDefinition]:
    return {definition.tool.name: definition for definition in build_tool_definitions(workspace_root, session_id=session_id)}
