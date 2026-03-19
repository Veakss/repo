from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from streamlit_python_only.mcp_tools import build_mcp_tool_definition
from streamlit_python_only.settings import Settings
from streamlit_python_only.mcp.gateway.app import create_gateway_registry
from streamlit_python_only.tools.mcp.client import McpClient
from streamlit_python_only.tools.mcp.normalizer import normalize_mcp_result
from streamlit_python_only.tools.metadata import ToolMetadata


class ReadFileInput(BaseModel):
    path: str = Field(description="Path relative to workspace root.")


class ListDirectoryInput(BaseModel):
    path: str = Field(default=".", description="Directory path relative to workspace root.")


class WriteFileInput(BaseModel):
    path: str = Field(description="Path relative to workspace root.")
    content: str = Field(description="UTF-8 file content.")


class OpenUrlInput(BaseModel):
    url: str = Field(description="HTTP(S) URL to open.")


class OpenPathInput(BaseModel):
    path: str = Field(description="Path relative to workspace root.")


class WebSearchInput(BaseModel):
    query: str = Field(description="Search query.")
    limit: int = Field(default=5, ge=1, le=10, description="Maximum number of results.")


class RunTerminalInput(BaseModel):
    command: str = Field(description="Command to execute.")
    cwd: str | None = Field(default=".", description="Working directory.")
    timeout_sec: int = Field(default=20, ge=1, le=120, description="Execution timeout in seconds.")


def build_mcp_tool_definitions(settings: Settings, workspace_root: str) -> list[ToolMetadata]:
    if str(settings.mcp_mode).lower() == "disabled":
        return []
    mode = str(settings.mcp_mode).lower()
    registry = create_gateway_registry(workspace_root) if mode == "embedded" else None
    client = McpClient(settings.mcp_gateway_url, timeout_ms=settings.mcp_timeout_ms, retry_count=settings.mcp_retry_count) if mode == "remote" else None

    def call_mcp(tool_name: str, payload: dict[str, Any]) -> str:
        if registry is not None:
            raw = registry.invoke(tool_name, payload)
        elif client is not None:
            raw = client.invoke_tool(tool_name, payload)
        else:
            raw = {"status": "error", "payload": {"error": f"Unsupported MCP mode: {mode}"}}
        normalized = normalize_mcp_result(raw)
        return json.dumps(normalized, ensure_ascii=False)

    definitions: list[ToolMetadata] = []
    if settings.mcp_enable_files:
        definitions.append(
            build_mcp_tool_definition(
                tool_name="read_file",
                description="Read file content through MCP gateway.",
                args_schema=ReadFileInput,
                handler=lambda path: call_mcp("read_file", {"path": path}),
                module_id="files",
                risk_level="safe",
                supports_thales=True,
                supports_native_tools=True,
                supports_textual_replay=True,
                returns_evidence_kind="file",
            )
        )
        definitions.append(
            build_mcp_tool_definition(
                tool_name="list_directory",
                description="List directory entries through MCP gateway.",
                args_schema=ListDirectoryInput,
                handler=lambda path=".": call_mcp("list_directory", {"path": path}),
                module_id="files",
                risk_level="safe",
                supports_thales=True,
                supports_native_tools=True,
                supports_textual_replay=True,
                returns_evidence_kind="directory",
            )
        )
        definitions.append(
            build_mcp_tool_definition(
                tool_name="write_file",
                description="Write file content through MCP gateway.",
                args_schema=WriteFileInput,
                handler=lambda path, content: call_mcp("write_file", {"path": path, "content": content}),
                module_id="files",
                risk_level="risky",
                supports_thales=True,
                supports_native_tools=True,
                supports_textual_replay=True,
                returns_evidence_kind="write",
            )
        )
    if settings.mcp_enable_web:
        definitions.append(
            build_mcp_tool_definition(
                tool_name="web_search",
                description="Search web through MCP gateway.",
                args_schema=WebSearchInput,
                handler=lambda query, limit=5: call_mcp("web_search", {"query": query, "limit": limit}),
                module_id="web",
                risk_level="safe",
                supports_thales=True,
                supports_native_tools=True,
                supports_textual_replay=True,
                returns_evidence_kind="web",
            )
        )
    if settings.mcp_enable_resources:
        definitions.append(
            build_mcp_tool_definition(
                tool_name="open_url",
                description="Open URL through MCP gateway.",
                args_schema=OpenUrlInput,
                handler=lambda url: call_mcp("open_url", {"url": url}),
                module_id="app_actions",
                risk_level="safe",
                supports_thales=True,
                supports_native_tools=True,
                supports_textual_replay=True,
                returns_evidence_kind="app_action",
            )
        )
        definitions.append(
            build_mcp_tool_definition(
                tool_name="open_file",
                description="Open file through MCP gateway resources handler.",
                args_schema=OpenPathInput,
                handler=lambda path: call_mcp("open_resource", {"path": path}),
                module_id="app_actions",
                risk_level="safe",
                supports_thales=True,
                supports_native_tools=True,
                supports_textual_replay=True,
                returns_evidence_kind="app_action",
            )
        )
        definitions.append(
            build_mcp_tool_definition(
                tool_name="open_resource",
                description="Open resource through MCP gateway resources handler.",
                args_schema=OpenPathInput,
                handler=lambda path: call_mcp("open_resource", {"path": path}),
                module_id="app_actions",
                risk_level="safe",
                supports_thales=True,
                supports_native_tools=True,
                supports_textual_replay=True,
                returns_evidence_kind="app_action",
            )
        )
    if settings.mcp_enable_terminal:
        definitions.append(
            build_mcp_tool_definition(
                tool_name="run_terminal",
                description="Run terminal command through MCP gateway.",
                args_schema=RunTerminalInput,
                handler=lambda command, cwd=".", timeout_sec=20: call_mcp(
                    "run_terminal",
                    {"command": command, "cwd": cwd, "timeout_sec": timeout_sec},
                ),
                module_id="terminal",
                risk_level="safe",
                supports_thales=True,
                supports_native_tools=True,
                supports_textual_replay=True,
                returns_evidence_kind="terminal",
            )
        )
    return definitions
