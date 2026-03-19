from __future__ import annotations

import json
import httpx
from langchain_core.tools import StructuredTool

from streamlit_python_only.rag import RagService
from streamlit_python_only.terminal_manager import TerminalManager, get_terminal_manager
from streamlit_python_only.tooling_shared import ToolMetadata as ToolDefinition
from streamlit_python_only.tools.catalog import build_tool_catalog


def build_tool_definitions(
    workspace_root: str,
    session_id: str | None = None,
    rag_service: RagService | None = None,
    run_id: str | None = None,
    terminal_manager: TerminalManager | None = None,
) -> list[ToolDefinition]:
    rag = rag_service or RagService()
    terminals = terminal_manager or get_terminal_manager()
    return build_tool_catalog(workspace_root, session_id=session_id, rag_service=rag, run_id=run_id, terminal_manager=terminals)


def build_tools(workspace_root: str, session_id: str | None = None, run_id: str | None = None, terminal_manager: TerminalManager | None = None) -> list[StructuredTool]:
    return [definition.tool for definition in build_tool_definitions(workspace_root, session_id=session_id, run_id=run_id, terminal_manager=terminal_manager)]


def build_tool_lookup(
    workspace_root: str,
    session_id: str | None = None,
    run_id: str | None = None,
    terminal_manager: TerminalManager | None = None,
) -> dict[str, ToolDefinition]:
    return {
        definition.tool.name: definition
        for definition in build_tool_definitions(workspace_root, session_id=session_id, run_id=run_id, terminal_manager=terminal_manager)
    }
