from __future__ import annotations

from streamlit_python_only.rag import RagService
from streamlit_python_only.settings import get_settings
from streamlit_python_only.terminal_manager import TerminalManager, get_terminal_manager
from streamlit_python_only.tools.local import (
    build_control_tool_definitions,
    build_core_tool_definitions,
    build_retrieval_tool_definitions,
    build_terminal_tool_definitions,
)
from streamlit_python_only.tools.mcp.adapters import build_mcp_tool_definitions
from streamlit_python_only.tools.metadata import ToolMetadata


def build_tool_catalog(
    workspace_root: str,
    session_id: str | None = None,
    rag_service: RagService | None = None,
    run_id: str | None = None,
    terminal_manager: TerminalManager | None = None,
) -> list[ToolMetadata]:
    settings = get_settings()
    rag = rag_service or RagService()
    terminals = terminal_manager or get_terminal_manager()
    local_definitions: list[ToolMetadata] = []
    local_definitions.extend(build_core_tool_definitions(workspace_root))
    local_definitions.extend(build_control_tool_definitions())
    local_definitions.extend(build_terminal_tool_definitions(workspace_root, session_id, run_id, terminals))
    local_definitions.extend(build_retrieval_tool_definitions(session_id, rag))

    mcp_definitions = build_mcp_tool_definitions(settings, workspace_root=workspace_root)
    if not mcp_definitions:
        return local_definitions

    # MCP-first catalog: keep local definitions as controlled fallback.
    existing_names = {item.tool.name for item in mcp_definitions}
    fallback = [item for item in local_definitions if item.tool.name not in existing_names]
    return mcp_definitions + fallback
