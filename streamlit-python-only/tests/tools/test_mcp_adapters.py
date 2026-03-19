from __future__ import annotations

from streamlit_python_only.settings import Settings
from streamlit_python_only.tools.mcp.adapters import build_mcp_tool_definitions


def test_build_mcp_tool_definitions_disabled() -> None:
    settings = Settings(mcp_mode="disabled")
    assert build_mcp_tool_definitions(settings, workspace_root=".") == []


def test_build_mcp_tool_definitions_remote_includes_tools() -> None:
    settings = Settings(
        mcp_mode="remote",
        mcp_gateway_url="http://localhost:4010",
        mcp_enable_files=True,
        mcp_enable_web=True,
    )
    definitions = build_mcp_tool_definitions(settings, workspace_root=".")
    names = [item.tool.name for item in definitions]
    assert "read_file" in names
    assert "web_search" in names
