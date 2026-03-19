from __future__ import annotations

from pathlib import Path

from streamlit_python_only.settings import Settings
from streamlit_python_only.tools.catalog import build_tool_catalog


def test_tool_catalog_returns_local_tools_when_mcp_disabled(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("streamlit_python_only.tools.catalog.get_settings", lambda: Settings(mcp_mode="disabled"))
    definitions = build_tool_catalog(str(tmp_path), session_id="s1", run_id="r1")
    names = [item.tool.name for item in definitions]
    assert "read_file" in names
    assert "run_terminal" in names
