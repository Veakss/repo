from __future__ import annotations

from streamlit_python_only.mcp.gateway.registry import GatewayRegistry
from streamlit_python_only.mcp.gateway.servers.files import register_file_tools


def test_file_server_read_and_list(tmp_path) -> None:
    registry = GatewayRegistry()
    register_file_tools(registry, str(tmp_path))
    target = tmp_path / "a.txt"
    target.write_text("hello", encoding="utf-8")
    listed = registry.invoke("list_directory", {"path": "."})
    assert listed["status"] == "ok"
    read = registry.invoke("read_file", {"path": "a.txt"})
    assert read["status"] == "ok"
    assert read["payload"]["content"] == "hello"
