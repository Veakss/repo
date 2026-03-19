from streamlit_python_only.mcp.gateway.servers.files import register_file_tools
from streamlit_python_only.mcp.gateway.servers.resources import register_resource_tools
from streamlit_python_only.mcp.gateway.servers.terminal import register_terminal_tools
from streamlit_python_only.mcp.gateway.servers.web import register_web_tools

__all__ = [
    "register_file_tools",
    "register_web_tools",
    "register_terminal_tools",
    "register_resource_tools",
]
