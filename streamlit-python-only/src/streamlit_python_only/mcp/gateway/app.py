from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from streamlit_python_only.mcp.gateway.registry import GatewayRegistry
from streamlit_python_only.mcp.gateway.servers import (
    register_file_tools,
    register_resource_tools,
    register_terminal_tools,
    register_web_tools,
)

try:  # pragma: no cover - optional dependency path
    from fastmcp import FastMCP  # type: ignore
except Exception:  # pragma: no cover - optional dependency path
    FastMCP = None


class InvokeRequest(BaseModel):
    tool: str = Field(description="Tool name.")
    arguments: dict[str, Any] = Field(default_factory=dict)


def create_gateway_registry(workspace_root: str) -> GatewayRegistry:
    registry = GatewayRegistry()
    register_file_tools(registry, workspace_root=workspace_root)
    register_web_tools(registry)
    register_terminal_tools(registry)
    register_resource_tools(registry)
    return registry


def create_gateway_app(workspace_root: str) -> FastAPI:
    registry = create_gateway_registry(workspace_root)
    app = FastAPI(title="streamlit-python-only MCP Gateway")

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/invoke")
    def invoke(request: InvokeRequest) -> dict[str, Any]:
        return registry.invoke(request.tool, request.arguments)

    return app


def create_fastmcp_server(workspace_root: str):
    if FastMCP is None:
        return None
    registry = create_gateway_registry(workspace_root)
    server = FastMCP("streamlit-python-only-gateway")

    @server.tool
    def invoke_tool(tool: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        return registry.invoke(tool, arguments or {})

    return server
