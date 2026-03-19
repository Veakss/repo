from __future__ import annotations

from typing import Any

from streamlit_python_only.mcp.gateway.registry import GatewayRegistry


def register_resource_tools(registry: GatewayRegistry) -> None:
    def open_url(arguments: dict[str, Any]) -> dict[str, Any]:
        url = str(arguments.get("url") or "")
        return {"status": "ok", "payload": {"url": url, "opened": False}, "evidence_kind": "app_action"}

    def open_resource(arguments: dict[str, Any]) -> dict[str, Any]:
        path = str(arguments.get("path") or "")
        return {"status": "ok", "payload": {"path": path, "opened": False}, "evidence_kind": "app_action"}

    registry.register("open_url", open_url)
    registry.register("open_resource", open_resource)
