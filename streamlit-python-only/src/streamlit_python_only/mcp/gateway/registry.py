from __future__ import annotations

from typing import Any, Callable

ToolHandler = Callable[[dict[str, Any]], dict[str, Any]]


class GatewayRegistry:
    def __init__(self) -> None:
        self._handlers: dict[str, ToolHandler] = {}

    def register(self, tool_name: str, handler: ToolHandler) -> None:
        self._handlers[tool_name] = handler

    def invoke(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        handler = self._handlers.get(tool_name)
        if handler is None:
            return {"status": "error", "payload": {"error": f"Unknown tool: {tool_name}"}}
        try:
            return handler(arguments)
        except Exception as exc:  # pragma: no cover - defensive path
            return {"status": "error", "payload": {"error": str(exc)}}
