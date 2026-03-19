from __future__ import annotations

import json
from typing import Any

import httpx


class McpClient:
    def __init__(self, base_url: str, timeout_ms: int = 10_000, retry_count: int = 1) -> None:
        self.base_url = str(base_url).rstrip("/")
        self.timeout = max(100, int(timeout_ms)) / 1000.0
        self.retry_count = max(0, int(retry_count))

    def invoke_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        payload = {"tool": tool_name, "arguments": dict(arguments or {})}
        last_error: Exception | None = None
        for _attempt in range(self.retry_count + 1):
            try:
                response = httpx.post(
                    f"{self.base_url}/invoke",
                    json=payload,
                    timeout=self.timeout,
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()
                data = response.json()
                if isinstance(data, dict):
                    return data
                return {"status": "ok", "payload": data}
            except Exception as exc:  # pragma: no cover - network failures are environment dependent
                last_error = exc
        message = str(last_error) if last_error else "Unknown MCP client error"
        return {"status": "error", "payload": {"error": message}, "diagnostics": [message]}

    @staticmethod
    def serialize_result(result: dict[str, Any]) -> str:
        return json.dumps(result, ensure_ascii=False)
