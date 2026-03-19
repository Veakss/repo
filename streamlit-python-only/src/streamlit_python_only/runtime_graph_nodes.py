from __future__ import annotations

from typing import Any


def build_runtime_nodes(engine: Any) -> dict[str, Any]:
    return {
        "preflight": engine._preflight_node,
        "clarify_gate": engine._clarify_gate_node,
        "clarify": engine._clarify_node,
        "agent": engine._agent_node,
        "tool_router": engine._tool_router_node,
        "tool_executor": engine._tools_node,
        "state_update": engine._state_update_node,
        "verify": engine._verify_node,
        "repair": engine._repair_node,
        "finish": engine._finish_node,
        "parse_text_tool_calls": engine._parse_text_tool_calls,
    }
