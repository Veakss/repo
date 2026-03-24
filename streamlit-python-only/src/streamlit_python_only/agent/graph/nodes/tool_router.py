from __future__ import annotations

from streamlit_python_only.agent.graph.state import AgentGraphState


def tool_router_node(runtime: object, state: AgentGraphState) -> AgentGraphState:
    return runtime._tool_router_node(state)  # type: ignore[attr-defined]
