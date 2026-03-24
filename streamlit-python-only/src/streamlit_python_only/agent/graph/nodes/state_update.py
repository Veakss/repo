from __future__ import annotations

from streamlit_python_only.agent.graph.state import AgentGraphState


def state_update_node(runtime: object, state: AgentGraphState) -> AgentGraphState:
    return runtime._state_update_node(state)  # type: ignore[attr-defined]
