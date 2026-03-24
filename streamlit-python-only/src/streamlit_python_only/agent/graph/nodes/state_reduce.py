from __future__ import annotations

from streamlit_python_only.agent.graph.state import AgentGraphState


def state_reduce_node(runtime: object, state: AgentGraphState) -> AgentGraphState:
    return runtime._state_reduce_node(state)  # type: ignore[attr-defined]

