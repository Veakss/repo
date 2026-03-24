from __future__ import annotations

from streamlit_python_only.agent.graph.state import AgentGraphState


def fail_node(runtime: object, state: AgentGraphState) -> AgentGraphState:
    return runtime._fail_node(state)  # type: ignore[attr-defined]

