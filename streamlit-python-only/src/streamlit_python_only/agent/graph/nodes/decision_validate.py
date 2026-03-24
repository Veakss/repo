from __future__ import annotations

from streamlit_python_only.agent.graph.state import AgentGraphState


def decision_validate_node(runtime: object, state: AgentGraphState) -> AgentGraphState:
    return runtime._decision_validate_node(state)  # type: ignore[attr-defined]

