from __future__ import annotations

from streamlit_python_only.agent.graph.state import AgentGraphState


def agent_node(runtime: object, state: AgentGraphState) -> AgentGraphState:
    return runtime._agent_node(state)  # type: ignore[attr-defined]
