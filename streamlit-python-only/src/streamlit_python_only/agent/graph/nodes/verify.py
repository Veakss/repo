from __future__ import annotations

from streamlit_python_only.agent.graph.state import AgentGraphState


def verify_node(runtime: object, state: AgentGraphState) -> AgentGraphState:
    return runtime._verify_node(state)  # noqa: SLF001
