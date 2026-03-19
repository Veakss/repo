from __future__ import annotations

from streamlit_python_only.agent.graph.state import AgentGraphState


def preflight_node(runtime: object, state: AgentGraphState) -> AgentGraphState:
    return runtime._preflight_node(state)  # noqa: SLF001
