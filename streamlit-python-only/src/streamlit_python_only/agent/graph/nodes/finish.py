from __future__ import annotations

from streamlit_python_only.agent.graph.state import AgentGraphState


def finish_node(runtime: object, state: AgentGraphState) -> AgentGraphState:
    return runtime._finish_node(state)  # noqa: SLF001
