from __future__ import annotations

from streamlit_python_only.agent.graph.state import AgentGraphState


def repair_node(runtime: object, state: AgentGraphState) -> AgentGraphState:
    return runtime._repair_node(state)  # noqa: SLF001
