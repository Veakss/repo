from __future__ import annotations

from streamlit_python_only.agent.graph.state import AgentGraphState


def tool_executor_node(runtime: object, state: AgentGraphState) -> AgentGraphState:
    return runtime._tools_node(state)  # noqa: SLF001
