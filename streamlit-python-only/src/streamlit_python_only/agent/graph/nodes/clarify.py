from __future__ import annotations

from streamlit_python_only.agent.graph.state import AgentGraphState


def clarify_gate_node(runtime: object, state: AgentGraphState) -> AgentGraphState:
    return runtime._clarify_gate_node(state)  # type: ignore[attr-defined]


def clarify_node(runtime: object, state: AgentGraphState) -> AgentGraphState:
    return runtime._clarify_node(state)  # type: ignore[attr-defined]
