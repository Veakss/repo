from __future__ import annotations

from langchain_core.messages import AIMessage

from streamlit_python_only.agent.graph.factory import create_graph


class FakeRuntime:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def _preflight_node(self, state):
        self.calls.append("preflight")
        return {**state, "graph_route": None}

    def _clarify_gate_node(self, state):
        self.calls.append("clarify_gate")
        return {**state, "graph_route": "agent"}

    def _clarify_node(self, state):
        self.calls.append("clarify")
        return {**state, "graph_route": "finish"}

    def _agent_node(self, state):
        self.calls.append("agent")
        return {**state, "messages": [AIMessage(content="done")], "graph_route": None}

    def _tool_router_node(self, state):
        self.calls.append("tool_router")
        return {**state, "graph_route": "verify"}

    def _tools_node(self, state):
        self.calls.append("tool_executor")
        return state

    def _state_update_node(self, state):
        self.calls.append("state_update")
        return state

    def _verify_node(self, state):
        self.calls.append("verify")
        return {**state, "graph_route": "finish"}

    def _repair_node(self, state):
        self.calls.append("repair")
        return state

    def _finish_node(self, state):
        self.calls.append("finish")
        return state


def test_graph_reaches_finish_through_verify() -> None:
    runtime = FakeRuntime()
    graph = create_graph(runtime)
    state = {
        "messages": [],
        "graph_route": None,
    }
    graph.invoke(state)
    assert runtime.calls == ["preflight", "clarify_gate", "agent", "tool_router", "verify", "finish"]
