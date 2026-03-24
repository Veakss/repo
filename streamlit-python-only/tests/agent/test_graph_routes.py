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

    def _decision_validate_node(self, state):
        self.calls.append("decision_validate")
        return {**state, "graph_route": "tool_router"}

    def _tool_router_node(self, state):
        self.calls.append("tool_router")
        return {**state, "graph_route": "verify"}

    def _tools_node(self, state):
        self.calls.append("tool_executor")
        return state

    def _state_reduce_node(self, state):
        self.calls.append("state_reduce")
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

    def _fail_node(self, state):
        self.calls.append("fail")
        return state


def test_graph_reaches_finish_through_verify() -> None:
    runtime = FakeRuntime()
    graph = create_graph(runtime)
    state = {
        "messages": [],
        "graph_route": None,
    }
    graph.invoke(state)
    assert runtime.calls == ["preflight", "clarify_gate", "agent", "decision_validate", "tool_router", "verify", "finish"]


def test_graph_passes_through_state_reduce_after_tool_execution() -> None:
    runtime = FakeRuntime()

    def route_tool(state):
        runtime.calls.append("tool_router")
        return {**state, "graph_route": "tool_executor"}

    def verify_after_reduce(state):
        runtime.calls.append("state_reduce")
        return {**state, "graph_route": "verify"}

    runtime._tool_router_node = route_tool
    runtime._state_reduce_node = verify_after_reduce
    graph = create_graph(runtime)
    state = {
        "messages": [],
        "graph_route": None,
    }
    graph.invoke(state)
    assert runtime.calls == ["preflight", "clarify_gate", "agent", "decision_validate", "tool_router", "tool_executor", "state_reduce", "verify", "finish"]


def test_graph_can_route_to_fail_after_decision_validate() -> None:
    runtime = FakeRuntime()

    def fail_validate(state):
        runtime.calls.append("decision_validate")
        return {**state, "graph_route": "fail"}

    runtime._decision_validate_node = fail_validate
    graph = create_graph(runtime)
    state = {
        "messages": [],
        "graph_route": None,
    }
    graph.invoke(state)
    assert runtime.calls == ["preflight", "clarify_gate", "agent", "decision_validate", "fail"]
