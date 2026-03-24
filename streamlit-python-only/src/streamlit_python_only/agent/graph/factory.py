from __future__ import annotations

from langgraph.graph import END, StateGraph

from streamlit_python_only.agent.graph.nodes import (
    agent_node,
    clarify_gate_node,
    clarify_node,
    decision_validate_node,
    fail_node,
    finish_node,
    preflight_node,
    repair_node,
    state_reduce_node,
    state_update_node,
    tool_executor_node,
    tool_router_node,
    verify_node,
)
from streamlit_python_only.agent.graph.state import AgentGraphState


def create_graph(runtime: object):
    def route_after_clarify_gate(state: AgentGraphState) -> str:
        return "clarify" if state.get("graph_route") == "clarify" else "agent"

    def route_after_decision_validate(state: AgentGraphState) -> str:
        route = state.get("graph_route")
        if route == "fail":
            return "fail"
        return "tool_router"

    def route_after_tool_router(state: AgentGraphState) -> str:
        return "tool_executor" if state.get("graph_route") == "tool_executor" else "verify"

    def route_after_state_reduce(state: AgentGraphState) -> str:
        route = state.get("graph_route")
        if route == "fail":
            return "fail"
        if route == "finish":
            return "finish"
        if route == "verify":
            return "verify"
        return "agent"

    def route_after_verify(state: AgentGraphState) -> str:
        route = state.get("graph_route")
        if route == "fail":
            return "fail"
        if route == "repair":
            return "repair"
        if route == "clarify":
            return "clarify"
        if route == "agent":
            return "agent"
        return "finish"

    graph = StateGraph(AgentGraphState)
    graph.add_node("preflight", lambda s: preflight_node(runtime, s))
    graph.add_node("clarify_gate", lambda s: clarify_gate_node(runtime, s))
    graph.add_node("clarify", lambda s: clarify_node(runtime, s))
    graph.add_node("agent", lambda s: agent_node(runtime, s))
    graph.add_node("decision_validate", lambda s: decision_validate_node(runtime, s))
    graph.add_node("tool_router", lambda s: tool_router_node(runtime, s))
    graph.add_node("tool_executor", lambda s: tool_executor_node(runtime, s))
    graph.add_node("state_reduce", lambda s: state_reduce_node(runtime, s))
    graph.add_node("state_update", lambda s: state_update_node(runtime, s))
    graph.add_node("verify", lambda s: verify_node(runtime, s))
    graph.add_node("repair", lambda s: repair_node(runtime, s))
    graph.add_node("finish", lambda s: finish_node(runtime, s))
    graph.add_node("fail", lambda s: fail_node(runtime, s))
    graph.set_entry_point("preflight")
    graph.add_edge("preflight", "clarify_gate")
    graph.add_conditional_edges("clarify_gate", route_after_clarify_gate, {"clarify": "clarify", "agent": "agent"})
    graph.add_edge("agent", "decision_validate")
    graph.add_conditional_edges("decision_validate", route_after_decision_validate, {"tool_router": "tool_router", "fail": "fail"})
    graph.add_conditional_edges("tool_router", route_after_tool_router, {"tool_executor": "tool_executor", "verify": "verify"})
    graph.add_edge("tool_executor", "state_reduce")
    graph.add_conditional_edges("state_reduce", route_after_state_reduce, {"agent": "agent", "verify": "verify", "finish": "finish", "fail": "fail"})
    graph.add_conditional_edges("verify", route_after_verify, {"repair": "repair", "clarify": "clarify", "agent": "agent", "finish": "finish", "fail": "fail"})
    graph.add_edge("repair", "verify")
    graph.add_edge("clarify", END)
    graph.add_edge("finish", END)
    graph.add_edge("fail", END)
    return graph.compile()
