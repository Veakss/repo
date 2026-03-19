from streamlit_python_only.agent.graph.policies.stop_policy import should_continue_after_verify


def test_stop_policy_continues_only_for_agent_action() -> None:
    assert should_continue_after_verify("agent") is True
    assert should_continue_after_verify("finish") is False
    assert should_continue_after_verify(None) is False


def test_stop_policy_respects_pending_blockers() -> None:
    assert should_continue_after_verify("agent", pending={"type": "approval"}) is False
