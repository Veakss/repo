from streamlit_python_only.agent.graph.policies.provider_policy import resolve_provider_policy


def test_provider_policy_thales_sequential() -> None:
    policy = resolve_provider_policy(
        {
            "requiresSequentialToolLoop": True,
            "supportsTextualReplay": True,
            "maxToolCallsPerTurn": 1,
        }
    )
    assert policy.name == "thales_sequential"
    assert policy.sequential_only is True
    assert policy.max_tool_calls_per_turn == 1


def test_provider_policy_native_defaults_to_single_step() -> None:
    policy = resolve_provider_policy(
        {
            "requiresSequentialToolLoop": False,
            "supportsTextualReplay": False,
            "maxToolCallsPerTurn": 8,
        }
    )
    assert policy.name == "native_default"
    assert policy.sequential_only is False
    assert policy.max_tool_calls_per_turn == 1
