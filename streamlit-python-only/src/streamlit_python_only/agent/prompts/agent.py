def build_agent_prompt() -> str:
    return (
        "Choose the next best action only. "
        "If verified evidence is sufficient, answer directly. "
        "If evidence is missing, call one useful tool."
    )
