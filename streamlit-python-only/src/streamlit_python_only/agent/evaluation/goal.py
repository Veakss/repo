from __future__ import annotations

from streamlit_python_only.graph_state import GoalAssessment


def evaluate_goal_completion(runtime: object, state: dict, final_text: str) -> GoalAssessment:
    return runtime._evaluate_goal_completion(state, final_text)  # noqa: SLF001
