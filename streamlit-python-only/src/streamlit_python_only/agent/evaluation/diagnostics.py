from __future__ import annotations


def build_goal_gap_diagnostics(owner: object, run_id: str, state: dict, reason: str) -> list[dict]:
    engine = getattr(owner, "evaluation", owner)
    return engine._goal_gap_events(run_id, state, reason)  # noqa: SLF001
