from __future__ import annotations


def build_goal_gap_diagnostics(runtime: object, run_id: str, state: dict, reason: str) -> list[dict]:
    return runtime._goal_gap_events(run_id, state, reason)  # noqa: SLF001
