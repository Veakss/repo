from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ProviderPolicy:
    name: str
    max_tool_calls_per_turn: int
    requires_textual_replay: bool
    sequential_only: bool


def resolve_provider_policy(capabilities: dict[str, Any] | None) -> ProviderPolicy:
    payload = capabilities or {}
    requires_sequential = bool(payload.get("requiresSequentialToolLoop"))
    supports_textual_replay = bool(payload.get("supportsTextualReplay"))
    max_tool_calls = int(payload.get("maxToolCallsPerTurn", 1 if requires_sequential else 1) or 1)
    if requires_sequential:
        return ProviderPolicy(
            name="thales_sequential" if supports_textual_replay else "sequential_default",
            max_tool_calls_per_turn=1,
            requires_textual_replay=supports_textual_replay,
            sequential_only=True,
        )
    # v1 default remains one tool per cycle even when native multi-tool is available.
    return ProviderPolicy(
        name="native_default",
        max_tool_calls_per_turn=1,
        requires_textual_replay=False,
        sequential_only=False,
    )
