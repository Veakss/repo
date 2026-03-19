from __future__ import annotations

from typing import Any


def should_continue_after_verify(verify_action: str | None, pending: dict[str, Any] | None = None) -> bool:
    if pending:
        return False
    return str(verify_action or "").strip().lower() in {"agent"}
