from __future__ import annotations

from typing import Any


def normalize_mcp_result(result: Any, evidence_kind: str = "mcp") -> dict[str, Any]:
    if isinstance(result, dict):
        payload = dict(result)
        return {
            "status": str(payload.get("status") or "ok"),
            "evidence_kind": str(payload.get("evidence_kind") or evidence_kind),
            "payload": payload.get("payload", payload),
            "citations": list(payload.get("citations") or []),
            "diagnostics": list(payload.get("diagnostics") or []),
        }
    return {
        "status": "ok",
        "evidence_kind": evidence_kind,
        "payload": result,
        "citations": [],
        "diagnostics": [],
    }
