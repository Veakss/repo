from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict

from streamlit_python_only.settings import get_settings


def _state_root() -> Path:
    root = get_settings().resolved_artifact_root.joinpath("runtime_state")
    root.mkdir(parents=True, exist_ok=True)
    root.joinpath("approvals").mkdir(parents=True, exist_ok=True)
    root.joinpath("clarifications").mkdir(parents=True, exist_ok=True)
    return root


def serialize_messages(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    return [message_to_dict(message) for message in messages]


def deserialize_messages(payload: list[dict[str, Any]]) -> list[BaseMessage]:
    return messages_from_dict(payload)


class RunStateStore:
    def __init__(self) -> None:
        self.root = _state_root()

    def _path(self, kind: str, item_id: str) -> Path:
        return self.root.joinpath(kind, f"{item_id}.json")

    def save(self, kind: str, item_id: str, payload: dict[str, Any]) -> None:
        self._path(kind, item_id).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def pop(self, kind: str, item_id: str) -> dict[str, Any] | None:
        path = self._path(kind, item_id)
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        path.unlink(missing_ok=True)
        return payload
