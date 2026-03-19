from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from langchain_core.tools import StructuredTool


@dataclass(slots=True)
class ToolMetadata:
    tool: StructuredTool
    risk_level: str
    module_id: str
    supports_thales: bool
    supports_native_tools: bool
    supports_textual_replay: bool
    returns_evidence_kind: str


def resolve_workspace_path(workspace_root: str, relative_path: str) -> Path:
    root = Path(workspace_root).resolve()
    target = root.joinpath(relative_path).resolve()
    if target != root and root not in target.parents:
        raise ValueError("Path escapes workspace root")
    return target
