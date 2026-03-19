from __future__ import annotations

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatStreamRequest(BaseModel):
    session_id: str = Field(alias="session_id")
    message: str
    workspace_root: str | None = None
    allow_writes: bool = False
    policy_profile: str = "ask_when_necessary"
    model: str | None = None
    run_id: str | None = None
    profile: str | None = None
    tool_toggles: dict[str, bool] | None = None
    force_tool_use: str | None = None


class SidecarChatRequest(BaseModel):
    sessionId: str
    messages: list[ChatMessage]
    workspaceRoot: str | None = None
    allowWrites: bool = False
    policyProfile: str = "ask_when_necessary"
    model: str | None = None
    runId: str | None = None
    profile: str | None = None
    toolToggles: dict[str, bool] | None = None
    forceToolUse: str | None = None


class ApprovalDecisionRequest(BaseModel):
    approval_id: str
    decision: str


class ClarificationDecisionRequest(BaseModel):
    clarification_id: str
    answer: str


class TerminalCreateRequest(BaseModel):
    session_id: str | None = None
    run_id: str | None = None
    workspace_root: str | None = None
    cwd: str | None = None
    shell: str | None = None
    cols: int | None = None
    rows: int | None = None
    owner: str = "user"


class TerminalWriteRequest(BaseModel):
    data: str
    source: str = "user"


class TerminalResizeRequest(BaseModel):
    cols: int
    rows: int


class TerminalControlRequest(BaseModel):
    owner: str
    reason: str | None = None


class TerminalInterruptRequest(BaseModel):
    source: str = "user"


class TerminalWaitRequest(BaseModel):
    pattern: str
    timeout_ms: int = 30_000
    regex: bool = False


class TerminalResolveRequest(BaseModel):
    terminal_id: str | None = None
    run_id: str | None = None
    session_id: str | None = None
    require_alive: bool = True


class SessionCreateRequest(BaseModel):
    title: str | None = None


class SessionUpdateRequest(BaseModel):
    title: str


class RagProfileCreateRequest(BaseModel):
    name: str


class RagProfileRenameRequest(BaseModel):
    new_name: str


class RagImportRequest(BaseModel):
    path: str


class RagSessionMemoryPatchRequest(BaseModel):
    enabled: bool | None = None
    threshold_pct: float | None = None
    token_budget: int | None = None


class RagLookupScope(BaseModel):
    source_priority: list[str] | None = None
    session_docs: bool | None = None
    session_memory: bool | None = None
    profiles: list[str] | None = None


class RagLookupRequest(BaseModel):
    question: str
    session_id: str | None = None
    scope: RagLookupScope | None = None


class RagMemoryAppendRequest(BaseModel):
    prompt: str
    answer: str


class MatrixRunRequest(BaseModel):
    models: list[str] | None = None
    scenarios: list[str] | None = None
    scenario_ids: list[str] | None = None
    variant_groups: list[str] | None = None
    categories: list[str] | None = None
    repeat: int = 1
    profiles: list[str] | None = None
    surfaces: list[str] | None = None
    all_profiles: bool = False
    all_surfaces: bool = False
