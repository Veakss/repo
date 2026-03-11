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


class SessionCreateRequest(BaseModel):
    title: str | None = None


class SessionUpdateRequest(BaseModel):
    title: str
