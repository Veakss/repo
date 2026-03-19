from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypedDict

from langgraph.graph.message import add_messages
from typing_extensions import Annotated


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    session_id: str
    run_id: str
    workspace_root: str
    profile: str | None
    model: str | None
    provider_mode: str
    policy_profile: str
    tool_toggles: dict[str, bool] | None
    force_tool_use: str | None
    final_text: str
    pending_approval: dict[str, Any] | None
    pending_clarification: dict[str, Any] | None
    tool_events: list[dict[str, Any]]
    provider_capabilities: dict[str, Any]
    final_contract: dict[str, Any]
    multi_step_contract: dict[str, Any]
    goal_summary: dict[str, Any]
    realization_summary: dict[str, Any]
    goal_gap_summary: dict[str, Any]
    step_progress: dict[str, Any]
    evidence_map: dict[str, Any]
    tool_results: list[dict[str, Any]]
    used_tool_names: list[str]
    executed_tools: list[str]
    recent_tool_signatures: list[str]
    repeated_tool_signature_streak: dict[str, Any] | None
    no_progress_turns: int
    repair_attempts: int
    stale_final_retry_count: int
    last_read_file_path: str | None
    last_read_file_content: str | None
    last_opened_url: str | None
    last_opened_file_path: str | None
    written_files: list[str]
    web_search_last_result_count: int
    web_search_last_result_urls: list[str]
    rag_lookup_last_status: str | None
    rag_lookup_last_hit_count: int | None
    rag_lookup_last_hits: list[dict[str, Any]]
    rag_lookup_last_structured: bool
    final_repair_attempted: bool
    multistep_repair_attempted: bool
    multistep_no_progress_turns: int
    verify_action: str | None
    graph_route: str | None
    conversation: dict[str, Any]
    runtime: dict[str, Any]
    goal: dict[str, Any]
    evidence: dict[str, Any]
    decision: dict[str, Any]
    control: dict[str, Any]
    output: dict[str, Any]


@dataclass(slots=True)
class FinalAnswerContract:
    requested_language: str | None = None
    exact_output_text: str | None = None
    require_sources: bool = False
    require_grounding: bool = False
    require_grounding_from_rag: bool = False
    require_citations_when_rag_used: bool = False

    def to_payload(self) -> dict[str, Any]:
        return {
            "requestedLanguage": self.requested_language,
            "exactOutputText": self.exact_output_text,
            "requireSources": self.require_sources,
            "requireGrounding": self.require_grounding,
            "requireGroundingFromRag": self.require_grounding_from_rag,
            "requireCitationsWhenRagUsed": self.require_citations_when_rag_used,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any] | None) -> "FinalAnswerContract":
        payload = payload or {}
        return cls(
            requested_language=payload.get("requestedLanguage"),
            exact_output_text=payload.get("exactOutputText"),
            require_sources=bool(payload.get("requireSources")),
            require_grounding=bool(payload.get("requireGrounding")),
            require_grounding_from_rag=bool(payload.get("requireGroundingFromRag")),
            require_citations_when_rag_used=bool(payload.get("requireCitationsWhenRagUsed")),
        )


@dataclass(slots=True)
class MultiStepContract:
    requested_artifacts: list[str] = field(default_factory=list)
    required_inputs: list[dict[str, Any]] = field(default_factory=list)
    required_answer_fields: list[str] = field(default_factory=list)
    expected_step_count: int = 0

    def to_payload(self) -> dict[str, Any]:
        return {
            "requestedArtifacts": list(self.requested_artifacts),
            "requiredInputs": [dict(item) for item in self.required_inputs],
            "requiredAnswerFields": list(self.required_answer_fields),
            "expectedStepCount": int(self.expected_step_count),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any] | None) -> "MultiStepContract":
        payload = payload or {}
        required_inputs = payload.get("requiredInputs") if isinstance(payload.get("requiredInputs"), list) else []
        return cls(
            requested_artifacts=[str(item) for item in payload.get("requestedArtifacts", []) if str(item).strip()],
            required_inputs=[dict(item) for item in required_inputs if isinstance(item, dict)],
            required_answer_fields=[str(item) for item in payload.get("requiredAnswerFields", []) if str(item).strip()],
            expected_step_count=max(0, int(payload.get("expectedStepCount") or 0)),
        )


@dataclass(slots=True)
class GoalState:
    user_intent: str
    required_evidence_kinds: list[str] = field(default_factory=list)
    required_facts: list[str] = field(default_factory=list)
    required_citations: bool = False
    required_output_constraint: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "userIntent": self.user_intent,
            "requiredEvidenceKinds": list(self.required_evidence_kinds),
            "requiredFacts": list(self.required_facts),
            "requiredCitations": bool(self.required_citations),
            "requiredOutputConstraint": dict(self.required_output_constraint),
        }


@dataclass(slots=True)
class RealizationState:
    available_evidence: dict[str, Any] = field(default_factory=dict)
    proven_facts: dict[str, Any] = field(default_factory=dict)
    candidate_answer: str = ""

    def to_payload(self) -> dict[str, Any]:
        return {
            "availableEvidence": dict(self.available_evidence),
            "provenFacts": dict(self.proven_facts),
            "candidateAnswer": self.candidate_answer,
        }


@dataclass(slots=True)
class GoalGap:
    missing_evidence: list[str] = field(default_factory=list)
    missing_facts: list[str] = field(default_factory=list)
    answer_defects: list[str] = field(default_factory=list)
    can_gather_more_evidence: bool = False
    is_complete: bool = False

    def to_payload(self) -> dict[str, Any]:
        return {
            "missingEvidence": list(self.missing_evidence),
            "missingFacts": list(self.missing_facts),
            "answerDefects": list(self.answer_defects),
            "canGatherMoreEvidence": bool(self.can_gather_more_evidence),
            "isComplete": bool(self.is_complete),
        }


@dataclass(slots=True)
class GoalAssessment:
    action: str
    mismatch_codes: list[tuple[str, str]] = field(default_factory=list)
    goal: GoalState = field(default_factory=lambda: GoalState(user_intent=""))
    realization: RealizationState = field(default_factory=RealizationState)
    gap: GoalGap = field(default_factory=GoalGap)
