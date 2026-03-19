from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from streamlit_python_only.events import (
    approval_decision,
    approval_required,
    clarification_required,
    done,
    error_event,
    run_phase,
    run_diagnostic,
    run_state,
    terminal_error,
    terminal_exit,
    terminal_opened,
    token,
)
from streamlit_python_only.graph_factory import create_runtime_graph
from streamlit_python_only.graph_state import (
    AgentState,
    FinalAnswerContract,
    GoalAssessment,
    GoalGap,
    GoalState,
    MultiStepContract,
    RealizationState,
)
from streamlit_python_only.providers import ProviderCapabilities, ResolvedProvider, build_chat_model, resolve_provider
from streamlit_python_only.rag import RagService, tokenize
from streamlit_python_only.runtime_graph_orchestration import RuntimeGraphOrchestrationMixin
from streamlit_python_only.run_state import RunStateStore, deserialize_messages, serialize_messages
from streamlit_python_only.schemas import ApprovalDecisionRequest, ClarificationDecisionRequest, SidecarChatRequest
from streamlit_python_only.terminal_manager import TerminalManager
from streamlit_python_only.tool_registry import ToolRegistry, create_default_tool_registry

@dataclass(slots=True)
class RuntimeDependencies:
    model_factory: Callable[[str | None, str | None], Any]
    provider_resolver: Callable[[str | None, str | None], ResolvedProvider]
    tool_registry_factory: Callable[[str, str | None, str | None, dict[str, bool] | None], ToolRegistry]
    state_store: RunStateStore
    terminal_manager: TerminalManager = field(default_factory=TerminalManager)
    rag_service: RagService | None = None

def split_for_streaming(text: str) -> list[str]:
    parts = text.split()
    if not parts:
        return [text]
    return [f"{part} " for part in parts]


URL_PATTERN = re.compile(r"https?://[^\s)>]+", flags=re.IGNORECASE)
FRENCH_HINT_PATTERN = re.compile(r"\b(le|la|les|des|une|un|bonjour|merci|avec|sans|pour|dans|sur|est|et|ou|que|qui)\b", flags=re.IGNORECASE)
ENGLISH_HINT_PATTERN = re.compile(r"\b(the|and|with|without|for|from|this|that|latest|news|please|hello|thanks|what|which)\b", flags=re.IGNORECASE)
EXACT_REPLY_PATTERNS = (
    re.compile(r"(?:say|answer|reply|respond|return)\s+with\s+exactly\s+[`'\"]?(.+?)(?:[`'\"]|$)", flags=re.IGNORECASE),
    re.compile(r"(?:say|answer|reply|respond|return)\s+exactly\s+[`'\"]?(.+?)(?:[`'\"]|$)", flags=re.IGNORECASE),
    re.compile(r"(?:say|answer|reply|respond|return)\s+only\s+[`'\"]?(.+?)(?:[`'\"]|$)", flags=re.IGNORECASE),
)
TEXT_VALUE_PATTERN = re.compile(r"with\s+the\s+text\s+[`'\"]?([^`'\",\n]+)[`'\"]?", flags=re.IGNORECASE)
EXACT_OUTPUT_TRAILING_QUALIFIERS = (
    re.compile(r"\s+(?:and|with)\s+nothing\s+else[.!]?\s*$", flags=re.IGNORECASE),
    re.compile(r"\s+with\s+no\s+extra\s+(?:words?|text|explanation)[.!]?\s*$", flags=re.IGNORECASE),
    re.compile(r"\s+without\s+(?:any\s+)?extra\s+(?:words?|text|explanation)[.!]?\s*$", flags=re.IGNORECASE),
    re.compile(r"\s+and\s+no\s+extra\s+(?:words?|text|explanation)[.!]?\s*$", flags=re.IGNORECASE),
)
LANGUAGE_OVERRIDE_PATTERNS = (
    (re.compile(r"\b(?:reply|respond|answer)\s+in\s+english\b", flags=re.IGNORECASE), "en"),
    (re.compile(r"\b(?:reply|respond|answer)\s+in\s+french\b", flags=re.IGNORECASE), "fr"),
    (re.compile(r"\b(?:réponds|reponds|répondre|repondre)\s+en\s+fran[cç]ais\b", flags=re.IGNORECASE), "fr"),
    (re.compile(r"\b(?:réponds|reponds|répondre|repondre)\s+en\s+anglais\b", flags=re.IGNORECASE), "en"),
)
COMMON_GROUNDING_STOPWORDS = {
    "command",
    "output",
    "owner",
    "resolution",
    "matched",
    "tool",
    "terminal",
    "result",
    "true",
    "false",
    "none",
    "file",
    "files",
    "directory",
    "directories",
    "session",
    "memory",
    "query",
    "sources",
    "source",
    "score",
    "kind",
    "entry",
    "entries",
    "current",
    "workspace",
}
RAG_SALIENCE_STOPWORDS = COMMON_GROUNDING_STOPWORDS.union(
    {
        "phase",
        "status",
        "checkpoint",
        "current",
        "recorded",
        "indexed",
        "document",
        "documents",
        "brief",
        "briefly",
        "answer",
        "citation",
        "citations",
        "compact",
        "using",
        "session",
        "sessions",
        "rag",
        "what",
        "which",
        "with",
        "very",
    }
)
FOLLOWUP_SUMMARY_PATTERN = re.compile(r"\b(summar(?:y|ize|ise)|one\s+line|brief(?:ly)?|short(?:ly)?)\b", flags=re.IGNORECASE)
FOLLOWUP_REFERENCE_PATTERN = re.compile(r"\b(it|that|this|above|previous|now|again)\b", flags=re.IGNORECASE)
GROUNDING_TOOL_NAMES = {
    "run_terminal",
    "terminal_wait_for_output",
    "terminal_snapshot",
    "rag_lookup",
    "read_file",
    "list_directory",
    "open_file",
    "open_resource",
}
PATH_LIKE_PATTERN = re.compile(r"(?:^|[\s`\"'])([A-Za-z0-9_./-]+\.[A-Za-z0-9]{1,8})(?=$|[\s`\"',:;!?])")
DIRECTORY_PHRASE_PATTERN = re.compile(r"\b(?:inspect|list|check|explore)\s+(?:the\s+)?([A-Za-z0-9_./-]+)\s+directory\b", flags=re.IGNORECASE)
SEQUENTIAL_CUE_PATTERN = re.compile(r"\b(?:then|now|after that|and now|next)\b", flags=re.IGNORECASE)
MAX_NO_PROGRESS_TURNS = 2
MAX_REPAIR_ATTEMPTS = 3
FAILURE_CODE_SET = {
    "missing_evidence",
    "verification_failed",
    "tool_failed",
    "approval_blocked",
    "clarification_blocked",
    "no_progress_limit",
    "invalid_final_answer",
    "runtime_exception",
}


def parse_text_tool_calls(text: str) -> list[dict[str, Any]]:
    trimmed = str(text or "").strip()
    if not trimmed:
        return []
    candidates = [line.strip() for line in trimmed.splitlines() if line.strip()] if "\n" in trimmed else [trimmed]
    parsed: list[dict[str, Any]] = []
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except Exception:
            continue
        if isinstance(payload, dict) and isinstance(payload.get("name"), str) and isinstance(payload.get("arguments"), dict):
            parsed.append({"id": str(uuid.uuid4()), "name": payload["name"], "args": payload["arguments"]})
    return parsed


def normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize_exact_output(value: str) -> str:
    return normalize_whitespace(value).strip("`\"' ")


def clean_exact_output_candidate(prompt: str, candidate: str) -> str:
    cleaned = normalize_exact_output(candidate)
    for pattern in EXACT_OUTPUT_TRAILING_QUALIFIERS:
        cleaned = pattern.sub("", cleaned).strip()
    if not cleaned:
        return ""
    if cleaned[-1:] in {".", "!", "?"}:
        bare = cleaned[:-1].strip()
        if bare:
            escaped = re.escape(bare)
            occurrences = len(re.findall(rf"(?<![A-Za-z0-9_:-]){escaped}(?![A-Za-z0-9_:-])", prompt))
            if occurrences >= 2:
                cleaned = bare
    return cleaned


def extract_urls(text: str) -> list[str]:
    return URL_PATTERN.findall(str(text or ""))


def detect_message_language(text: str) -> str | None:
    lowered = str(text or "").strip().lower()
    if not lowered:
        return None
    if re.search(r"[àâçéèêëîïôûùüÿœ]", lowered):
        return "fr"
    french_score = len(FRENCH_HINT_PATTERN.findall(lowered))
    english_score = len(ENGLISH_HINT_PATTERN.findall(lowered))
    if french_score > english_score and french_score > 0:
        return "fr"
    if english_score > french_score and english_score > 0:
        return "en"
    return None


def detect_requested_language(prompt: str) -> str | None:
    text = str(prompt or "")
    for pattern, language in LANGUAGE_OVERRIDE_PATTERNS:
        if pattern.search(text):
            return language
    return detect_message_language(text)


def detect_exact_output_target(prompt: str) -> str | None:
    text = str(prompt or "").strip()
    for pattern in EXACT_REPLY_PATTERNS:
        match = pattern.search(text)
        if match:
            candidate = clean_exact_output_candidate(text, match.group(1))
            if candidate:
                return candidate
    lowered = text.lower()
    if "exact content only" in lowered or "exactly the content only" in lowered:
        match = TEXT_VALUE_PATTERN.search(text)
        if match:
            candidate = normalize_exact_output(match.group(1))
            if candidate:
                return candidate
    return None


def build_final_answer_contract(user_prompt: str, forced_mode: str | None = None) -> FinalAnswerContract:
    lowered = str(user_prompt or "").lower()
    rag_keywords = ("rag", "session doc", "session docs", "document", "documents", "indexed document", "local docs")
    citation_keywords = ("source", "sources", "citation", "citations", "cite", "citer")
    require_grounding = bool(
        re.search(r"\b(summarize|summary|explain|inspect|answer|remind|brief|briefly|short|one\s+\w*\s*line)\b", lowered)
    )
    return FinalAnswerContract(
        requested_language=detect_requested_language(user_prompt),
        exact_output_text=detect_exact_output_target(user_prompt),
        require_sources=(
            forced_mode == "web"
            or any(keyword in lowered for keyword in ("latest", "news", "today"))
        ),
        require_grounding=require_grounding,
        require_grounding_from_rag=(forced_mode == "rag" or any(keyword in lowered for keyword in rag_keywords)),
        require_citations_when_rag_used=(forced_mode == "rag" or any(keyword in lowered for keyword in citation_keywords)),
    )


def describe_language(language: str | None) -> str:
    return {"fr": "French", "en": "English"}.get(str(language or "").lower(), "user language")


def summarize_final_contract(contract: FinalAnswerContract) -> str:
    parts: list[str] = []
    if contract.requested_language:
        parts.append(f"language={describe_language(contract.requested_language)}")
    if contract.exact_output_text:
        parts.append(f"exact_output={contract.exact_output_text}")
    if contract.require_sources:
        parts.append("sources>=2_urls")
    if contract.require_grounding:
        parts.append("grounded_summary")
    if contract.require_grounding_from_rag:
        parts.append("rag_grounded")
    if contract.require_citations_when_rag_used:
        parts.append("rag_citations")
    return ", ".join(parts) if parts else "default"


def build_contract_detected_diagnostic(run_id: str, contract: FinalAnswerContract) -> dict[str, Any]:
    return run_diagnostic(run_id, "final_contract_detected", f"Final answer contract detected: {summarize_final_contract(contract)}.")


def _dedupe_strings(values: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = normalize_whitespace(str(value or ""))
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        output.append(cleaned)
    return output


def _extract_prompt_paths(prompt: str) -> list[str]:
    values = [match.group(1).strip() for match in PATH_LIKE_PATTERN.finditer(str(prompt or ""))]
    return _dedupe_strings(values)


def _extract_prompt_directories(prompt: str) -> list[str]:
    text = str(prompt or "")
    directories = [match.group(1).strip().rstrip("/") for match in DIRECTORY_PHRASE_PATTERN.finditer(text)]
    if re.search(r"\b(?:inspect|list|check|explore)\s+the\s+directory\b", text, flags=re.IGNORECASE):
        directories.append(".")
    normalized_directories: list[str] = []
    for directory in directories:
        lowered = directory.lower()
        if lowered in {"the", "current", "here", "directory"}:
            normalized_directories.append(".")
        else:
            normalized_directories.append(directory)
    lowered = text.lower()
    if "workspace root" in lowered or "workspace" in lowered:
        normalized_directories.append(".")
    return _dedupe_strings(normalized_directories)


def _extract_required_answer_fields(prompt: str) -> list[str]:
    prompt_text = str(prompt or "")
    lowered = PATH_LIKE_PATTERN.sub(" ", prompt_text).lower()
    fields: list[str] = []
    if "checkpoint" in lowered:
        fields.append("checkpoint")
    if "product name" in lowered or "product" in lowered:
        fields.append("product_name")
    if "stack" in lowered:
        fields.append("stack")
    if "preferred editor" in lowered or "editor" in lowered:
        fields.append("preferred_editor")
    if "status" in lowered:
        fields.append("status")
    if "file that exists" in lowered or "fixture file" in lowered or "extra file" in lowered:
        fields.append("extra_file")
    return _dedupe_strings(fields)


def build_multi_step_contract(user_prompt: str) -> MultiStepContract:
    prompt = str(user_prompt or "")
    required_inputs: list[dict[str, Any]] = []
    for path in _extract_prompt_paths(prompt):
        required_inputs.append({"kind": "file", "target": path})
    for path in _extract_prompt_directories(prompt):
        required_inputs.append({"kind": "directory", "target": path})
    required_answer_fields = _extract_required_answer_fields(prompt)
    has_sequential_cue = bool(SEQUENTIAL_CUE_PATTERN.search(prompt))
    input_count = len(required_inputs)
    expected_step_count = 0
    if input_count >= 3:
        expected_step_count = input_count
    elif input_count >= 2 and (has_sequential_cue or bool(required_answer_fields)):
        expected_step_count = input_count
    elif input_count == 1 and has_sequential_cue and bool(required_answer_fields):
        expected_step_count = 2
    return MultiStepContract(
        requested_artifacts=_extract_prompt_paths(prompt),
        required_inputs=required_inputs,
        required_answer_fields=required_answer_fields,
        expected_step_count=expected_step_count,
    )


def summarize_multi_step_contract(contract: MultiStepContract) -> str:
    parts: list[str] = []
    if contract.required_inputs:
        parts.append(
            "inputs="
            + ", ".join(f"{item.get('kind')}:{item.get('target')}" for item in contract.required_inputs[:6])
        )
    if contract.required_answer_fields:
        parts.append("fields=" + ", ".join(contract.required_answer_fields))
    if contract.expected_step_count:
        parts.append(f"expected_steps={contract.expected_step_count}")
    return ", ".join(parts) if parts else "default"


def build_multistep_contract_diagnostic(run_id: str, contract: MultiStepContract) -> dict[str, Any]:
    return run_diagnostic(run_id, "multistep_contract_detected", f"Multi-step contract detected: {summarize_multi_step_contract(contract)}.")


def is_multistep_contract_active(contract: MultiStepContract) -> bool:
    return int(contract.expected_step_count or 0) > 1


def provider_capabilities_payload(provider: Any) -> dict[str, Any]:
    capabilities = getattr(provider, "capabilities", None)
    if isinstance(capabilities, ProviderCapabilities):
        return capabilities.to_payload()
    mode = str(getattr(provider, "mode", "native") or "native").strip().lower()
    provider_name = str(getattr(provider, "provider", "") or "").strip().lower()
    textual_replay = mode == "textual_replay"
    max_tool_calls_per_turn = 1 if textual_replay else 8
    return {
        "supportsNativeTools": mode in {"native", "textual_replay"},
        "supportsTextualReplay": textual_replay,
        "supportsMultiToolTurn": max_tool_calls_per_turn > 1,
        "maxToolCallsPerTurn": max_tool_calls_per_turn,
        "requiresSequentialToolLoop": max_tool_calls_per_turn <= 1,
        "configSource": "env",
        "providerFamily": provider_name or ("thales" if textual_replay else "openrouter"),
    }


def build_tool_replay_message(name: str, args: dict[str, Any], result: str, status: str) -> SystemMessage:
    return SystemMessage(
        content=(
            "Tool execution replay for provider compatibility.\n"
            f"Tool name: {name}\n"
            f"Arguments JSON: {json.dumps(args, ensure_ascii=False)}\n"
            f"Tool status: {status}\n"
            "Tool result:\n"
            f"{result}\n"
            "Continue the task using this result and do not rely on OpenAI tool-role replay."
        )
    )


def build_clarification_resume_message(clarification: dict[str, Any] | None, answer: str) -> SystemMessage:
    details = clarification or {}
    question = str(details.get("question") or "Could you clarify?").strip() or "Could you clarify?"
    lines = [
        "The clarification has been answered. Continue the task using this resolved scope.",
        f"Original clarification question: {question}",
        f"User clarification answer: {answer.strip() or '(empty answer)'}",
        "Do not ask the same clarification again unless the new answer is still genuinely insufficient.",
    ]
    options = details.get("options") or []
    option_labels = [
        str(option.get("label") or "").strip()
        for option in options
        if isinstance(option, dict) and str(option.get("label") or "").strip()
    ]
    if option_labels:
        lines.append("Original options: " + ", ".join(option_labels))
    return SystemMessage(content="\n".join(lines))


def forced_tool_instruction(mode: str | None) -> str | None:
    if mode == "rag":
        return "The user explicitly requested RAG mode for this run. You must use rag_lookup before answering. If RAG is unavailable or returns no useful result, explain that clearly."
    if mode == "web":
        return "The user explicitly requested Web mode for this run. You must use web_search before answering. If web access is unavailable or yields no useful result, explain that clearly."
    if mode == "apps":
        return "The user explicitly requested Apps mode for this run. Prefer open_file, open_url, or open_resource when relevant."
    if mode == "clarification":
        return "The user explicitly requested clarification mode for this run. Use request_clarification before proceeding unless the request is already fully specified."
    return None


def forced_tool_group(mode: str | None) -> str | None:
    return {
        "rag": "rag",
        "web": "web",
        "apps": "app_actions",
        "clarification": "clarification",
    }.get(str(mode or "").strip().lower())


def build_runtime_system_prompt(
    registry: ToolRegistry,
    selected_group: str | None = None,
    provider_capabilities: dict[str, Any] | None = None,
) -> str:
    tool_names = registry.enabled_tool_names()
    available_tools = ", ".join(tool_names) if tool_names else "(none)"
    provider_capabilities = provider_capabilities or {}
    prompt_parts = [
        "You are a coding assistant running inside a local tool-enabled runtime.",
        "You know general coding knowledge and reasoning without tools.",
        "Use tools whenever they help complete the task better, safer, or with more accuracy.",
        "Reply in the same language as the user unless they ask otherwise.",
        "Choose the tool that matches the user's exact intent. Opening a file or URL is different from reading it; reading is different from editing; local knowledge is different from current external facts.",
        "Distinguish between actions executed in the current run and events recalled from earlier conversation or retrieved memory.",
        f"Available tools for this turn: {available_tools}.",
        (
            f"Current active parent tool group: {selected_group}. Prefer tools from this group first; if they are insufficient, move to the best remaining tool only when the current goal gap is still open."
            if selected_group
            else "No parent tool group is pre-selected for this run."
        ),
        "Map the user's intent to the most direct tool: open/show a URL or file -> open_url/open_file; read file contents -> read_file; create or edit files -> write_file; current external facts -> web_search; local grounded retrieval -> rag_lookup; shell or repo inspection -> run_terminal.",
        "Prefer a dedicated tool over terminal improvisation when a dedicated tool already fits the request.",
        "For multi-step requests, continue only while verified evidence is still missing. Stop immediately once the goal gap is closed, even if that happens after only two steps.",
        "Do not add extra steps just because the task feels broad. If the current evidence is sufficient, answer now.",
        "If the user explicitly asks for an interactive terminal workflow, prefer open_terminal, terminal_write, terminal_wait_for_output, and terminal_close over run_terminal.",
        "For terminal work, prefer one command at a time. Do not use shell chaining or substitution operators. Inspect output, then decide the next command.",
        "CRITICAL: If you are asked to provide an exact output format (like TERMINAL_SINGLE_SHOT_OK:<basename>), you MUST execute the tool first, read its result, and THEN provide the final answer in the requested format. Do NOT guess or echo the command.",
        "Avoid interactive editors such as vim or nano. Use write_file instead for code changes.",
        "Before asking for clarification about project structure or file locations, use list_directory or run_terminal to explore the workspace.",
        "CRITICAL: If you need clarification from the user, you MUST use the `request_clarification` tool. DO NOT ask questions in plain text. DO NOT output phrases like 'I need clarification'. ONLY call the tool.",
        "When using request_clarification, ALWAYS provide 2-3 likely options using the option_a, option_b, and option_c arguments to help the user choose quickly.",
        "When the user asks to remember or store a session fact, use session_memory_upsert.",
        "When web_search is used, keep the answer body concise and place citations in a separate Sources section using Markdown links like [Source title](https://...).",
        "When rag_lookup returns hits, include a Sources section with path:line citations.",
        "Never return raw tool-call JSON as a final answer.",
        "Never claim an action was completed unless the corresponding tool call actually succeeded.",
        "Once you have enough evidence, stop tool use and provide the final answer.",
    ]
    if bool(provider_capabilities.get("requiresSequentialToolLoop")):
        prompt_parts.extend(
            [
                "Provider constraint: emit at most one tool call per assistant turn.",
                "Do not plan multiple tool calls in a single response.",
                "After each tool result, either produce the final answer or emit exactly one next tool call if the goal gap is still open.",
            ]
        )
    return " ".join(part for part in prompt_parts if part)


def normalize_final_markdown(text: str) -> str:
    cleaned = str(text or "").replace("(empty response)", "").strip()
    if not cleaned:
        return ""
    split = re.split(r"(?i)\bSources?\s*:\s*", cleaned, maxsplit=1)
    if len(split) != 2:
        return cleaned
    body, raw_sources = split[0].strip(), split[1].strip()
    if not raw_sources:
        return body
    if re.search(r"(?m)^\s*[-*]\s+", raw_sources):
        source_items = [line.strip() for line in raw_sources.splitlines() if line.strip()]
    else:
        parts = [part.strip() for part in re.split(r"\s+-\s+", raw_sources) if part.strip()]
        source_items = [f"- {part}" if not part.startswith(("-", "*")) else part for part in parts]
    compact_items: list[str] = []
    for item in source_items:
        token = item.strip()
        if not token:
            continue
        if token.startswith("* "):
            token = f"- {token[2:].strip()}"
        if not token.startswith("- "):
            token = f"- {token}"
        compact_items.append(token)
    if not compact_items:
        return body
    return f"{body}\n\nSources:\n" + "\n".join(compact_items)


def should_request_approval(policy_profile: str, risk_level: str) -> bool:
    normalized_policy = str(policy_profile or "ask_when_necessary").strip().lower()
    normalized_risk = str(risk_level or "risky").strip().lower()
    if normalized_policy == "always_allow":
        return False
    if normalized_policy == "always_ask":
        return True
    return normalized_risk != "safe"


def build_early_clarification(last_user_message: str, clarification_enabled: bool) -> dict[str, Any] | None:
    if not clarification_enabled:
        return None

    text = str(last_user_message or "").strip()
    lowered = text.lower()
    if not lowered:
        return None

    if re.fullmatch(r"(?:a\s+)?mobile\s+app[.!]?", lowered):
        return {
            "reason": "mobile_platform_unspecified",
            "questions": [
                "Which mobile platform do you want exactly: iPhone, Android, or cross-platform?"
            ],
            "options": [
                {"label": "iPhone"},
                {"label": "Android"},
                {"label": "Cross-platform"},
            ],
        }

    broad_app_triggers = (
        "build me an app",
        "create an app",
        "make an app",
        "create a food application",
        "help me create a food application",
        "mobile app",
        "web app",
    )
    if any(trigger in lowered for trigger in broad_app_triggers):
        return {
            "reason": "broad_product_prompt",
            "questions": [
                "What kind of app do you want exactly: mobile app, web app, or something else?"
            ],
            "options": [
                {"label": "Web app"},
                {"label": "Mobile app"},
                {"label": "Something else"},
            ],
        }

    if "fix a bug" in lowered and "project" in lowered:
        return {
            "reason": "bugfix_needs_symptom",
            "questions": [
                "Which file or component is affected by the bug?"
            ],
            "options": [
                {"label": "frontend/app.py"},
                {"label": "backend"},
                {"label": "I need help locating it"},
            ],
        }

    return None


def decode_terminal_payload(raw: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(str(raw or ""))
    except Exception:
        return None
    if not isinstance(payload, dict) or "terminalId" not in payload:
        return None
    return payload


def decode_terminal_tool_payload(raw: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(str(raw or ""))
    except Exception:
        return None
    if not isinstance(payload, dict) or payload.get("kind") != "terminal_tool":
        return None
    return payload


def decode_rag_tool_payload(raw: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(str(raw or ""))
    except Exception:
        return None
    if not isinstance(payload, dict) or payload.get("kind") != "rag_tool":
        return None
    lookup = payload.get("lookup")
    rendered = payload.get("rendered")
    if not isinstance(lookup, dict) or not isinstance(rendered, str):
        return None
    return payload


TERMINAL_TOOLS_REQUIRING_ID = {
    "terminal_write",
    "terminal_interrupt",
    "terminal_request_control",
    "terminal_release_control",
    "terminal_snapshot",
    "terminal_wait_for_output",
    "terminal_close",
}


def parse_run_terminal_summary(result: str) -> dict[str, Any]:
    text = str(result or "")
    command_match = re.search(r"^Command:\s*(.+)$", text, flags=re.MULTILINE)
    cwd_match = re.search(r"^CWD:\s*(.+)$", text, flags=re.MULTILINE)
    exit_match = re.search(r"^Exit code:\s*(-?\d+)$", text, flags=re.MULTILINE)
    output = ""
    if "Output:" in text:
        output = text.split("Output:", 1)[-1].strip()
    return {
        "command": str(command_match.group(1)).strip() if command_match else "",
        "cwd": str(cwd_match.group(1)).strip() if cwd_match else "",
        "exitCode": int(exit_match.group(1)) if exit_match else 0,
        "output": output,
    }


class RuntimeEngine(RuntimeGraphOrchestrationMixin):
    def __init__(self, dependencies: RuntimeDependencies | None = None) -> None:
        if dependencies is None:
            terminal_manager = TerminalManager()
            rag_service = RagService()
            dependencies = RuntimeDependencies(
                model_factory=build_chat_model,
                provider_resolver=resolve_provider,
                tool_registry_factory=lambda workspace_root, session_id, run_id, tool_toggles=None: create_default_tool_registry(
                    workspace_root,
                    session_id=session_id,
                    run_id=run_id,
                    terminal_manager=terminal_manager,
                    rag_service=rag_service,
                    tool_toggles=tool_toggles,
                ),
                state_store=RunStateStore(),
                terminal_manager=terminal_manager,
                rag_service=rag_service,
            )
        self.deps = dependencies
        self.graph = create_runtime_graph(self)

    def _provider_capabilities_payload(self, provider: Any) -> dict[str, Any]:
        return provider_capabilities_payload(provider)

    def _build_tool_replay_message(self, name: str, args: dict[str, Any], result: str, status: str) -> SystemMessage:
        return build_tool_replay_message(name, args, result, status)

    def _should_request_approval(self, policy_profile: str, risk_level: str) -> bool:
        return should_request_approval(policy_profile, risk_level)

    def capabilities_payload(self, workspace_root: str, session_id: str | None = None) -> list[dict]:
        return self.deps.tool_registry_factory(workspace_root, session_id, None, None).module_payloads()

    def _should_emit_contract_diagnostic(self, contract: FinalAnswerContract) -> bool:
        return bool(contract.requested_language or contract.exact_output_text or contract.require_sources or contract.require_grounding)

    def _provider_requires_sequential_tools(self, state: AgentState) -> bool:
        capabilities = state.get("provider_capabilities") or {}
        if "requiresSequentialToolLoop" in capabilities:
            return bool(capabilities.get("requiresSequentialToolLoop"))
        return int(capabilities.get("maxToolCallsPerTurn", 8) or 8) <= 1

    def _reset_no_progress(self, state: AgentState) -> None:
        state["no_progress_turns"] = 0

    def _bump_no_progress(self, state: AgentState) -> dict[str, Any]:
        count = int(state.get("no_progress_turns", 0) or 0) + 1
        state["no_progress_turns"] = count
        return {"count": count, "exceeded": count >= MAX_NO_PROGRESS_TURNS}

    def _normalized_tool_signature(self, tool_name: str, tool_args: dict[str, Any]) -> str:
        compact_args = json.dumps(tool_args or {}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return f"{tool_name}::{compact_args}"[:1200]

    def _should_stop_for_repeated_tool_call(self, state: AgentState, tool_name: str, tool_args: dict[str, Any]) -> dict[str, Any]:
        signature = self._normalized_tool_signature(tool_name, tool_args)
        previous = state.get("repeated_tool_signature_streak")
        if isinstance(previous, dict) and str(previous.get("signature") or "") == signature:
            streak = int(previous.get("count", 0) or 0) + 1
        else:
            streak = 1
        state["repeated_tool_signature_streak"] = {"signature": signature, "count": streak}
        recent = list(state.get("recent_tool_signatures", []))
        recent.append(signature)
        if len(recent) > 12:
            recent = recent[-12:]
        state["recent_tool_signatures"] = recent
        same_count = sum(1 for item in recent if item == signature)
        if streak >= 4:
            return {"stop": True, "reason": f"Repeated tool call loop detected ({tool_name} repeated {streak}x consecutively)."}
        if same_count >= 6:
            return {"stop": True, "reason": f"Tool call loop detected ({tool_name} repeated {same_count}x in recent turns)."}
        return {"stop": False}

    def _last_rag_tool_result(self, state: AgentState) -> dict[str, Any] | None:
        return next((item for item in reversed(state.get("tool_results", [])) if item.get("tool") == "rag_lookup"), None)

    def _rag_citation_labels(self, rag_meta: dict[str, Any] | None) -> list[str]:
        labels: list[str] = []
        rag_meta = rag_meta or {}
        for hit in rag_meta.get("hits", []):
            citation = hit.get("citation", {}) if isinstance(hit.get("citation"), dict) else {}
            path = str(citation.get("path") or "").strip()
            if not path:
                continue
            line_start = citation.get("lineStart")
            line_end = citation.get("lineEnd")
            if line_start and line_end and line_start != line_end:
                labels.append(f"{path}:{line_start}-{line_end}")
            elif line_start:
                labels.append(f"{path}:{line_start}")
            else:
                labels.append(path)
        return labels

    def _final_text_has_rag_citation(self, final_text: str, rag_meta: dict[str, Any] | None) -> bool:
        lowered = str(final_text or "").lower()
        for label in self._rag_citation_labels(rag_meta):
            if label.lower() in lowered:
                return True
            path = label.split(":", 1)[0]
            if Path(path).name.lower() in lowered:
                return True
        return False

    def _is_cautious_answer(self, final_text: str) -> bool:
        lowered = str(final_text or "").lower()
        return any(
            marker in lowered
            for marker in (
                "i couldn't find",
                "i could not find",
                "not enough information",
                "insufficient information",
                "documents are insufficient",
                "je n'ai pas trouvé",
                "je ne trouve pas",
                "pas assez d'informations",
                "informations insuffisantes",
            )
        )

    def _build_tool_followup_guidance(
        self,
        state: AgentState,
        tool_name: str,
        message_result: str,
        status: str,
        tool_meta: dict[str, Any] | None = None,
    ) -> SystemMessage:
        contract = FinalAnswerContract.from_payload(state.get("final_contract"))
        lines = [
            "Tool follow-up guidance.",
            f"Last tool: {tool_name}.",
            f"Tool status: {status}.",
            "Use only the verified tool result already present in the conversation. Do not invent missing facts.",
        ]
        if contract.requested_language:
            lines.append(f"Final answer language: {describe_language(contract.requested_language)}.")
        if contract.exact_output_text:
            lines.append(f"Final answer must be exactly: {contract.exact_output_text}")
        if contract.require_sources or tool_name == "web_search":
            lines.append("If you answer now, include at least two source URLs in the final answer.")
        if tool_name in GROUNDING_TOOL_NAMES:
            lines.append("Keep the final answer grounded in the tool result from this run.")
        if tool_name == "rag_lookup":
            meta = tool_meta or {}
            if int(meta.get("strongHitCount", 0)) <= 0:
                lines.append("The retrieved RAG hits are weak or absent. Be explicit if the documents are insufficient.")
            if contract.require_citations_when_rag_used or int(meta.get("strongHitCount", 0)) > 0:
                lines.append("Include compact document citations in the final answer, even if the answer is brief.")
            if meta.get("followupContextUsed"):
                lines.append("The current lookup reused the previous RAG turn as follow-up context; keep that topic only if the current hits still support it.")
        if self._provider_requires_sequential_tools(state):
            lines.append("Provider constraint: if another tool is needed, emit at most one tool call on the next assistant turn.")
        lines.extend(
            [
                "Verified tool result excerpt:",
                message_result[-1500:] if message_result else "(empty tool result)",
            ]
        )
        return SystemMessage(content="\n".join(lines))

    def _extract_grounding_terms(self, state: AgentState) -> list[str]:
        workspace_name = Path(str(state.get("workspace_root") or "")).name
        candidates: list[str] = [workspace_name] if workspace_name else []
        if workspace_name:
            candidates.append(workspace_name.replace("-", " ").replace("_", " "))
        for item in state.get("tool_results", []):
            tool_name = str(item.get("tool") or "")
            if tool_name not in GROUNDING_TOOL_NAMES:
                continue
            message = str(item.get("result") or "")
            meta = item.get("meta", {}) if isinstance(item.get("meta"), dict) else {}
            for path in meta.get("topDocumentPaths", []):
                candidates.append(Path(str(path)).name)
            for label in self._rag_citation_labels(meta):
                candidates.append(label)
            candidates.extend(str(term) for term in meta.get("matchedTerms", []))
            candidates.extend(re.findall(r"\b[a-zA-Z0-9]+(?:[-_][a-zA-Z0-9]+)+\b", message))
            candidates.extend(re.findall(r"\b[a-zA-Z0-9._-]+\.(?:py|md|txt|json|yaml|yml)\b", message))
            for path_match in re.findall(r"/[^\s]+", message):
                for part in Path(path_match).parts[-4:]:
                    cleaned = str(part).strip()
                    if len(cleaned) >= 4:
                        candidates.append(cleaned)
        unique: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            cleaned = str(candidate or "").strip("`\"' ,.:;").lower()
            if len(cleaned) < 4 or cleaned in COMMON_GROUNDING_STOPWORDS or cleaned.isdigit():
                continue
            if cleaned not in seen:
                seen.add(cleaned)
                unique.append(cleaned)
        return unique[:12]

    def _final_text_matches_evidence_facts(self, state: AgentState, final_text: str) -> bool:
        lowered = str(final_text or "").lower()
        evidence_map = dict(state.get("evidence_map") or {})
        facts = evidence_map.get("facts", {}) if isinstance(evidence_map.get("facts"), dict) else {}
        matches: list[bool] = []
        checkpoint = str(facts.get("checkpoint") or "").strip().lower()
        if checkpoint:
            matches.append(checkpoint in lowered)
        product_name = str(facts.get("product_name") or "").strip().lower()
        if product_name:
            matches.append(product_name in lowered)
        preferred_editor = str(facts.get("preferred_editor") or "").strip().lower()
        if preferred_editor:
            matches.append(preferred_editor in lowered)
        stack_terms = [str(item).strip().lower() for item in facts.get("stack_terms", []) if str(item).strip()]
        if stack_terms:
            matches.append(any(term in lowered for term in stack_terms))
        extra_files = [str(item).strip().lower() for item in facts.get("extra_files", []) if str(item).strip()]
        if extra_files:
            matches.append(any(term in lowered for term in extra_files[:6]))
        return bool(matches) and all(matches)

    def _preferred_workspace_label(self, state: AgentState) -> str | None:
        workspace_name = Path(str(state.get("workspace_root") or "")).name
        if not workspace_name:
            return None
        label = workspace_name.replace("-", " ").replace("_", " ").strip()
        return label or None

    def _extract_fact_map_from_text(self, text: str) -> dict[str, Any]:
        content = str(text or "")
        facts: dict[str, Any] = {}
        checkpoint = re.search(r"\b(?:current\s+checkpoint|recorded\s+checkpoint|checkpoint)(?:\s+is)?\s*:\s*([A-Z0-9_-]{4,})\b", content, flags=re.IGNORECASE)
        if not checkpoint:
            checkpoint = re.search(r"\b(?:recorded\s+checkpoint|checkpoint)\s+is\s+([A-Z0-9_-]{4,})\b", content, flags=re.IGNORECASE)
        if checkpoint:
            facts["checkpoint"] = checkpoint.group(1).strip()
        else:
            bare_identifier = normalize_whitespace(content).strip(" .`*_")
            if re.fullmatch(r"[A-Z0-9]+(?:[_-][A-Z0-9]+)+", bare_identifier):
                facts["checkpoint"] = bare_identifier
        product_name = re.search(r"\bproduct\s+name\s*:\s*([^\n.]+)", content, flags=re.IGNORECASE)
        if not product_name:
            product_name = re.search(r"\bproduct\s+name\s+is\s+([^\n.]+)", content, flags=re.IGNORECASE)
        if product_name:
            facts["product_name"] = normalize_whitespace(product_name.group(1)).strip(" .")
        editor = re.search(r"\b(?:preferred|recommended)\s+editor\s*:\s*([^\n.]+)", content, flags=re.IGNORECASE)
        if not editor:
            editor = re.search(r"\b(?:preferred|recommended)\s+editor\s+is\s+([^\n.]+)", content, flags=re.IGNORECASE)
        if editor:
            facts["preferred_editor"] = normalize_whitespace(editor.group(1)).strip(" .")
        elif re.search(r"\bvs\s*code\b|\bvscode\b", content, flags=re.IGNORECASE):
            facts["preferred_editor"] = "VS Code"
        status = re.search(r"\bphase\s+\d+\s+status\s*:\s*([^\n.]+)", content, flags=re.IGNORECASE)
        if not status:
            status = re.search(r"\bphase\s+\d+\s+status\s+is\s+([^\n.]+)", content, flags=re.IGNORECASE)
        if status:
            facts["status"] = normalize_whitespace(status.group(1)).strip(" .")
        stack_terms: list[str] = []
        if re.search(r"\bpython\b", content, flags=re.IGNORECASE):
            stack_terms.append("Python")
        if re.search(r"\bstreamlit\b", content, flags=re.IGNORECASE):
            stack_terms.append("Streamlit")
        if re.search(r"\bfastapi\b", content, flags=re.IGNORECASE):
            stack_terms.append("FastAPI")
        if stack_terms:
            facts["stack_terms"] = stack_terms
        extra_files = re.findall(r"\b([A-Za-z0-9_./-]+\.[A-Za-z0-9]{1,8})\b", content)
        if extra_files:
            facts["extra_files"] = _dedupe_strings([Path(item).name for item in extra_files])
        return facts

    def _merge_facts(self, base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
        merged = dict(base or {})
        for key, value in incoming.items():
            if key in {"stack_terms", "extra_files"}:
                merged[key] = _dedupe_strings([*list(merged.get(key, []) or []), *list(value or [])])
            elif value and not merged.get(key):
                merged[key] = value
            elif value:
                merged[key] = value
        return merged

    def _build_evidence_signature(self, evidence_map: dict[str, Any]) -> str:
        files = evidence_map.get("files", {}) if isinstance(evidence_map.get("files"), dict) else {}
        directories = evidence_map.get("directories", {}) if isinstance(evidence_map.get("directories"), dict) else {}
        terminals = evidence_map.get("terminals", []) if isinstance(evidence_map.get("terminals"), list) else []
        facts = evidence_map.get("facts", {}) if isinstance(evidence_map.get("facts"), dict) else {}
        signature_parts = {
            "files": sorted(files.keys()),
            "directories": sorted(directories.keys()),
            "terminalCommands": [str(item.get("command") or "") for item in terminals[-4:] if isinstance(item, dict)],
            "facts": facts,
        }
        return json.dumps(signature_parts, ensure_ascii=False, sort_keys=True)

    def _note_executed_tool(self, state: AgentState, tool_name: str) -> None:
        executed = list(state.get("executed_tools", []))
        executed.append(tool_name)
        state["executed_tools"] = executed[-50:]

    def _note_tool_success(
        self,
        state: AgentState,
        tool_name: str,
        tool_args: dict[str, Any],
        message_result: str,
        tool_meta: dict[str, Any] | None = None,
    ) -> None:
        self._note_executed_tool(state, tool_name)
        if tool_name == "read_file":
            path = normalize_whitespace(str(tool_args.get("path") or ""))
            if path:
                state["last_read_file_path"] = path
            state["last_read_file_content"] = str(message_result or "")
            return
        if tool_name in {"open_url", "open_resource"}:
            text = str(message_result or "")
            url_match = re.search(r"https?://[^\s)\]]+", text)
            if url_match:
                state["last_opened_url"] = url_match.group(0)
        if tool_name in {"open_file", "open_resource"}:
            path = normalize_whitespace(str(tool_args.get("path") or ""))
            if path:
                state["last_opened_file_path"] = path
        if tool_name == "write_file":
            path = normalize_whitespace(str(tool_args.get("path") or ""))
            if path:
                written = list(state.get("written_files", []))
                written.append(path)
                state["written_files"] = written[-100:]
            return
        if tool_name == "web_search":
            urls = extract_urls(message_result)
            state["web_search_last_result_count"] = len(urls)
            state["web_search_last_result_urls"] = urls[:20]
            return
        if tool_name == "rag_lookup":
            meta = tool_meta.get("meta", {}) if tool_meta else {}
            hits = tool_meta.get("hits", []) if tool_meta else []
            state["rag_lookup_last_structured"] = bool(tool_meta)
            state["rag_lookup_last_status"] = str(tool_meta.get("status") or "") if tool_meta and tool_meta.get("status") else None
            state["rag_lookup_last_hit_count"] = len(hits) if isinstance(hits, list) else 0
            state["rag_lookup_last_hits"] = list(hits)[:12] if isinstance(hits, list) else []

    def _maybe_build_post_tool_reasoning_hint(self, state: AgentState, tool_name: str, message_result: str, tool_meta: dict[str, Any] | None = None) -> str | None:
        assessment = self._refresh_goal_tracking(state, str(state.get("final_text") or ""))
        gap = assessment.gap
        if gap.missing_evidence:
            next_input_instruction = self._next_missing_input_instruction(gap.missing_evidence[0])
            lines = [
                "Continue only if the goal gap is still open.",
                "Missing evidence: " + ", ".join(gap.missing_evidence[:6]) + ".",
                "Use one next useful step; do not add steps just in case.",
            ]
            if next_input_instruction:
                lines.append(next_input_instruction)
            if self._provider_requires_sequential_tools(state):
                lines.append("If the gap remains open, emit at most one next tool call.")
            return "\n".join(lines)
        if gap.missing_facts:
            return (
                "The verified evidence may already be sufficient. If so, answer now. "
                "Otherwise repair only the missing facts from the verified evidence. "
                f"Missing facts: {', '.join(gap.missing_facts[:6])}."
            )
        contract = MultiStepContract.from_payload(state.get("multi_step_contract"))
        if is_multistep_contract_active(contract):
            step_progress = self._compute_step_progress(contract, dict(state.get("evidence_map") or {}))
            missing_inputs = list(step_progress.get("missingInputs") or [])
            missing_fields = list(step_progress.get("missingAnswerFields") or [])
            if missing_inputs:
                lines = [
                    "Continue only if the goal gap is still open.",
                    "You are still missing verified evidence from: " + ", ".join(missing_inputs[:6]) + ".",
                    "Take one next useful step only; avoid extra steps that do not add new evidence.",
                ]
                next_input_instruction = self._next_missing_input_instruction(missing_inputs[0])
                if next_input_instruction:
                    lines.append(next_input_instruction)
                if self._provider_requires_sequential_tools(state):
                    lines.append("If the gap remains open, emit at most one next tool call.")
                return "\n".join(lines)
            if missing_fields:
                lines = [
                    "The verified evidence may already be enough. If so, answer now.",
                    "If a field is still unsupported, say that explicitly instead of adding extra steps.",
                    "Missing answer fields: " + ", ".join(missing_fields[:6]) + ".",
                ]
                return "\n".join(lines)
        if tool_name == "rag_lookup":
            meta = tool_meta.get("meta", {}) if tool_meta else {}
            status = str(tool_meta.get("status") or meta.get("status") or "")
            strong_hits = int(meta.get("strongHitCount", 0) or 0)
            if status == "ok" and strong_hits > 0:
                return "rag_lookup already returned relevant results. Use those hits now and include citations only if the goal still requires them."
            if status in {"no_hits", "indexing", "disabled", "error"}:
                return f"rag_lookup returned status '{status}'. Be transparent about that state."
        if tool_name == "read_file":
            content = str(message_result or "")
            line_count = len(content.splitlines())
            if line_count <= 20:
                return "read_file returned a short file. If this already covers the goal, answer directly from the file contents."
            return "read_file returned a longer file. Summarize from the file contents if enough evidence is present; do not claim the file was opened unless open_file succeeded."
        if tool_name == "list_directory":
            details = self._directory_project_details({"result": message_result})
            if details and details.get("preferredTerms"):
                return "list_directory already exposed salient project signals. If the request is about project type or stack, answer from those signals if they are sufficient."
        if tool_name == "web_search":
            urls = extract_urls(message_result)
            if urls:
                return "web_search already returned sources. Answer first, then add a separate Sources section with Markdown links using those URLs."
        if tool_name == "run_terminal":
            parsed = parse_run_terminal_summary(message_result)
            if str(parsed.get("output") or "").strip():
                return "run_terminal already returned concrete output. Base the next step or final answer on that actual output, not on assumptions."
        return None

    def _seed_evidence_map_from_messages(self, messages: list[BaseMessage]) -> dict[str, Any]:
        facts: dict[str, Any] = {}
        for message in messages:
            if not isinstance(message, AIMessage):
                continue
            content = str(message.content or "")
            if not content.strip():
                continue
            facts = self._merge_facts(facts, self._extract_fact_map_from_text(content))
        evidence_map = {"files": {}, "directories": {}, "terminals": [], "facts": facts}
        evidence_map["signature"] = self._build_evidence_signature(evidence_map)
        return evidence_map

    def _update_evidence_map(
        self,
        state: AgentState,
        tool_name: str,
        tool_args: dict[str, Any],
        message_result: str,
        tool_meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        evidence_map = dict(state.get("evidence_map") or {})
        files = dict(evidence_map.get("files") or {})
        directories = dict(evidence_map.get("directories") or {})
        terminals = list(evidence_map.get("terminals") or [])
        facts = dict(evidence_map.get("facts") or {})

        if tool_name == "read_file":
            path = normalize_whitespace(str(tool_args.get("path") or "")).rstrip("/")
            excerpt = str(message_result or "")[:1500]
            file_facts = self._extract_fact_map_from_text(message_result)
            files[path or f"read_file:{len(files)+1}"] = {
                "path": path,
                "content_excerpt": excerpt,
                "facts": file_facts,
            }
            facts = self._merge_facts(facts, file_facts)
        elif tool_name == "list_directory":
            path = normalize_whitespace(str(tool_args.get("path") or ".")).rstrip("/") or "."
            details = self._directory_project_details({"result": message_result}) or {}
            directories[path] = {
                "path": path,
                "entries": list(details.get("entries") or []),
                "descriptors": list(details.get("descriptors") or []),
                "salientTerms": list(details.get("salientTerms") or []),
            }
            facts = self._merge_facts(facts, {"stack_terms": list(details.get("preferredTerms") or details.get("salientTerms") or [])})
            if details.get("entries"):
                facts = self._merge_facts(facts, {"extra_files": [Path(str(item)).name for item in details.get("entries", [])]})
        elif tool_name == "run_terminal":
            parsed = parse_run_terminal_summary(message_result)
            terminals.append(
                {
                    "command": parsed.get("command"),
                    "cwd": parsed.get("cwd"),
                    "output_excerpt": str(parsed.get("output") or "")[:1000],
                    "exitCode": parsed.get("exitCode"),
                }
            )
            facts = self._merge_facts(facts, self._extract_fact_map_from_text(str(parsed.get("output") or "")))
        elif tool_name == "rag_lookup":
            meta = tool_meta.get("meta") if tool_meta else {}
            evidence_map["rag"] = {
                "hits": list((tool_meta or {}).get("hits", []) or []),
                "meta": dict(meta or {}),
            }
            facts = self._merge_facts(facts, self._extract_fact_map_from_text(message_result))
        elif tool_name == "web_search":
            urls = extract_urls(message_result)
            evidence_map["web"] = {"urls": urls, "excerpt": str(message_result or "")[:1500]}

        evidence_map["files"] = files
        evidence_map["directories"] = directories
        evidence_map["terminals"] = terminals[-8:]
        evidence_map["facts"] = facts
        evidence_map["signature"] = self._build_evidence_signature(evidence_map)
        return evidence_map

    def _compute_step_progress(self, contract: MultiStepContract, evidence_map: dict[str, Any]) -> dict[str, Any]:
        covered_inputs: list[str] = []
        missing_inputs: list[str] = []
        files = evidence_map.get("files", {}) if isinstance(evidence_map.get("files"), dict) else {}
        directories = evidence_map.get("directories", {}) if isinstance(evidence_map.get("directories"), dict) else {}
        facts = evidence_map.get("facts", {}) if isinstance(evidence_map.get("facts"), dict) else {}

        for item in contract.required_inputs:
            kind = str(item.get("kind") or "")
            target = normalize_whitespace(str(item.get("target") or "")).rstrip("/") or "."
            key = f"{kind}:{target}"
            if kind == "file":
                matched = any(Path(path).name == Path(target).name or normalize_whitespace(path).rstrip("/") == target for path in files)
            elif kind == "directory":
                matched = target in directories or (target == "." and directories)
            else:
                matched = False
            (covered_inputs if matched else missing_inputs).append(key)

        covered_fields: list[str] = []
        missing_fields: list[str] = []
        for field_name in contract.required_answer_fields:
            matched = False
            if field_name == "checkpoint":
                matched = bool(facts.get("checkpoint"))
            elif field_name == "product_name":
                matched = bool(facts.get("product_name"))
            elif field_name == "stack":
                matched = bool(facts.get("stack_terms"))
            elif field_name == "preferred_editor":
                matched = bool(facts.get("preferred_editor"))
            elif field_name == "status":
                matched = bool(facts.get("status"))
            elif field_name == "extra_file":
                matched = bool(facts.get("extra_files"))
            (covered_fields if matched else missing_fields).append(field_name)

        completion_numerator = len(covered_inputs) + len(covered_fields)
        completion_denominator = len(contract.required_inputs) + len(contract.required_answer_fields)
        return {
            "coveredInputs": covered_inputs,
            "missingInputs": missing_inputs,
            "coveredAnswerFields": covered_fields,
            "missingAnswerFields": missing_fields,
            "completedStepCount": len(covered_inputs),
            "completionRatio": (completion_numerator / completion_denominator) if completion_denominator else 1.0,
        }

    def _summarize_evidence_map(self, evidence_map: dict[str, Any]) -> dict[str, Any]:
        files = evidence_map.get("files", {}) if isinstance(evidence_map.get("files"), dict) else {}
        directories = evidence_map.get("directories", {}) if isinstance(evidence_map.get("directories"), dict) else {}
        terminals = evidence_map.get("terminals", []) if isinstance(evidence_map.get("terminals"), list) else []
        facts = evidence_map.get("facts", {}) if isinstance(evidence_map.get("facts"), dict) else {}
        return {
            "fileCount": len(files),
            "directoryCount": len(directories),
            "terminalCount": len(terminals),
            "factKeys": sorted(str(key) for key in facts.keys()),
            "files": sorted(str(path) for path in files.keys())[:6],
            "directories": sorted(str(path) for path in directories.keys())[:6],
            "facts": {
                key: value
                for key, value in facts.items()
                if key in {"checkpoint", "product_name", "preferred_editor", "status", "stack_terms", "extra_files"}
            },
        }

    def _build_goal_state(self, state: AgentState) -> GoalState:
        prompt = normalize_whitespace(self._last_user_message(state.get("messages", [])))
        final_contract = FinalAnswerContract.from_payload(state.get("final_contract"))
        multi_step_contract = MultiStepContract.from_payload(state.get("multi_step_contract"))
        required_evidence_kinds: list[str] = []
        required_facts = list(multi_step_contract.required_answer_fields)
        if multi_step_contract.required_inputs:
            required_evidence_kinds.extend(
                f"{item.get('kind')}:{normalize_whitespace(str(item.get('target') or ''))}"
                for item in multi_step_contract.required_inputs
                if str(item.get("kind") or "").strip() and str(item.get("target") or "").strip()
            )
        if final_contract.require_sources:
            required_evidence_kinds.append("web:sources")
        if final_contract.require_grounding_from_rag:
            required_evidence_kinds.append("rag:hits")
        if final_contract.require_grounding and any(
            keyword in prompt.lower() for keyword in ("inspect", "read", "search", "look up", "terminal", "workspace")
        ):
            required_evidence_kinds.append("tool:grounding")
        return GoalState(
            user_intent=prompt,
            required_evidence_kinds=_dedupe_strings(required_evidence_kinds),
            required_facts=_dedupe_strings(required_facts),
            required_citations=bool(final_contract.require_sources or final_contract.require_citations_when_rag_used),
            required_output_constraint={
                "language": final_contract.requested_language,
                "exactOutputText": final_contract.exact_output_text,
                "requireGrounding": final_contract.require_grounding,
                "requireGroundingFromRag": final_contract.require_grounding_from_rag,
            },
        )

    def _build_realization_state(self, state: AgentState, candidate_answer: str) -> RealizationState:
        evidence_map = dict(state.get("evidence_map") or {})
        evidence_summary = self._summarize_evidence_map(evidence_map)
        facts = evidence_map.get("facts", {}) if isinstance(evidence_map.get("facts"), dict) else {}
        available_evidence = {
            **evidence_summary,
            "executedTools": list(dict.fromkeys(state.get("executed_tools", [])))[-20:],
            "webSearchUrls": list(state.get("web_search_last_result_urls", []))[:8],
            "ragStatus": state.get("rag_lookup_last_status"),
            "ragHitCount": state.get("rag_lookup_last_hit_count"),
            "lastReadFilePath": state.get("last_read_file_path"),
            "lastOpenedUrl": state.get("last_opened_url"),
        }
        return RealizationState(
            available_evidence=available_evidence,
            proven_facts=dict(facts),
            candidate_answer=normalize_whitespace(candidate_answer),
        )

    def _build_goal_gap(
        self,
        state: AgentState,
        goal: GoalState,
        realization: RealizationState,
        base_violations: list[tuple[str, str]] | None = None,
    ) -> GoalGap:
        evidence_map = dict(state.get("evidence_map") or {})
        contract = MultiStepContract.from_payload(state.get("multi_step_contract"))
        step_progress = self._compute_step_progress(contract, evidence_map)
        missing_evidence = list(step_progress.get("missingInputs") or [])
        missing_facts = list(step_progress.get("missingAnswerFields") or [])

        facts = realization.proven_facts
        if "web:sources" in goal.required_evidence_kinds and len(state.get("web_search_last_result_urls", [])) < 2:
            missing_evidence.append("web:sources")
        if "rag:hits" in goal.required_evidence_kinds and int(state.get("rag_lookup_last_hit_count") or 0) <= 0:
            missing_evidence.append("rag:hits")
        if "tool:grounding" in goal.required_evidence_kinds and not state.get("tool_results"):
            missing_evidence.append("tool:grounding")

        for fact_name in goal.required_facts:
            if fact_name == "checkpoint" and not facts.get("checkpoint"):
                missing_facts.append("checkpoint")
            elif fact_name == "product_name" and not facts.get("product_name"):
                missing_facts.append("product_name")
            elif fact_name == "stack" and not facts.get("stack_terms"):
                missing_facts.append("stack")
            elif fact_name == "preferred_editor" and not facts.get("preferred_editor"):
                missing_facts.append("preferred_editor")
            elif fact_name == "status" and not facts.get("status"):
                missing_facts.append("status")
            elif fact_name == "extra_file" and not facts.get("extra_files"):
                missing_facts.append("extra_file")

        answer_defects = [code for code, _message in (base_violations or [])]
        return GoalGap(
            missing_evidence=_dedupe_strings(missing_evidence),
            missing_facts=_dedupe_strings(missing_facts),
            answer_defects=_dedupe_strings(answer_defects),
            can_gather_more_evidence=bool(missing_evidence),
            is_complete=not missing_evidence and not missing_facts and not answer_defects,
        )

    def _refresh_goal_tracking(
        self,
        state: AgentState,
        candidate_answer: str | None = None,
        base_violations: list[tuple[str, str]] | None = None,
    ) -> GoalAssessment:
        goal = self._build_goal_state(state)
        realization = self._build_realization_state(state, candidate_answer if candidate_answer is not None else str(state.get("final_text") or ""))
        gap = self._build_goal_gap(state, goal, realization, base_violations)
        action = "finish"
        multi_step_contract = MultiStepContract.from_payload(state.get("multi_step_contract"))
        if (
            is_multistep_contract_active(multi_step_contract)
            and gap.missing_evidence
            and gap.can_gather_more_evidence
            and self._has_runtime_actionable_missing_evidence(gap.missing_evidence)
        ):
            action = "gather_more_evidence"
        elif gap.missing_facts or gap.answer_defects:
            action = "repair_answer"
        assessment = GoalAssessment(
            action=action,
            mismatch_codes=list(base_violations or []),
            goal=goal,
            realization=realization,
            gap=gap,
        )
        state["goal_summary"] = goal.to_payload()
        state["realization_summary"] = realization.to_payload()
        state["goal_gap_summary"] = gap.to_payload()
        return assessment

    def _goal_gap_events(self, run_id: str, state: AgentState, reason: str) -> list[dict[str, Any]]:
        return [
            run_diagnostic(
                run_id,
                "goal_summary",
                "Goal summary captured.",
                data={**dict(state.get("goal_summary") or {}), "reason": reason},
            ),
            run_diagnostic(
                run_id,
                "realization_summary",
                "Realization summary captured.",
                data={**dict(state.get("realization_summary") or {}), "reason": reason},
            ),
            run_diagnostic(
                run_id,
                "goal_gap_summary",
                "Goal gap summary captured.",
                data={**dict(state.get("goal_gap_summary") or {}), "reason": reason},
            ),
        ]

    def _build_multistep_progress_events(
        self,
        run_id: str,
        contract: MultiStepContract,
        step_progress: dict[str, Any],
        evidence_map: dict[str, Any],
        *,
        reason: str,
    ) -> list[dict[str, Any]]:
        progress_data = {
            "reason": reason,
            "expectedStepCount": int(contract.expected_step_count or 0),
            "requiredInputCount": len(contract.required_inputs),
            "requiredAnswerFieldCount": len(contract.required_answer_fields),
            "completedStepCount": int(step_progress.get("completedStepCount") or 0),
            "completionRatio": round(float(step_progress.get("completionRatio") or 0.0), 4),
            "coveredInputs": list(step_progress.get("coveredInputs") or []),
            "missingInputs": list(step_progress.get("missingInputs") or []),
            "coveredAnswerFields": list(step_progress.get("coveredAnswerFields") or []),
            "missingAnswerFields": list(step_progress.get("missingAnswerFields") or []),
        }
        evidence_summary = self._summarize_evidence_map(evidence_map)
        return [
            run_diagnostic(
                run_id,
                "multistep_progress_snapshot",
                "Multi-step progress snapshot captured.",
                data=progress_data,
            ),
            run_diagnostic(
                run_id,
                "multistep_evidence_summary",
                "Multi-step evidence summary captured.",
                data={**evidence_summary, "reason": reason},
            ),
        ]

    def _workspace_label_from_text(self, text: str) -> str | None:
        match = re.search(r"\bin the\s+(.+?)\s+workspace\b", str(text or ""), flags=re.IGNORECASE)
        if match:
            label = normalize_whitespace(match.group(1)).strip(" ,.")
            if label:
                return label
        path_match = re.search(r"current directory is\s+`?([^`\s,]+)`?", str(text or ""), flags=re.IGNORECASE)
        if path_match:
            basename = Path(str(path_match.group(1)).strip()).name
            label = basename.replace("-", " ").replace("_", " ").strip()
            if label:
                return label
        return None

    def _previous_assistant_message(self, state: AgentState) -> str:
        messages = list(state.get("messages", []))
        if messages and isinstance(messages[-1], AIMessage):
            messages = messages[:-1]
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                content = normalize_whitespace(str(message.content or ""))
                if content:
                    return content
        return ""

    def _followup_grounding_terms_from_text(self, text: str) -> list[str]:
        candidates = re.findall(r"\b[a-zA-Z0-9]+(?:[-_][a-zA-Z0-9]+)+\b", str(text or ""))
        candidates.extend(re.findall(r"\b[A-Z0-9]{4,}\b", str(text or "")))
        terms: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            cleaned = str(candidate or "").strip("`\"' ,.:;").lower()
            if len(cleaned) < 4 or cleaned in COMMON_GROUNDING_STOPWORDS or cleaned.isdigit():
                continue
            if cleaned not in seen:
                seen.add(cleaned)
                terms.append(cleaned)
        return terms[:8]

    def _rag_snippet_segments(self, snippet: str) -> list[str]:
        lines = [normalize_whitespace(line) for line in str(snippet or "").splitlines() if normalize_whitespace(line)]
        segments = list(lines)
        for chunk in re.split(r"(?<=[.!?])\s+", str(snippet or "")):
            cleaned = normalize_whitespace(chunk)
            if cleaned and cleaned not in segments:
                segments.append(cleaned)
        return segments

    def _best_rag_evidence(self, state: AgentState) -> dict[str, Any] | None:
        rag_result = self._last_rag_tool_result(state)
        if not rag_result:
            return None
        rag_meta = rag_result.get("meta", {}) if isinstance(rag_result.get("meta"), dict) else {}
        hits = rag_meta.get("hits", []) if isinstance(rag_meta.get("hits"), list) else []
        if int(rag_meta.get("strongHitCount", 0) or 0) <= 0 or not hits:
            return None

        prompt = self._last_user_message(state.get("messages", []))
        query_tokens = [token for token in tokenize(prompt) if len(token) >= 3]
        best: dict[str, Any] | None = None
        best_score = -1.0

        for hit in hits:
            hit_score = float(hit.get("score", 0.0) or 0.0)
            for segment in self._rag_snippet_segments(str(hit.get("snippet") or "")):
                lowered = segment.lower()
                overlap = sum(1.0 for token in query_tokens if token in lowered)
                if ":" in segment:
                    label, _, value = segment.partition(":")
                    if any(token in label.lower() for token in query_tokens):
                        overlap += 1.5
                    if value.strip():
                        overlap += 0.25
                score = overlap + hit_score / 10.0
                if score <= best_score:
                    continue
                best_score = score
                best = {"hit": hit, "segment": segment, "query_tokens": query_tokens}

        if not best:
            return None

        segment = str(best["segment"] or "").strip()
        query_tokens = list(best.get("query_tokens", []))
        label_text = ""
        value_text = segment
        if ":" in segment:
            label, _, value = segment.partition(":")
            if any(token in label.lower() for token in query_tokens):
                label_text = normalize_whitespace(label)
                value_text = normalize_whitespace(value)
        salient_seed = value_text or segment
        salient_terms = [
            token
            for token in tokenize(salient_seed)
            if token not in RAG_SALIENCE_STOPWORDS and token not in query_tokens and len(token) >= 3
        ]
        citation_label = next(iter(self._rag_citation_labels({"hits": [best["hit"]]})), "")
        return {
            "segment": segment,
            "label": label_text,
            "value": value_text,
            "salientTerms": salient_terms[:6],
            "citationLabel": citation_label,
        }

    def _compact_rag_answer(self, state: AgentState) -> str | None:
        evidence = self._best_rag_evidence(state)
        if not evidence:
            return None

        contract = FinalAnswerContract.from_payload(state.get("final_contract"))
        prompt = self._last_user_message(state.get("messages", []))
        wants_brief = bool(re.search(r"\b(very briefly|briefly|brief|one line|short)\b", str(prompt or "").lower()))
        value = str(evidence.get("value") or "").strip()
        label = str(evidence.get("label") or "").strip()
        sentence = str(evidence.get("segment") or "").strip()
        citation_label = str(evidence.get("citationLabel") or "").strip()

        if wants_brief and value:
            answer = value.rstrip(".")
        elif label and value:
            answer = f"{label}: {value}".rstrip(".")
        else:
            answer = sentence.rstrip(".")
        if citation_label:
            return f"{answer}. ({citation_label})"
        return f"{answer}."

    def _is_followup_grounded_turn(self, state: AgentState) -> bool:
        if state.get("used_tool_names"):
            return False
        prompt = self._last_user_message(state.get("messages", []))
        lowered = str(prompt or "").strip().lower()
        if not lowered:
            return False
        if not FOLLOWUP_SUMMARY_PATTERN.search(lowered):
            return False
        return bool(FOLLOWUP_REFERENCE_PATTERN.search(lowered))

    def _synthesize_followup_grounded_summary(self, state: AgentState) -> str | None:
        previous = self._previous_assistant_message(state)
        if not previous:
            return None
        text = normalize_whitespace(previous)
        workspace_label = self._workspace_label_from_text(text) or self._preferred_workspace_label(state)
        if workspace_label:
            match = re.search(
                r"current directory is\s+(`?[^`,]+`?)\.?\s+(?:in the\s+.+?\s+workspace,\s+)?(?:and\s+)?the contents of the workspace root are:\s+(.+)",
                text,
                flags=re.IGNORECASE,
            )
            if not match:
                match = re.search(
                    r"current directory is\s+(`?[^`,]+`?)\.?\s+(?:in the\s+.+?\s+workspace,\s+)?(?:and\s+)?the workspace root contains\s+(.+)",
                    text,
                    flags=re.IGNORECASE,
                )
            if match:
                directory = normalize_whitespace(match.group(1)).strip(" ,")
                entries = [entry.strip(" ,.") for entry in re.split(r",\s*", match.group(2)) if entry.strip(" ,.")]
                excerpt = ", ".join(entries[:4])
                return f"The {workspace_label} workspace is at {directory}, and its root includes {excerpt}."
        return text[:260].rstrip()

    def _build_terminal_fallback_summary(self, state: AgentState, terminal_result: dict[str, Any]) -> str | None:
        contract = FinalAnswerContract.from_payload(state.get("final_contract"))
        requested_language = str(contract.requested_language or "").lower()
        parsed = parse_run_terminal_summary(str(terminal_result.get("result") or ""))
        output = str(parsed.get("output") or "").strip()
        command = str(parsed.get("command") or "").strip()
        exit_code = int(parsed.get("exitCode") or 0)
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        sample = normalize_whitespace(lines[0] if lines else output)
        if sample:
            sample = sample[:160].rstrip()
        if exit_code != 0:
            if requested_language == "fr":
                detail = f' : "{sample}"' if sample else ""
                return f"La commande du terminal a échoué avec le code {exit_code}{detail}."
            detail = f': "{sample}"' if sample else ""
            return f"The terminal command failed with exit code {exit_code}{detail}."
        if sample:
            if requested_language == "fr":
                return f'Le resultat affiche par le terminal est "{sample}".'
            return f'The terminal result was "{sample}".'
        if requested_language == "fr":
            if command:
                return f'La commande du terminal "{command}" s\'est terminée sans sortie visible.'
            return "La commande du terminal s'est terminée sans sortie visible."
        if command:
            return f'The terminal command "{command}" completed without visible output.'
        return "The terminal command completed without visible output."

    def _build_directory_fallback_summary(self, state: AgentState, directory_result: dict[str, Any]) -> str | None:
        details = self._directory_project_details(directory_result)
        if not details:
            return None
        prompt = str(self._last_user_message(state.get("messages", [])) or "").lower()
        if "project" not in prompt and "workspace" not in prompt:
            return None

        descriptors = details["descriptors"]
        entries = details["entries"]
        summary = ", ".join(dict.fromkeys(descriptors))
        excerpt = ", ".join(entries[:5])
        return f"This looks like {summary} with workspace entries such as {excerpt}."

    def _directory_project_details(self, directory_result: dict[str, Any]) -> dict[str, Any] | None:
        result = str(directory_result.get("result") or "")
        entries: list[str] = []
        for line in result.splitlines():
            parts = line.split("\t", 1)
            entry = parts[-1].strip() if parts else line.strip()
            if entry:
                entries.append(Path(entry).name.rstrip("/"))
        if not entries:
            return None

        lowered_entries = {entry.lower() for entry in entries}
        descriptors: list[str] = []
        if {"frontend", "backend", "src"}.issubset(lowered_entries):
            descriptors.append("a split frontend/backend/src application")
        if "streamlit" in " ".join(lowered_entries) or ".streamlit" in lowered_entries:
            descriptors.append("a Streamlit app")
        if "pyproject.toml" in lowered_entries or any(entry.endswith(".py") or entry == "src" for entry in lowered_entries):
            descriptors.append("a Python project")
        if any("assistant" in entry for entry in lowered_entries):
            descriptors.append("an assistant app")
        if not descriptors:
            descriptors.append("a software project")
        salient_terms: list[str] = []
        if any("streamlit" in entry for entry in lowered_entries) or ".streamlit" in lowered_entries:
            salient_terms.append("streamlit")
        if "pyproject.toml" in lowered_entries or "src" in lowered_entries or any(entry.endswith(".py") for entry in lowered_entries):
            salient_terms.append("python")
        if any("assistant" in entry for entry in lowered_entries):
            salient_terms.append("assistant")
        if "frontend" in lowered_entries:
            salient_terms.append("frontend")
        if "backend" in lowered_entries:
            salient_terms.append("backend")
        salient_terms = list(dict.fromkeys(salient_terms))
        preferred_terms = [term for term in salient_terms if term in {"streamlit", "python", "assistant"}]
        return {
            "entries": entries,
            "descriptors": list(dict.fromkeys(descriptors)),
            "salientTerms": salient_terms,
            "preferredTerms": preferred_terms,
        }

    def _inject_workspace_label_into_followup(self, text: str, workspace_label: str, language: str | None = None) -> str | None:
        cleaned = normalize_whitespace(text)
        if not cleaned or not workspace_label:
            return None
        lowered = cleaned.lower()
        workspace_lower = workspace_label.lower()
        if workspace_lower in lowered:
            return cleaned
        if lowered.startswith("the workspace "):
            return f"The {workspace_label} workspace {cleaned[len('The workspace '):]}".strip()
        if lowered.startswith("the workspace"):
            return cleaned.replace("The workspace", f"The {workspace_label} workspace", 1)
        if lowered.startswith("le workspace ") or lowered.startswith("l'espace de travail "):
            return f"L'espace de travail {workspace_label} {cleaned.split(' ', 2)[-1]}".strip()
        return None

    def _collect_multistep_violations(self, state: AgentState, final_text: str) -> list[tuple[str, str]]:
        contract = MultiStepContract.from_payload(state.get("multi_step_contract"))
        if not is_multistep_contract_active(contract):
            return []
        evidence_map = dict(state.get("evidence_map") or {})
        step_progress = self._compute_step_progress(contract, evidence_map)
        state["step_progress"] = step_progress
        violations: list[tuple[str, str]] = []
        missing_inputs = list(step_progress.get("missingInputs") or [])
        missing_fields = list(step_progress.get("missingAnswerFields") or [])
        if missing_inputs or missing_fields:
            details: list[str] = []
            if missing_inputs:
                details.append("missing inputs: " + ", ".join(missing_inputs[:6]))
            if missing_fields:
                details.append("missing answer fields: " + ", ".join(missing_fields[:6]))
            message = "; ".join(details) if details else "Multi-step task is incomplete."
            violations.append(("multistep_incomplete", message))
            return violations

        lowered_final = str(final_text or "").lower()
        facts = evidence_map.get("facts", {}) if isinstance(evidence_map.get("facts"), dict) else {}
        for field_name in contract.required_answer_fields:
            if field_name == "checkpoint" and facts.get("checkpoint") and str(facts["checkpoint"]).lower() not in lowered_final:
                violations.append(("multistep_final_synthesis_weak", "Final answer omitted the verified checkpoint."))
            elif field_name == "product_name" and facts.get("product_name") and str(facts["product_name"]).lower() not in lowered_final:
                violations.append(("multistep_final_synthesis_weak", "Final answer omitted the verified product name."))
            elif field_name == "preferred_editor" and facts.get("preferred_editor") and str(facts["preferred_editor"]).lower() not in lowered_final:
                violations.append(("multistep_final_synthesis_weak", "Final answer omitted the verified preferred editor."))
            elif field_name == "stack":
                stack_terms = [str(item).lower() for item in facts.get("stack_terms", []) if str(item).strip()]
                if stack_terms and not any(term in lowered_final for term in stack_terms):
                    violations.append(("multistep_final_synthesis_weak", "Final answer omitted the verified stack signals."))
            elif field_name == "extra_file":
                extra_files = [str(item).lower() for item in facts.get("extra_files", []) if str(item).strip()]
                if extra_files and not any(term in lowered_final for term in extra_files[:6]):
                    violations.append(("multistep_final_synthesis_weak", "Final answer omitted the verified extra file evidence."))
        if final_text.strip() and str(state.get("final_text") or "").strip() == final_text.strip() and violations:
            violations.append(("multistep_stale_answer", "Final answer reused stale wording instead of synthesizing the accumulated evidence."))
        return violations

    def _next_missing_input_instruction(self, missing_input: str) -> str | None:
        kind, _, target = str(missing_input or "").partition(":")
        target = normalize_whitespace(target)
        if not target:
            return None
        if kind == "file":
            return f"Recommended next step: call read_file with path `{target}` now."
        if kind == "directory":
            return f"Recommended next step: call list_directory with path `{target}` now."
        return None

    def _build_multistep_continue_prompt(self, state: AgentState, violations: list[tuple[str, str]]) -> str:
        step_progress = dict(state.get("step_progress") or {})
        assessment = self._refresh_goal_tracking(state, str(state.get("final_text") or ""), violations)
        lines = [
            "Continue only if the goal gap is still open.",
            "Use the evidence already gathered and take one next useful step only.",
        ]
        if self._provider_requires_sequential_tools(state):
            lines.append("If the gap remains open, emit at most one next tool call in your next response.")
        missing_inputs = list(assessment.gap.missing_evidence or step_progress.get("missingInputs") or [])
        missing_fields = list(assessment.gap.missing_facts or step_progress.get("missingAnswerFields") or [])
        if missing_inputs:
            lines.append("Missing evidence: " + ", ".join(missing_inputs[:6]))
            next_input_instruction = self._next_missing_input_instruction(missing_inputs[0])
            if next_input_instruction:
                lines.append(next_input_instruction)
        if missing_fields:
            lines.append("Missing answer fields: " + ", ".join(missing_fields[:6]))
        return "\n".join(lines)

    def _parse_missing_input_target(self, missing_input: str) -> tuple[str, str] | None:
        kind, _, target = str(missing_input or "").partition(":")
        kind = normalize_whitespace(kind)
        target = normalize_whitespace(target)
        if kind not in {"file", "directory"} or not target:
            return None
        return kind, target

    def _explicit_evidence_acquisition_tool(self, missing_input: str) -> tuple[str, dict[str, Any]] | None:
        parsed = self._parse_missing_input_target(missing_input)
        if not parsed:
            return None
        kind, target = parsed
        if kind == "file":
            return "read_file", {"path": target}
        if kind == "directory":
            return "list_directory", {"path": target}
        return None

    def _has_runtime_actionable_missing_evidence(self, missing_evidence: list[str]) -> bool:
        return any(self._explicit_evidence_acquisition_tool(item) is not None for item in missing_evidence)

    def _run_explicit_evidence_acquisition(
        self,
        state: AgentState,
        missing_input: str,
    ) -> tuple[AgentState, list[dict[str, Any]], bool]:
        plan = self._explicit_evidence_acquisition_tool(missing_input)
        if not plan:
            return state, [], False
        tool_name, tool_args = plan
        registry = self.deps.tool_registry_factory(state["workspace_root"], state["session_id"], state["run_id"], state.get("tool_toggles"))
        registered = registry.get(tool_name)
        if not registered or not registered.enabled:
            return state, [], False

        tool_id = str(uuid.uuid4())
        tool_events: list[dict[str, Any]] = [
            {
                "type": "tool_call",
                "runId": state["run_id"],
                "actionId": tool_id,
                "name": tool_name,
                "arguments": json.dumps(tool_args, ensure_ascii=False),
                "riskLevel": registered.risk_level,
            },
            run_diagnostic(
                state["run_id"],
                "explicit_evidence_acquisition",
                f"Runtime acquired missing evidence via {tool_name} for {missing_input}.",
            ),
        ]
        try:
            result = str(registered.tool.invoke(tool_args))
            result_events, message_result, _ok, tool_meta = self._materialize_tool_result(state["run_id"], tool_name, tool_id, result)
            tool_events.extend(result_events)
            if state["provider_mode"] == "textual_replay":
                tool_message: BaseMessage = build_tool_replay_message(tool_name, tool_args, message_result, "succeeded")
            else:
                tool_message = ToolMessage(content=message_result, tool_call_id=tool_id)
            followup_guidance = self._build_tool_followup_guidance(state, tool_name, message_result, "succeeded", tool_meta.get("meta") if tool_meta else None)
            evidence_map = self._update_evidence_map(state, tool_name, tool_args, message_result, tool_meta)
            updated_state: AgentState = {
                **state,
                "messages": list(state["messages"]) + [tool_message, followup_guidance],
                "tool_events": list(state.get("tool_events", [])) + tool_events,
                "tool_results": list(state.get("tool_results", []))
                + [
                    {
                        "tool": tool_name,
                        "result": message_result,
                        "status": "succeeded",
                        **(
                            {
                                "meta": {**(tool_meta.get("meta") or {}), "hits": tool_meta.get("hits", [])},
                                "lookupStatus": tool_meta.get("status"),
                            }
                            if tool_meta
                            else {}
                        ),
                    }
                ],
                "used_tool_names": list(state.get("used_tool_names", [])) + [tool_name],
                "evidence_map": evidence_map,
            }
            self._note_tool_success(updated_state, tool_name, tool_args, message_result, tool_meta)
            post_tool_hint = self._maybe_build_post_tool_reasoning_hint(updated_state, tool_name, message_result, tool_meta)
            if post_tool_hint:
                updated_state["messages"] = list(updated_state["messages"]) + [SystemMessage(content=post_tool_hint)]
            return updated_state, tool_events, True
        except Exception as exc:
            failure_events, message_result = self._tool_failure_result(state["run_id"], tool_name, tool_id, exc)
            tool_events.extend(failure_events)
            failed_state: AgentState = {
                **state,
                "tool_events": list(state.get("tool_events", [])) + tool_events,
                "tool_results": list(state.get("tool_results", [])) + [{"tool": tool_name, "result": message_result, "status": "failed"}],
                "used_tool_names": list(state.get("used_tool_names", [])) + [tool_name],
            }
            self._note_executed_tool(failed_state, tool_name)
            return failed_state, tool_events, False

    async def _continue_multistep_if_needed(self, run_id: str, state: AgentState, detail: str) -> AgentState:
        contract = MultiStepContract.from_payload(state.get("multi_step_contract"))
        assessment = self._refresh_goal_tracking(state, str(state.get("final_text") or ""))
        if assessment.action != "gather_more_evidence":
            return state

        while True:
            step_progress = self._compute_step_progress(contract, dict(state.get("evidence_map") or {}))
            state["step_progress"] = step_progress
            assessment = self._refresh_goal_tracking(state, str(state.get("final_text") or ""))
            missing_inputs = list(assessment.gap.missing_evidence or [])
            missing_fields = list(assessment.gap.missing_facts or [])
            if not missing_inputs:
                return state

            if state.get("multistep_repair_attempted") and int(state.get("multistep_no_progress_turns", 0) or 0) > 0:
                state["tool_events"] = list(state.get("tool_events", [])) + [
                    *self._goal_gap_events(run_id, state, "continuation_exhausted"),
                    *(
                        self._build_multistep_progress_events(
                            run_id,
                            contract,
                            step_progress,
                            dict(state.get("evidence_map") or {}),
                            reason="continuation_exhausted",
                        )
                        if is_multistep_contract_active(contract)
                        else []
                    ),
                    run_diagnostic(
                        run_id,
                        "multistep_missing_evidence",
                        "Task completion is still unverified because the missing evidence was not gathered.",
                        level="warn",
                        data={
                            "reason": "continuation_exhausted",
                            "missingInputs": missing_inputs,
                            "missingAnswerFields": missing_fields,
                        },
                    ),
                ]
                return state

            prior_signature = str((state.get("evidence_map") or {}).get("signature") or "")
            acquired_state = state
            acquired_state, _events, acquired = self._run_explicit_evidence_acquisition(state, missing_inputs[0])
            if acquired:
                acquired_state["multistep_repair_attempted"] = True
                result = None
                async for event in self._run_graph_with_heartbeats(run_id, acquired_state, detail=detail):
                    if event.get("type") == "_graph_result":
                        result = event["result"]
                if result is None:
                    return acquired_state
                updated_signature = str((result.get("evidence_map") or {}).get("signature") or "")
                if updated_signature == prior_signature:
                    result["multistep_no_progress_turns"] = int(state.get("multistep_no_progress_turns", 0) or 0) + 1
                    result["multistep_repair_attempted"] = True
                    result["tool_events"] = list(result.get("tool_events", [])) + [
                        run_diagnostic(
                            run_id,
                            "multistep_no_progress_reason",
                            "Continuation produced no new evidence.",
                            level="warn",
                            data={
                                "reason": "no_new_evidence",
                                "noProgressTurns": int(result.get("multistep_no_progress_turns", 0) or 0),
                            },
                        ),
                    ]
                else:
                    result["multistep_no_progress_turns"] = 0
                    result["multistep_repair_attempted"] = False
                state = result
                continue

            prompt = self._build_multistep_continue_prompt(state, [("task_completion_unverified", "Task completion is not yet verified.")])
            state["messages"] = list(state["messages"]) + [SystemMessage(content=prompt)]
            state["final_text"] = ""
            state["multistep_repair_attempted"] = True
            state["tool_events"] = list(state.get("tool_events", [])) + [
                *self._goal_gap_events(run_id, state, "verify_guardrail_continue"),
                *(
                    self._build_multistep_progress_events(
                        run_id,
                        contract,
                        step_progress,
                        dict(state.get("evidence_map") or {}),
                        reason="verify_guardrail_continue",
                    )
                    if is_multistep_contract_active(contract)
                    else []
                ),
                run_diagnostic(
                    run_id,
                    "task_completion_unverified",
                    "; ".join(
                        part
                        for part in [
                            ("missing inputs: " + ", ".join(missing_inputs[:6])) if missing_inputs else "",
                            ("missing answer fields: " + ", ".join(missing_fields[:6])) if missing_fields else "",
                        ]
                        if part
                    ) or "Task completion is not yet verified.",
                    level="warn",
                ),
                run_diagnostic(
                    run_id,
                    "verify_guardrail_repair",
                    "Task completion is not yet verified. Continuing with one more execution turn.",
                    level="warn",
                ),
            ]

            result = None
            async for event in self._run_graph_with_heartbeats(run_id, state, detail=detail):
                if event.get("type") == "_graph_result":
                    result = event["result"]
            if result is None:
                return state

            updated_signature = str((result.get("evidence_map") or {}).get("signature") or "")
            if updated_signature == prior_signature:
                result["multistep_no_progress_turns"] = int(state.get("multistep_no_progress_turns", 0) or 0) + 1
                result["tool_events"] = list(result.get("tool_events", [])) + [
                    run_diagnostic(
                        run_id,
                        "multistep_no_progress_reason",
                        "Continuation produced no new evidence.",
                        level="warn",
                        data={
                            "reason": "no_new_evidence",
                            "noProgressTurns": int(result.get("multistep_no_progress_turns", 0) or 0),
                        },
                    ),
                ]
                return result

            result["multistep_no_progress_turns"] = 0
            state = result

    def _looks_like_pseudo_tool_json(self, text: str) -> bool:
        trimmed = str(text or "").strip()
        if not (trimmed.startswith("{") and trimmed.endswith("}")):
            return False
        try:
            parsed = json.loads(trimmed)
        except Exception:
            return False
        return isinstance(parsed, dict) and isinstance(parsed.get("name"), str) and "arguments" in parsed

    def _looks_like_stale_final_answer(self, state: AgentState, final_text: str) -> bool:
        normalized = normalize_whitespace(final_text)
        if not normalized:
            return False
        contract = FinalAnswerContract.from_payload(state.get("final_contract"))
        if contract.exact_output_text and normalize_exact_output(normalized) == normalize_exact_output(contract.exact_output_text):
            return False
        if self._is_followup_grounded_turn(state):
            return False
        if "(empty response)" in normalized.lower():
            return True
        if self._looks_like_pseudo_tool_json(normalized):
            return True
        previous = normalize_whitespace(self._previous_assistant_message(state))
        if previous and normalized == previous:
            return True
        if previous and previous in normalized and normalized != previous:
            return True
        sentences = [item.strip() for item in re.split(r"(?<=[.!?])\s+", normalized) if item.strip()]
        lowered_sentences = [item.lower() for item in sentences]
        if len(lowered_sentences) != len(set(lowered_sentences)):
            return True
        if normalized.count("(empty response)") > 0:
            return True
        return False

    def _detect_action_claim_mismatch(self, state: AgentState, final_text: str) -> tuple[str, str] | None:
        text = str(final_text or "")
        lowered = text.lower()
        executed = set(state.get("executed_tools", [])) | set(state.get("used_tool_names", []))
        claims_open_url = bool(re.search(r"(url|link|site|https?://)", text, flags=re.IGNORECASE) and re.search(r"\b(opened|open|opened successfully|ouvert|ouverte)\b", text, flags=re.IGNORECASE))
        claims_open_file = bool(re.search(r"\b(file|fichier)\b", text, flags=re.IGNORECASE) and re.search(r"\b(opened|open|ouvert|ouverte)\b", text, flags=re.IGNORECASE))
        if claims_open_url and not executed.intersection({"open_url", "open_resource"}):
            return (
                "action_claim_without_tool_open_url",
                "Final answer claims a URL was opened, but no open_url/open_resource tool succeeded in this run.",
            )
        if claims_open_file and not executed.intersection({"open_file", "open_resource"}):
            return (
                "action_claim_without_tool_open_file",
                "Final answer claims a file was opened, but no open_file/open_resource tool succeeded in this run.",
            )
        return None

    def _detect_task_completion_unverified(self, state: AgentState, final_text: str) -> tuple[str, str] | None:
        contract = MultiStepContract.from_payload(state.get("multi_step_contract"))
        if not is_multistep_contract_active(contract):
            return None
        step_progress = self._compute_step_progress(contract, dict(state.get("evidence_map") or {}))
        missing_inputs = list(step_progress.get("missingInputs") or [])
        missing_fields = list(step_progress.get("missingAnswerFields") or [])
        if not missing_inputs and not missing_fields:
            return None
        if not re.search(r"\b(done|completed|ready|finished|voici|here|summary|résumé|result|résultat|is|are)\b", str(final_text or ""), flags=re.IGNORECASE):
            return None
        parts: list[str] = []
        if missing_inputs:
            parts.append("missing inputs: " + ", ".join(missing_inputs[:6]))
        if missing_fields:
            parts.append("missing answer fields: " + ", ".join(missing_fields[:6]))
        return ("task_completion_unverified", "; ".join(parts))

    def _build_run_metrics_event(self, state: AgentState) -> dict[str, Any]:
        return run_diagnostic(
            state["run_id"],
            "run_metrics",
            "Runtime metrics snapshot.",
            data={
                "toolCalls": len(state.get("tool_results", [])),
                "toolSuccesses": sum(1 for item in state.get("tool_results", []) if item.get("status") == "succeeded"),
                "toolFailures": sum(1 for item in state.get("tool_results", []) if item.get("status") != "succeeded"),
                "repairs": int(state.get("repair_attempts", 0) or 0),
                "noProgressTurns": int(state.get("no_progress_turns", 0) or 0),
                "executedTools": list(dict.fromkeys(state.get("executed_tools", [])))[-20:],
            },
        )

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _build_initial_run_trace(self, state: AgentState) -> dict[str, Any]:
        goal_summary = dict(state.get("goal_summary") or {})
        goal_text = normalize_whitespace(str(goal_summary.get("userIntent") or self._last_user_message(state.get("messages", [])) or ""))
        return {
            "run_id": state.get("run_id"),
            "session_id": state.get("session_id"),
            "started_at": self._now_iso(),
            "ended_at": None,
            "goal": goal_text,
            "steps": [],
            "outcome": None,
            "failure": None,
            "_event_cursor": 0,
        }

    def _ensure_run_trace(self, state: AgentState) -> dict[str, Any]:
        trace = state.get("run_trace")
        if not isinstance(trace, dict):
            trace = self._build_initial_run_trace(state)
            state["run_trace"] = trace
        trace.setdefault("run_id", state.get("run_id"))
        trace.setdefault("session_id", state.get("session_id"))
        trace.setdefault("started_at", self._now_iso())
        trace.setdefault("goal", normalize_whitespace(str((state.get("goal_summary") or {}).get("userIntent") or self._last_user_message(state.get("messages", [])) or "")))
        trace.setdefault("steps", [])
        trace.setdefault("outcome", None)
        trace.setdefault("failure", None)
        trace.setdefault("_event_cursor", 0)
        return trace

    def _append_run_trace_step(
        self,
        state: AgentState,
        *,
        kind: str,
        status: str,
        summary: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        trace = self._ensure_run_trace(state)
        steps = trace.get("steps")
        if not isinstance(steps, list):
            steps = []
            trace["steps"] = steps
        step = {
            "index": len(steps) + 1,
            "kind": kind,
            "status": status,
            "summary": normalize_whitespace(summary),
            "timestamp": self._now_iso(),
        }
        if data:
            step["data"] = data
        steps.append(step)

    def _append_trace_steps_from_tool_events(self, state: AgentState) -> None:
        trace = self._ensure_run_trace(state)
        cursor = int(trace.get("_event_cursor", 0) or 0)
        tool_events = list(state.get("tool_events", []))
        if cursor >= len(tool_events):
            return
        for event in tool_events[cursor:]:
            event_type = str(event.get("type") or "")
            if event_type == "tool_call":
                tool_name = str(event.get("name") or "tool")
                self._append_run_trace_step(
                    state,
                    kind="tool_call",
                    status="pending",
                    summary=f"Call tool `{tool_name}`.",
                    data={"tool": tool_name, "actionId": event.get("actionId")},
                )
            elif event_type == "tool_result":
                tool_name = str(event.get("name") or "tool")
                ok = bool(event.get("ok"))
                self._append_run_trace_step(
                    state,
                    kind="tool_result",
                    status="ok" if ok else "failed",
                    summary=f"Tool `{tool_name}` {'succeeded' if ok else 'failed'}.",
                    data={"tool": tool_name, "ok": ok},
                )
            elif event_type == "run_diagnostic":
                code = str(event.get("code") or "")
                message = str(event.get("message") or code or "diagnostic")
                if code in {"goal_gap_summary", "task_completion_unverified"}:
                    self._append_run_trace_step(
                        state,
                        kind="verify",
                        status="warn" if code == "task_completion_unverified" else "ok",
                        summary=message,
                        data={"code": code, "data": event.get("data")},
                    )
                elif code in {"final_contract_repair_requested", "verify_guardrail_repair", "verify_guardrail_repeated", "final_contract_repair_failed"}:
                    self._append_run_trace_step(
                        state,
                        kind="repair",
                        status="warn",
                        summary=message,
                        data={"code": code},
                    )
        trace["_event_cursor"] = len(tool_events)

    def _failure_defaults(self, code: str) -> tuple[str, str]:
        defaults: dict[str, tuple[str, str]] = {
            "missing_evidence": ("Missing required evidence to complete the goal.", "gather_missing_evidence"),
            "verification_failed": ("Final verification failed.", "repair_from_existing_evidence"),
            "tool_failed": ("A required tool action failed.", "retry_or_switch_tool"),
            "approval_blocked": ("Execution is blocked pending user approval.", "provide_approval_decision"),
            "clarification_blocked": ("Execution is blocked pending user clarification.", "provide_clarification_answer"),
            "no_progress_limit": ("No-progress guardrail stopped the run.", "stop_and_report_partial"),
            "invalid_final_answer": ("Final answer is invalid or ambiguous.", "repair_final_answer"),
            "runtime_exception": ("Runtime raised an exception.", "retry_run"),
        }
        return defaults.get(code, defaults["verification_failed"])

    def _build_failure_payload(
        self,
        code: str,
        *,
        message: str | None = None,
        missing_evidence: list[str] | None = None,
        context: dict[str, Any] | None = None,
        next_action: str | None = None,
    ) -> dict[str, Any]:
        normalized_code = code if code in FAILURE_CODE_SET else "verification_failed"
        default_message, default_next_action = self._failure_defaults(normalized_code)
        payload: dict[str, Any] = {
            "code": normalized_code,
            "message": normalize_whitespace(message or default_message),
            "missing_evidence": list(missing_evidence or []),
            "next_action": next_action or default_next_action,
        }
        if context:
            payload["context"] = dict(context)
        return payload

    def _failure_from_goal_assessment(
        self,
        state: AgentState,
        assessment: GoalAssessment,
        violations: list[tuple[str, str]],
        final_text: str,
    ) -> dict[str, Any]:
        violation_codes = {code for code, _message in violations}
        if not normalize_whitespace(final_text) or {"empty_final_answer", "invalid_final_answer"}.intersection(violation_codes):
            message = "; ".join(message for _code, message in violations if _code in {"empty_final_answer", "invalid_final_answer"}) or None
            return self._build_failure_payload("invalid_final_answer", message=message)
        if assessment.gap.missing_evidence or assessment.gap.missing_facts or "task_completion_unverified" in violation_codes:
            missing = _dedupe_strings([*assessment.gap.missing_evidence, *assessment.gap.missing_facts])
            return self._build_failure_payload(
                "missing_evidence",
                message="Goal gap is still open: missing verified evidence.",
                missing_evidence=missing,
            )
        if int(state.get("no_progress_turns", 0) or 0) >= MAX_NO_PROGRESS_TURNS:
            return self._build_failure_payload("no_progress_limit")
        if any(code in {"tool_execution_error", "terminal_tool_error"} for code, _message in violations):
            return self._build_failure_payload("tool_failed")
        message = "; ".join(message for _code, message in violations[:4]) if violations else None
        return self._build_failure_payload("verification_failed", message=message)

    def _build_failure_final_text(self, failure: dict[str, Any]) -> str:
        code = str(failure.get("code") or "verification_failed")
        message = str(failure.get("message") or "Final verification failed.")
        missing = [str(item) for item in failure.get("missing_evidence", []) if str(item).strip()]
        next_action = str(failure.get("next_action") or "")
        parts = [f"Run failed (`{code}`): {message}"]
        if missing:
            parts.append("Missing evidence: " + ", ".join(missing[:6]) + ".")
        if next_action:
            parts.append(f"Next action: {next_action}.")
        return " ".join(parts)

    def _finalize_run_trace(self, state: AgentState, *, outcome: str, failure: dict[str, Any] | None = None) -> dict[str, Any]:
        trace = self._ensure_run_trace(state)
        trace["ended_at"] = self._now_iso()
        trace["outcome"] = "completed" if outcome == "completed" else "failed"
        trace["failure"] = None if outcome == "completed" else dict(failure or self._build_failure_payload("verification_failed"))
        return trace

    def _assert_terminal_consistency(self, trace: dict[str, Any]) -> tuple[bool, str]:
        outcome = str(trace.get("outcome") or "")
        failure = trace.get("failure")
        steps = trace.get("steps")
        if outcome not in {"completed", "failed"}:
            return False, "Trace outcome is missing or invalid."
        if not isinstance(steps, list) or not steps:
            return False, "Trace has no steps."
        if outcome == "completed" and failure:
            return False, "Completed trace should not include failure payload."
        if outcome == "failed":
            if not isinstance(failure, dict) or str(failure.get("code") or "") not in FAILURE_CODE_SET:
                return False, "Failed trace must include a valid failure code."
        return True, "ok"

    def _run_trace_summary(self, trace: dict[str, Any]) -> dict[str, Any]:
        failure = trace.get("failure") if isinstance(trace.get("failure"), dict) else None
        summary = {
            "outcome": trace.get("outcome"),
            "step_count": len(trace.get("steps", [])) if isinstance(trace.get("steps"), list) else 0,
        }
        if failure:
            summary["failure"] = {"code": failure.get("code"), "next_action": failure.get("next_action")}
        return summary

    def _is_blocking_violation(self, code: str) -> bool:
        return code in {
            "empty_final_answer",
            "invalid_final_answer",
            "action_claim_without_tool_open_url",
            "action_claim_without_tool_open_file",
            "sources_missing_in_final_answer",
            "rag_citations_missing",
            "rag_status_not_disclosed",
            "rag_grounding_weak",
            "missing_required_evidence",
            "missing_required_facts",
            "answer_ignores_strong_evidence",
            "task_completion_unverified",
            "exact_output_mismatch",
            "wrong_response_language",
        }

    def _evaluate_goal_completion(self, state: AgentState, final_text: str) -> GoalAssessment:
        contract = FinalAnswerContract.from_payload(state.get("final_contract"))
        multi_step_contract = MultiStepContract.from_payload(state.get("multi_step_contract"))
        violations: list[tuple[str, str]] = []
        normalized_final = normalize_whitespace(final_text)
        if not normalized_final:
            violations.append(("empty_final_answer", "Model returned an empty final answer."))
        elif self._looks_like_stale_final_answer(state, final_text):
            violations.append(("invalid_final_answer", "Final answer is stale, polluted by intermediate output, or looks like tool JSON."))

        action_claim = self._detect_action_claim_mismatch(state, final_text)
        if action_claim:
            violations.append(action_claim)

        answer_language = detect_message_language(final_text)
        if contract.requested_language and answer_language and answer_language != contract.requested_language:
            violations.append(
                (
                    "wrong_response_language",
                    f"Expected {describe_language(contract.requested_language)}, got {describe_language(answer_language)}.",
                )
            )

        if contract.exact_output_text and normalized_final and normalize_exact_output(final_text) != normalize_exact_output(contract.exact_output_text):
            violations.append(("exact_output_mismatch", f"Expected exact output `{contract.exact_output_text}`."))

        used_tools = set(state.get("used_tool_names", []))
        urls = extract_urls(final_text)
        requires_sources = contract.require_sources or "web_search" in used_tools
        if requires_sources and len(urls) < 2:
            violations.append(("sources_missing_in_final_answer", "Final answer should include a Sources section with at least two source links."))

        rag_result = self._last_rag_tool_result(state)
        rag_meta = rag_result.get("meta", {}) if rag_result and isinstance(rag_result.get("meta"), dict) else {}
        rag_used = "rag_lookup" in used_tools and rag_result is not None
        if rag_used:
            strong_hits = int(rag_meta.get("strongHitCount", 0) or 0)
            requires_rag_citations = contract.require_citations_when_rag_used or strong_hits > 0
            if requires_rag_citations and not self._final_text_has_rag_citation(final_text, rag_meta):
                violations.append(("rag_citations_missing", "Final answer should include compact document citations for the RAG evidence used."))
            if strong_hits <= 0:
                if not self._is_cautious_answer(final_text):
                    violations.append(("rag_grounding_weak", "RAG hits are weak; the final answer should stay cautious and avoid unsupported claims."))
            else:
                lowered_final = normalized_final.lower()
                grounding_terms = self._extract_grounding_terms(state)
                if grounding_terms and not any(term in lowered_final for term in grounding_terms):
                    violations.append(("rag_grounding_weak", "Final answer is not clearly grounded in the retrieved RAG evidence."))
                evidence = self._best_rag_evidence(state)
                salient_terms = evidence.get("salientTerms", []) if evidence else []
                if salient_terms and not any(term in lowered_final for term in salient_terms):
                    violations.append(("rag_grounding_weak", "Final answer does not preserve the salient wording of the strongest retrieved RAG evidence."))

        requires_grounding = contract.require_grounding and bool(used_tools.intersection(GROUNDING_TOOL_NAMES))
        if requires_grounding and not rag_used:
            lowered_final = normalized_final.lower()
            grounding_terms = self._extract_grounding_terms(state)
            preferred_workspace_label = self._preferred_workspace_label(state)
            read_only_grounding = bool(used_tools) and set(used_tools).issubset({"read_file", "open_file", "open_resource"})
            fact_grounded = self._final_text_matches_evidence_facts(state, final_text)
            workspace_label_missing = bool(
                preferred_workspace_label
                and preferred_workspace_label.lower() not in lowered_final
                and bool(used_tools.intersection({"run_terminal", "terminal_wait_for_output", "terminal_snapshot"}))
            )
            if ((grounding_terms and not any(term in lowered_final for term in grounding_terms) and not (read_only_grounding and fact_grounded)) or workspace_label_missing):
                violations.append(("final_contract_repair_requested", "Final answer is not clearly grounded in the tool result."))
            directory_result = next((item for item in reversed(state.get("tool_results", [])) if item.get("tool") == "list_directory"), None)
            directory_details = self._directory_project_details(directory_result) if directory_result else None
            prompt = str(self._last_user_message(state.get("messages", [])) or "").lower()
            if directory_details and "project" in prompt:
                required_terms = directory_details.get("preferredTerms") or directory_details.get("salientTerms", [])
                if required_terms and not any(term in lowered_final for term in required_terms):
                    violations.append(("final_contract_repair_requested", "Project summary should preserve at least one salient descriptor from the directory evidence."))
        elif contract.require_grounding and self._is_followup_grounded_turn(state):
            previous_assistant = self._previous_assistant_message(state)
            preferred_workspace_label = self._workspace_label_from_text(previous_assistant) or self._preferred_workspace_label(state)
            final_lowered = normalized_final.lower()
            expected_terms = self._followup_grounding_terms_from_text(previous_assistant)
            missing_workspace_label = bool(preferred_workspace_label and preferred_workspace_label.lower() not in final_lowered)
            missing_followup_terms = bool(expected_terms and not any(term in final_lowered for term in expected_terms))
            if missing_workspace_label or missing_followup_terms:
                violations.append(("final_contract_repair_requested", "Follow-up summary should stay grounded in the previous verified assistant context."))

        assessment = self._refresh_goal_tracking(state, final_text, violations)
        if is_multistep_contract_active(multi_step_contract) and (assessment.gap.missing_evidence or assessment.gap.missing_facts):
            missing_parts: list[str] = []
            if assessment.gap.missing_evidence:
                missing_parts.append("missing evidence: " + ", ".join(assessment.gap.missing_evidence[:6]))
            if assessment.gap.missing_facts:
                missing_parts.append("missing facts: " + ", ".join(assessment.gap.missing_facts[:6]))
            assessment.mismatch_codes.append(("task_completion_unverified", "; ".join(missing_parts)))
            if assessment.gap.answer_defects is not None:
                assessment.gap.answer_defects = _dedupe_strings([*assessment.gap.answer_defects, "task_completion_unverified"])
            assessment.action = "gather_more_evidence" if assessment.gap.missing_evidence and assessment.gap.can_gather_more_evidence else "repair_answer"
            state["goal_gap_summary"] = assessment.gap.to_payload()
        elif is_multistep_contract_active(multi_step_contract) and assessment.gap.is_complete and not violations:
            state["goal_gap_summary"] = assessment.gap.to_payload()
            state["tool_events"] = list(state.get("tool_events", [])) + [
                run_diagnostic(
                    state["run_id"],
                    "finished_after_sufficient_evidence",
                    "Verified evidence fully covers the goal, so the run is complete.",
                    level="info",
                    data=dict(state.get("goal_gap_summary") or {}),
                )
            ]
            required_inputs = len(multi_step_contract.required_inputs)
            tool_result_count = len(state.get("tool_results", []))
            if tool_result_count > max(required_inputs, 0):
                state["tool_events"] = list(state.get("tool_events", [])) + [
                    run_diagnostic(
                        state["run_id"],
                        "overcontinued_without_need",
                        "The goal was already covered, but the run used more tool steps than the explicit evidence needs.",
                        level="info",
                        data={
                            **dict(state.get("goal_gap_summary") or {}),
                            "toolResultCount": tool_result_count,
                            "requiredInputCount": required_inputs,
                        },
                    )
                ]
        return assessment

    def _collect_contract_violations(self, state: AgentState, final_text: str) -> list[tuple[str, str]]:
        assessment = self._evaluate_goal_completion(state, final_text)
        return list(assessment.mismatch_codes)

    def _build_final_repair_prompt(self, state: AgentState, violations: list[tuple[str, str]]) -> str:
        contract = FinalAnswerContract.from_payload(state.get("final_contract"))
        assessment = self._refresh_goal_tracking(state, str(state.get("final_text") or ""), violations)
        lines = [
            "Final answer repair only.",
            "Do not call any tools. Use only the verified tool outputs already in the conversation.",
            "Fix the final answer so it satisfies the goal and the verified evidence exactly.",
            "Do not repeat stale fragments from earlier assistant replies.",
            "Do not include '(empty response)' anywhere.",
        ]
        if assessment.goal.user_intent:
            lines.append(f"Objective: {assessment.goal.user_intent}")
        if assessment.realization.proven_facts:
            lines.append("Current realization from verified evidence:")
            for key, value in assessment.realization.proven_facts.items():
                lines.append(f"- {key}: {value}")
        if assessment.gap.missing_evidence or assessment.gap.missing_facts or assessment.gap.answer_defects:
            lines.append("Goal gap to close:")
            if assessment.gap.missing_evidence:
                lines.append("- missing evidence: " + ", ".join(assessment.gap.missing_evidence[:6]))
            if assessment.gap.missing_facts:
                lines.append("- missing facts: " + ", ".join(assessment.gap.missing_facts[:6]))
            if assessment.gap.answer_defects:
                lines.append("- answer defects: " + ", ".join(assessment.gap.answer_defects[:6]))
        if contract.requested_language:
            lines.append(f"Answer in {describe_language(contract.requested_language)}.")
        if contract.exact_output_text:
            lines.append(f"Return exactly this text and nothing else: {contract.exact_output_text}")
        if contract.require_sources or "web_search" in set(state.get("used_tool_names", [])):
            lines.append("Add a separate Sources section with at least two Markdown links like [Title](https://...).")
            lines.append("Keep URLs out of the main body unless the user explicitly asks to display raw links.")
        if contract.require_grounding and set(state.get("used_tool_names", [])).intersection(GROUNDING_TOOL_NAMES):
            lines.append("Keep the answer grounded in the tool evidence from this run.")
        elif contract.require_grounding and self._is_followup_grounded_turn(state):
            previous_assistant = self._previous_assistant_message(state)
            if previous_assistant:
                lines.append("Keep the answer grounded in the previous verified assistant answer from this same conversation.")
                lines.append("Verified conversation context excerpt:")
                lines.append(previous_assistant[-1200:])
        rag_result = self._last_rag_tool_result(state)
        rag_meta = rag_result.get("meta", {}) if rag_result and isinstance(rag_result.get("meta"), dict) else {}
        if any(code == "task_completion_unverified" for code, _message in violations):
            lines.append("If the verified evidence is still incomplete, answer honestly about what is missing instead of pretending the task is complete.")
        if "rag_lookup" in set(state.get("used_tool_names", [])):
            if contract.require_citations_when_rag_used or int(rag_meta.get("strongHitCount", 0)) > 0:
                lines.append("Include compact document citations from the retrieved RAG hits.")
            if int(rag_meta.get("strongHitCount", 0)) <= 0:
                lines.append("If the retrieved RAG hits are weak, say that the documents are insufficient instead of guessing.")
        lines.append("Violations to repair:")
        lines.extend(f"- {message}" for _code, message in violations)
        return "\n".join(lines)

    def _synthesize_fallback_final_answer(self, state: AgentState) -> str | None:
        contract = FinalAnswerContract.from_payload(state.get("final_contract"))
        multi_step_contract = MultiStepContract.from_payload(state.get("multi_step_contract"))
        evidence_map = dict(state.get("evidence_map") or {})
        facts = evidence_map.get("facts", {}) if isinstance(evidence_map.get("facts"), dict) else {}
        if contract.exact_output_text:
            return contract.exact_output_text

        if is_multistep_contract_active(multi_step_contract) and multi_step_contract.required_answer_fields:
            parts: list[str] = []
            product_name = str(facts.get("product_name") or "").strip()
            checkpoint = str(facts.get("checkpoint") or "").strip()
            preferred_editor = str(facts.get("preferred_editor") or "").strip()
            stack_terms = [str(item).strip() for item in facts.get("stack_terms", []) if str(item).strip()]
            extra_files = [str(item).strip() for item in facts.get("extra_files", []) if str(item).strip()]
            if product_name and "product_name" in multi_step_contract.required_answer_fields:
                parts.append(product_name)
            if checkpoint and "checkpoint" in multi_step_contract.required_answer_fields:
                parts.append(checkpoint)
            if stack_terms and "stack" in multi_step_contract.required_answer_fields:
                parts.append(" / ".join(stack_terms[:3]))
            if preferred_editor and "preferred_editor" in multi_step_contract.required_answer_fields:
                parts.append(preferred_editor)
            if extra_files and "extra_file" in multi_step_contract.required_answer_fields:
                parts.append(extra_files[0])
            if parts:
                return ". ".join(parts) + "."

        tool_results = list(state.get("tool_results", []))
        web_result = next((item for item in reversed(tool_results) if item.get("tool") == "web_search"), None)
        if web_result:
            result_text = str(web_result.get("result") or "")
            markdown_matches = re.findall(r"-\s+\[([^\]]+)\]\((https?://[^)]+)\)", result_text)
            colon_matches = re.findall(r"-\s+(.+?):\s+(https?://\S+)", result_text)
            matches = markdown_matches or colon_matches
            if len(matches) >= 2:
                first_title, first_url = matches[0]
                second_title, second_url = matches[1]
                return (
                    f"Latest OpenAI coverage includes {first_title} and {second_title}.\n\n"
                    "Sources:\n"
                    f"- [{first_title}]({first_url})\n"
                    f"- [{second_title}]({second_url})"
                )

        terminal_outputs = [item for item in tool_results if item.get("tool") == "run_terminal"]
        if terminal_outputs and contract.require_grounding:
            pwd_output = ""
            ls_output = ""
            for item in terminal_outputs:
                result = str(item.get("result") or "")
                if "Command: pwd" in result and "Output:" in result:
                    pwd_output = result.split("Output:", 1)[-1].strip()
                if "Command: ls" in result and "Output:" in result:
                    ls_output = result.split("Output:", 1)[-1].strip()
            if pwd_output or ls_output:
                entries = [line.strip() for line in ls_output.splitlines() if line.strip()][:4]
                workspace_label = self._preferred_workspace_label(state)
                if pwd_output and entries:
                    if workspace_label:
                        return f"The current directory is {pwd_output} in the {workspace_label} workspace, and the workspace root contains {', '.join(entries)}."
                    return f"The current directory is {pwd_output}, and the workspace root contains {', '.join(entries)}."
                if pwd_output:
                    return f"The current directory is {pwd_output}."
            generic_terminal = self._build_terminal_fallback_summary(state, terminal_outputs[-1])
            if generic_terminal:
                return generic_terminal
        if contract.require_grounding:
            directory_result = next((item for item in reversed(tool_results) if item.get("tool") == "list_directory"), None)
            if directory_result:
                directory_summary = self._build_directory_fallback_summary(state, directory_result)
                if directory_summary:
                    return directory_summary
        if contract.require_grounding and self._is_followup_grounded_turn(state):
            workspace_label = self._preferred_workspace_label(state)
            injected = self._inject_workspace_label_into_followup(
                str(state.get("final_text") or ""),
                workspace_label or "",
                contract.requested_language,
            )
            if injected:
                return injected
            followup_summary = self._synthesize_followup_grounded_summary(state)
            if followup_summary:
                return followup_summary
        rag_result = self._last_rag_tool_result(state)
        if rag_result:
            rag_meta = rag_result.get("meta", {}) if isinstance(rag_result.get("meta"), dict) else {}
            strong_hits = int(rag_meta.get("strongHitCount", 0) or 0)
            if strong_hits <= 0:
                return "I couldn't find enough reliable information in the session documents to answer confidently."
            compact_rag_answer = self._compact_rag_answer(state)
            if compact_rag_answer:
                return compact_rag_answer
        return None

    def _invoke_final_repair(self, state: AgentState, violations: list[tuple[str, str]]) -> tuple[AgentState, list[dict[str, Any]]]:
        state["repair_attempts"] = int(state.get("repair_attempts", 0) or 0) + 1
        events = [
            run_diagnostic(
                state["run_id"],
                "final_contract_repair_requested",
                "; ".join(message for _code, message in violations),
                level="warn",
            ),
            run_diagnostic(
                state["run_id"],
                "verify_guardrail_repair",
                "; ".join(message for _code, message in violations),
                level="warn",
            )
        ]
        repair_prompt = self._build_final_repair_prompt(state, violations)
        model = self.deps.model_factory(state.get("profile"), state.get("model"))
        response = model.invoke(list(state["messages"]) + [SystemMessage(content=repair_prompt)])
        repaired_text = response.content if isinstance(response.content, str) else str(response.content or "")
        repaired_text = str(repaired_text or "").strip()
        if not repaired_text:
            self._bump_no_progress(state)
            fallback = self._synthesize_fallback_final_answer(state)
            if fallback:
                repaired_text = fallback
        repaired_state: AgentState = {
            **state,
            "messages": list(state["messages"]) + [SystemMessage(content=repair_prompt), AIMessage(content=repaired_text)],
            "final_text": repaired_text,
            "final_repair_attempted": True,
        }
        remaining = self._collect_contract_violations(repaired_state, repaired_text)
        if remaining:
            fallback = self._synthesize_fallback_final_answer(state)
            if fallback and normalize_whitespace(fallback) != normalize_whitespace(repaired_text):
                repaired_text = fallback
                repaired_state = {
                    **state,
                    "messages": list(state["messages"]) + [SystemMessage(content=repair_prompt), AIMessage(content=repaired_text)],
                    "final_text": repaired_text,
                    "final_repair_attempted": True,
                }
                remaining = self._collect_contract_violations(repaired_state, repaired_text)
        if remaining:
            events.append(
                run_diagnostic(
                    state["run_id"],
                    "final_contract_repair_failed",
                    "; ".join(message for _code, message in remaining),
                    level="warn",
                )
            )
            events.append(
                run_diagnostic(
                    state["run_id"],
                    "verify_guardrail_repeated",
                    "; ".join(message for _code, message in remaining),
                    level="warn",
                )
            )
            self._bump_no_progress(repaired_state)
        else:
            self._reset_no_progress(repaired_state)
        return repaired_state, events

    def _finalize_result_contract(self, state: AgentState) -> AgentState:
        if state.get("pending_approval") or state.get("pending_clarification"):
            return state
        assessment = self._evaluate_goal_completion(state, state.get("final_text", ""))
        multistep_violations = self._collect_multistep_violations(state, state.get("final_text", ""))
        multistep_events: list[dict[str, Any]] = []
        multistep_events.extend(self._goal_gap_events(state["run_id"], state, "finalize_result"))
        multi_step_contract = MultiStepContract.from_payload(state.get("multi_step_contract"))
        if is_multistep_contract_active(multi_step_contract):
            multistep_events.extend(
                self._build_multistep_progress_events(
                    state["run_id"],
                    multi_step_contract,
                    dict(state.get("step_progress") or {}),
                    dict(state.get("evidence_map") or {}),
                    reason="finalize_result",
                )
            )
        for code, message in multistep_violations:
            multistep_events.append(run_diagnostic(state["run_id"], code, message, level="warn"))
        if multistep_events:
            state["tool_events"] = list(state.get("tool_events", [])) + multistep_events
        violations = list(assessment.mismatch_codes)
        extra_events: list[dict[str, Any]] = []
        for code, message in violations:
            if code in {
                "sources_missing_in_final_answer",
                "exact_output_mismatch",
                "wrong_response_language",
                "rag_grounding_weak",
                "answer_ignores_strong_evidence",
                "rag_citations_missing",
                "empty_final_answer",
                "invalid_final_answer",
                "action_claim_without_tool_open_url",
                "action_claim_without_tool_open_file",
                "task_completion_unverified",
            }:
                extra_events.append(run_diagnostic(state["run_id"], code, message, level="warn"))
        if assessment.action == "gather_more_evidence":
            extra_events.append(
                run_diagnostic(
                    state["run_id"],
                    "task_completion_unverified",
                    "Goal gap still contains missing evidence; the run should continue before finishing.",
                    level="warn",
                    data=dict(state.get("goal_gap_summary") or {}),
                )
            )
        if is_multistep_contract_active(multi_step_contract) and assessment.gap.is_complete and not violations:
            extra_events.append(
                run_diagnostic(
                    state["run_id"],
                    "finished_after_sufficient_evidence",
                    "Verified evidence fully covers the goal, so the run is complete.",
                    level="info",
                    data=dict(state.get("goal_gap_summary") or {}),
                )
            )
            extra_events.append(
                run_diagnostic(
                    state["run_id"],
                    "stopped_because_gap_closed",
                    "Verified evidence covers the goal, so the run can stop without forcing more steps.",
                    level="info",
                    data=dict(state.get("goal_gap_summary") or {}),
                )
            )
            required_inputs = len(multi_step_contract.required_inputs)
            tool_result_count = len(state.get("tool_results", []))
            if tool_result_count > max(required_inputs, 0):
                extra_events.append(
                    run_diagnostic(
                        state["run_id"],
                        "overcontinued_without_need",
                        "The goal was already covered, but the run used more tool steps than the explicit evidence needs.",
                        level="info",
                        data={
                            **dict(state.get("goal_gap_summary") or {}),
                            "toolResultCount": tool_result_count,
                            "requiredInputCount": required_inputs,
                        },
                    )
                )
        if multistep_violations and not any(code == "multistep_incomplete" for code, _message in multistep_violations):
            if not state.get("final_repair_attempted"):
                state["tool_events"] = list(state.get("tool_events", [])) + [
                    run_diagnostic(
                        state["run_id"],
                        "final_contract_repair_requested",
                        "; ".join(message for _code, message in multistep_violations),
                        level="warn",
                    ),
                    run_diagnostic(
                        state["run_id"],
                        "verify_guardrail_repair",
                        "; ".join(message for _code, message in multistep_violations),
                        level="warn",
                    )
                ]
                fallback = self._synthesize_fallback_final_answer(state)
                if fallback:
                    state["messages"] = list(state["messages"]) + [AIMessage(content=fallback)]
                    state["final_text"] = fallback
                    state["final_repair_attempted"] = True
                    multistep_violations = self._collect_multistep_violations(state, state.get("final_text", ""))
                if multistep_violations:
                    state["tool_events"] = list(state.get("tool_events", [])) + [
                        run_diagnostic(
                            state["run_id"],
                            "final_contract_repair_failed",
                            "; ".join(message for _code, message in multistep_violations),
                            level="warn",
                        ),
                        run_diagnostic(
                            state["run_id"],
                            "verify_guardrail_repeated",
                            "; ".join(message for _code, message in multistep_violations),
                            level="warn",
                        )
                    ]
        if violations and not state.get("final_repair_attempted") and int(state.get("repair_attempts", 0) or 0) < MAX_REPAIR_ATTEMPTS:
            repaired_state, repair_events = self._invoke_final_repair(state, violations)
            repaired_state["tool_events"] = list(repaired_state.get("tool_events", [])) + extra_events + repair_events
            return repaired_state
        if violations and state.get("final_repair_attempted") and not any(event.get("code") == "verify_guardrail_repeated" for event in extra_events):
            extra_events.append(
                run_diagnostic(
                    state["run_id"],
                    "final_contract_repair_failed",
                    "; ".join(message for _code, message in violations),
                    level="warn",
                )
            )
            extra_events.append(
                run_diagnostic(
                    state["run_id"],
                    "verify_guardrail_repeated",
                    "; ".join(message for _code, message in violations),
                    level="warn",
                )
            )
        if extra_events:
            state["tool_events"] = list(state.get("tool_events", [])) + extra_events
        return state

    def _verify_state_for_loop(self, state: AgentState) -> AgentState:
        if state.get("pending_approval") or state.get("pending_clarification"):
            return {**state, "verify_action": "end"}

        assessment = self._evaluate_goal_completion(state, state.get("final_text", ""))
        multistep_violations = self._collect_multistep_violations(state, state.get("final_text", ""))
        tool_events = list(state.get("tool_events", [])) + self._goal_gap_events(state["run_id"], state, "verify_loop")
        violations = list(assessment.mismatch_codes)

        for code, message in multistep_violations:
            tool_events.append(run_diagnostic(state["run_id"], code, message, level="warn"))
        for code, message in violations:
            if code in {
                "sources_missing_in_final_answer",
                "exact_output_mismatch",
                "wrong_response_language",
                "rag_grounding_weak",
                "answer_ignores_strong_evidence",
                "rag_citations_missing",
                "empty_final_answer",
                "invalid_final_answer",
                "action_claim_without_tool_open_url",
                "action_claim_without_tool_open_file",
                "task_completion_unverified",
            }:
                tool_events.append(run_diagnostic(state["run_id"], code, message, level="warn"))

        if assessment.action == "gather_more_evidence":
            missing_inputs = list(assessment.gap.missing_evidence or [])
            tool_events.append(
                run_diagnostic(
                    state["run_id"],
                    "stopped_because_missing_evidence",
                    "Verified evidence is still missing, so the run should continue instead of finishing.",
                    level="info",
                    data=dict(state.get("goal_gap_summary") or {}),
                )
            )
            if missing_inputs and self._has_runtime_actionable_missing_evidence(missing_inputs):
                prior_signature = str((state.get("evidence_map") or {}).get("signature") or "")
                acquired_state, _events, acquired = self._run_explicit_evidence_acquisition(state, missing_inputs[0])
                if acquired:
                    updated_signature = str((acquired_state.get("evidence_map") or {}).get("signature") or "")
                    acquired_state["multistep_no_progress_turns"] = 0 if updated_signature != prior_signature else int(state.get("multistep_no_progress_turns", 0) or 0) + 1
                    acquired_state["multistep_repair_attempted"] = updated_signature == prior_signature
                    acquired_state["verify_action"] = "agent"
                    acquired_state["tool_events"] = list(acquired_state.get("tool_events", [])) + tool_events
                    return acquired_state
            continue_prompt = self._build_multistep_continue_prompt(
                state,
                [("task_completion_unverified", "Task completion is not yet verified.")],
            )
            return {
                **state,
                "messages": list(state["messages"]) + [SystemMessage(content=continue_prompt)],
                "tool_events": tool_events
                + [
                    run_diagnostic(
                        state["run_id"],
                        "task_completion_unverified",
                        "Goal gap still contains missing evidence; continuing the run before finishing.",
                        level="warn",
                        data=dict(state.get("goal_gap_summary") or {}),
                    )
                ],
                "final_text": "",
                "multistep_repair_attempted": True,
                "verify_action": "agent",
            }

        if is_multistep_contract_active(MultiStepContract.from_payload(state.get("multi_step_contract"))) and assessment.gap.is_complete and not violations:
            tool_events.append(
                run_diagnostic(
                    state["run_id"],
                    "stopped_because_gap_closed",
                    "Verified evidence covers the goal, so the run can stop without forcing more steps.",
                    level="info",
                    data=dict(state.get("goal_gap_summary") or {}),
                )
            )

        if violations and not state.get("final_repair_attempted") and int(state.get("repair_attempts", 0) or 0) < MAX_REPAIR_ATTEMPTS:
            repaired_state, repair_events = self._invoke_final_repair(state, violations)
            repaired_state["tool_events"] = list(repaired_state.get("tool_events", [])) + tool_events + repair_events
            repaired_state["verify_action"] = "end"
            return repaired_state

        return {
            **state,
            "tool_events": tool_events,
            "verify_action": "end",
        }

    def _normalize_tool_args(self, state: AgentState, tool_name: str, tool_args: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        normalized_args = dict(tool_args or {})
        if tool_name not in TERMINAL_TOOLS_REQUIRING_ID:
            return normalized_args, []
        resolution = self.deps.terminal_manager.resolve_terminal_reference(
            normalized_args.get("terminalId"),
            run_id=state["run_id"],
            session_id=state["session_id"],
            require_alive=(tool_name != "terminal_close"),
        )
        diagnostics: list[dict[str, Any]] = []
        resolved_terminal_id = str(resolution.get("terminalId") or "")
        requested_terminal_id = resolution.get("requestedTerminalId")
        strategy = str(resolution.get("strategy") or "")
        if resolved_terminal_id:
            normalized_args["terminalId"] = resolved_terminal_id
        if strategy and strategy != "explicit":
            diagnostics.append(
                run_diagnostic(
                    state["run_id"],
                    "terminal_id_inferred",
                    f"{tool_name} reused terminal {resolved_terminal_id} via {strategy}.",
                )
            )
        elif requested_terminal_id and str(requested_terminal_id).strip() != resolved_terminal_id:
            diagnostics.append(
                run_diagnostic(
                    state["run_id"],
                    "terminal_id_rewritten",
                    f"{tool_name} redirected from {requested_terminal_id} to {resolved_terminal_id}.",
                    level="warn",
                )
            )
        return normalized_args, diagnostics

    def _tool_failure_result(
        self,
        run_id: str,
        tool_name: str,
        tool_id: str,
        error: Exception,
    ) -> tuple[list[dict[str, Any]], str]:
        message = str(error) or f"{tool_name} failed"
        events: list[dict[str, Any]] = [
            {
                "type": "tool_result",
                "runId": run_id,
                "actionId": tool_id,
                "name": tool_name,
                "ok": False,
                "preview": message[:400],
            }
        ]
        code = "terminal_tool_error" if tool_name in TERMINAL_TOOLS_REQUIRING_ID else "tool_execution_error"
        events.append(run_diagnostic(run_id, code, f"{tool_name} failed: {message}", level="warn"))
        return events, f"Tool {tool_name} failed: {message}"

    def _materialize_tool_result(
        self,
        run_id: str,
        tool_name: str,
        tool_id: str,
        raw_result: str,
    ) -> tuple[list[dict[str, Any]], str, bool, dict[str, Any] | None]:
        if tool_name != "run_terminal":
            rag_tool_payload = decode_rag_tool_payload(raw_result)
            if rag_tool_payload:
                lookup = rag_tool_payload.get("lookup", {})
                rendered = str(rag_tool_payload.get("rendered") or "")
                meta = lookup.get("meta", {}) if isinstance(lookup.get("meta"), dict) else {}
                events: list[dict[str, Any]] = [
                    {
                        "type": "tool_result",
                        "runId": run_id,
                        "actionId": tool_id,
                        "name": tool_name,
                        "ok": lookup.get("status") != "error",
                        "preview": rendered[:400],
                    }
                ]
                if meta.get("followupContextUsed"):
                    events.append(run_diagnostic(run_id, "rag_followup_context_used", "RAG lookup reused the previous session-doc context for this follow-up turn."))
                elif meta.get("followupEligible"):
                    events.append(run_diagnostic(run_id, "rag_followup_context_skipped", "RAG lookup treated this turn as standalone because the previous context was not reliable enough to reuse."))
                if int(meta.get("strongHitCount", 0) or 0) <= 0:
                    events.append(run_diagnostic(run_id, "rag_lookup_no_strong_hits", "RAG lookup returned only weak hits or no strong evidence.", level="warn"))
                return events, rendered, lookup.get("status") != "error", {
                    "status": lookup.get("status"),
                    "meta": meta,
                    "hits": lookup.get("hits", []),
                }
            terminal_tool_payload = decode_terminal_tool_payload(raw_result)
            if terminal_tool_payload:
                events = list(terminal_tool_payload.get("events", []))
                terminal = terminal_tool_payload.get("terminal", {}) if isinstance(terminal_tool_payload.get("terminal"), dict) else {}
                snapshot = terminal_tool_payload.get("snapshot", {}) if isinstance(terminal_tool_payload.get("snapshot"), dict) else {}
                resolution = terminal_tool_payload.get("resolution", {}) if isinstance(terminal_tool_payload.get("resolution"), dict) else {}
                matched = terminal_tool_payload.get("matched")
                tail = str(terminal_tool_payload.get("tail") or terminal.get("tail") or snapshot.get("tail") or "")
                preview_lines = [
                    f"Tool: {terminal_tool_payload.get('tool', tool_name)}",
                    f"Terminal: {terminal.get('terminalId') or snapshot.get('terminalId') or ''}",
                ]
                if terminal.get("owner"):
                    preview_lines.append(f"Owner: {terminal['owner']}")
                if resolution.get("strategy"):
                    preview_lines.append(f"Resolution: {resolution['strategy']}")
                if matched is not None:
                    preview_lines.append(f"Matched: {matched}")
                if tail:
                    preview_lines.extend(["Tail:", tail[-2000:]])
                formatted = "\n".join(line for line in preview_lines if line)
                events.append(
                    {
                        "type": "tool_result",
                        "runId": run_id,
                        "actionId": tool_id,
                        "name": tool_name,
                        "ok": True,
                        "preview": formatted[:400],
                    }
                )
                return events, formatted, True, None
            return (
                [
                    {
                        "type": "tool_result",
                        "runId": run_id,
                        "actionId": tool_id,
                        "name": tool_name,
                        "ok": True,
                        "preview": raw_result[:400],
                    }
                ],
                raw_result,
                True,
                None,
            )
        payload = decode_terminal_payload(raw_result)
        if not payload:
            return (
                [
                    {
                        "type": "tool_result",
                        "runId": run_id,
                        "actionId": tool_id,
                        "name": tool_name,
                        "ok": False,
                        "preview": raw_result[:400],
                    },
                    run_diagnostic(run_id, "terminal_payload_invalid", "Terminal tool returned an invalid payload.", level="warn"),
                ],
                raw_result,
                False,
                None,
            )
        terminal_id = str(payload.get("terminalId") or uuid.uuid4())
        command = str(payload.get("command") or "")
        cwd = str(payload.get("cwd") or "")
        stdout = str(payload.get("stdout") or "")
        stderr = str(payload.get("stderr") or "")
        blocked = bool(payload.get("blocked"))
        exit_code = int(payload.get("exitCode") or 0)
        output = "\n".join(part for part in [stdout.strip(), stderr.strip()] if part).strip()
        events = [terminal_opened(run_id, terminal_id, command, cwd)]
        if blocked:
            events.append(run_diagnostic(run_id, "terminal_command_blocked", stderr or "Terminal command blocked by policy.", level="warn"))
            events.append(terminal_error(run_id, terminal_id, stderr or "Terminal command blocked by policy."))
        else:
            events.append(terminal_exit(run_id, terminal_id, exit_code, output[:8000]))
            if exit_code != 0:
                events.append(run_diagnostic(run_id, "terminal_nonzero_exit", f"Terminal command exited with code {exit_code}.", level="warn"))
        formatted = "\n".join(
            line
            for line in [
                f"Command: {command}",
                f"CWD: {cwd}",
                f"Exit code: {exit_code}",
                "Output:",
                output or "(no output)",
            ]
            if line
        )
        events.append(
            {
                "type": "tool_result",
                "runId": run_id,
                "actionId": tool_id,
                "name": tool_name,
                "ok": not blocked and exit_code == 0,
                "preview": formatted[:400],
            }
        )
        return events, formatted, (not blocked and exit_code == 0), None

    async def _stream_result(self, run_id: str, result: AgentState):
        self._append_trace_steps_from_tool_events(result)
        for event in result.get("tool_events", []):
            yield event
        if result.get("pending_approval"):
            approval = result["pending_approval"]
            failure = self._build_failure_payload(
                "approval_blocked",
                context={"approvalId": approval.get("approvalId"), "tool": approval.get("name")},
            )
            self._append_run_trace_step(
                result,
                kind="finish",
                status="blocked",
                summary="Execution blocked waiting for approval.",
                data={"code": failure["code"]},
            )
            trace = self._finalize_run_trace(result, outcome="failed", failure=failure)
            ok, message = self._assert_terminal_consistency(trace)
            if not ok:
                yield run_diagnostic(run_id, "assert_terminal_consistency", message, level="warn")
            self.deps.state_store.save("approvals", approval["approvalId"], {**self._serialize_snapshot(result)})
            yield run_state(run_id, "awaiting_approval")
            yield approval_required(
                run_id=run_id,
                approval_id=approval["approvalId"],
                name=approval["name"],
                arguments=json.dumps(approval["arguments"], ensure_ascii=False),
                risk_level=approval["riskLevel"],
            )
            yield done(run_id=run_id, run_trace=self._run_trace_summary(trace))
            return
        if result.get("pending_clarification"):
            clarification = result["pending_clarification"]
            failure = self._build_failure_payload(
                "clarification_blocked",
                context={"clarificationId": clarification.get("clarificationId")},
            )
            self._append_run_trace_step(
                result,
                kind="finish",
                status="blocked",
                summary="Execution blocked waiting for clarification.",
                data={"code": failure["code"]},
            )
            trace = self._finalize_run_trace(result, outcome="failed", failure=failure)
            ok, message = self._assert_terminal_consistency(trace)
            if not ok:
                yield run_diagnostic(run_id, "assert_terminal_consistency", message, level="warn")
            self.deps.state_store.save("clarifications", clarification["clarificationId"], {**self._serialize_snapshot(result)})
            yield run_state(run_id, "awaiting_clarification")
            yield clarification_required(
                run_id=run_id,
                clarification_id=clarification["clarificationId"],
                question=clarification["question"],
                options=clarification["options"],
            )
            yield done(run_id=run_id, run_trace=self._run_trace_summary(trace))
            return
        final_text = normalize_final_markdown(str(result.get("final_text", "") or ""))
        if not final_text:
            fallback = self._synthesize_fallback_final_answer(result)
            if fallback:
                final_text = normalize_final_markdown(fallback)
        final_text = normalize_final_markdown(final_text)
        assessment = self._evaluate_goal_completion(result, final_text)
        violations = list(assessment.mismatch_codes)
        blocking_violations = [(code, message) for code, message in violations if self._is_blocking_violation(code)]
        verify_ok = bool(final_text) and not blocking_violations and assessment.action != "gather_more_evidence"
        self._append_run_trace_step(
            result,
            kind="verify",
            status="ok" if verify_ok else "warn",
            summary="Final verification passed." if verify_ok else "Final verification failed.",
            data={
                "action": assessment.action,
                "mismatchCodes": [code for code, _message in blocking_violations] if blocking_violations else [code for code, _message in violations],
            },
        )
        yield run_phase(run_id, "finish")
        if verify_ok:
            visible_text = final_text
            self._append_run_trace_step(result, kind="finish", status="ok", summary="Run completed with verified final answer.")
            trace = self._finalize_run_trace(result, outcome="completed")
            end_state = "completed"
        else:
            failure = self._failure_from_goal_assessment(result, assessment, blocking_violations or violations, final_text)
            violation_codes = {code for code, _message in (blocking_violations or violations)}
            if (
                normalize_whitespace(final_text)
                and failure.get("code") in {"missing_evidence", "verification_failed", "no_progress_limit"}
                and not violation_codes.intersection({"empty_final_answer", "invalid_final_answer"})
            ):
                visible_text = final_text
            else:
                visible_text = normalize_final_markdown(self._build_failure_final_text(failure))
            self._append_run_trace_step(
                result,
                kind="finish",
                status="failed",
                summary=f"Run failed: {failure['code']}.",
                data={"code": failure["code"]},
            )
            trace = self._finalize_run_trace(result, outcome="failed", failure=failure)
            end_state = "failed"
        for piece in split_for_streaming(visible_text):
            yield token(piece)
            await asyncio.sleep(0)
        yield self._build_run_metrics_event(result)
        ok, message = self._assert_terminal_consistency(trace)
        if not ok:
            yield run_diagnostic(run_id, "assert_terminal_consistency", message, level="warn")
        yield run_state(run_id, end_state)
        yield done(run_id=run_id, run_trace=self._run_trace_summary(trace))
        return

    def _create_graph(self):
        return create_runtime_graph(self)

    def _initial_state(self, request: SidecarChatRequest, run_id: str, messages: list[BaseMessage]) -> AgentState:
        provider = self.deps.provider_resolver(request.profile, request.model)
        final_contract = build_final_answer_contract(self._last_user_message(messages), request.forceToolUse)
        multi_step_contract = build_multi_step_contract(self._last_user_message(messages))
        seeded_evidence_map = self._seed_evidence_map_from_messages(messages)
        seeded_state: AgentState = {
            "messages": messages,
            "session_id": request.sessionId,
            "run_id": run_id,
            "workspace_root": request.workspaceRoot or "",
            "profile": request.profile,
            "model": request.model,
            "provider_mode": provider.mode,
            "policy_profile": request.policyProfile,
            "tool_toggles": request.toolToggles,
            "force_tool_use": request.forceToolUse,
            "final_text": "",
            "pending_approval": None,
            "pending_clarification": None,
            "tool_events": [],
            "provider_capabilities": provider_capabilities_payload(provider),
            "final_contract": final_contract.to_payload(),
            "multi_step_contract": multi_step_contract.to_payload(),
            "goal_summary": {},
            "realization_summary": {},
            "goal_gap_summary": {},
            "step_progress": {"coveredInputs": [], "missingInputs": [], "coveredAnswerFields": [], "missingAnswerFields": [], "completedStepCount": 0, "completionRatio": 1.0},
            "evidence_map": seeded_evidence_map,
            "tool_results": [],
            "used_tool_names": [],
            "executed_tools": [],
            "recent_tool_signatures": [],
            "repeated_tool_signature_streak": None,
            "no_progress_turns": 0,
            "repair_attempts": 0,
            "stale_final_retry_count": 0,
            "last_read_file_path": None,
            "last_read_file_content": None,
            "last_opened_url": None,
            "last_opened_file_path": None,
            "written_files": [],
            "web_search_last_result_count": -1,
            "web_search_last_result_urls": [],
            "rag_lookup_last_status": None,
            "rag_lookup_last_hit_count": None,
            "rag_lookup_last_hits": [],
            "rag_lookup_last_structured": False,
            "final_repair_attempted": False,
            "multistep_repair_attempted": False,
            "multistep_no_progress_turns": 0,
            "verify_action": None,
            "graph_route": None,
            "conversation": {},
            "runtime": {},
            "goal": {},
            "evidence": {},
            "decision": {},
            "control": {},
            "output": {},
            "run_trace": {},
        }
        self._refresh_goal_tracking(seeded_state, "")
        self._sync_graph_state_slices(seeded_state)
        seeded_state["run_trace"] = self._build_initial_run_trace(seeded_state)
        return seeded_state

    def _last_user_message(self, messages: list[BaseMessage]) -> str:
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                return str(message.content or "")
        return ""

    async def _run_graph_with_heartbeats(self, run_id: str, state: AgentState, detail: str | None = None):
        loop = asyncio.get_running_loop()
        result_holder: dict[str, Any] = {}
        completed = asyncio.Event()

        def invoke_graph() -> None:
            try:
                result_holder["result"] = self.graph.invoke(state)
            except Exception as exc:  # pragma: no cover - exercised via public stream methods
                result_holder["error"] = exc
            finally:
                loop.call_soon_threadsafe(completed.set)

        worker = asyncio.create_task(asyncio.to_thread(invoke_graph))
        heartbeat_count = 0
        try:
            while True:
                try:
                    await asyncio.wait_for(completed.wait(), timeout=10.0)
                    break
                except asyncio.TimeoutError:
                    heartbeat_count += 1
                    suffix = f" while {detail}" if detail else ""
                    yield run_diagnostic(
                        run_id,
                        "runtime_heartbeat",
                        f"Run still in progress{suffix} ({heartbeat_count * 10}s elapsed).",
                    )
        finally:
            await worker

        if "error" in result_holder:
            raise result_holder["error"]
        yield {"type": "_graph_result", "result": result_holder["result"]}

    def _serialize_snapshot(self, state: AgentState) -> dict[str, Any]:
        return {
            "run_id": state["run_id"],
            "workspace_root": state["workspace_root"],
            "profile": state.get("profile"),
            "model": state.get("model"),
            "provider_mode": state["provider_mode"],
            "policy_profile": state["policy_profile"],
            "tool_toggles": state.get("tool_toggles"),
            "force_tool_use": state.get("force_tool_use"),
            "messages": serialize_messages(state["messages"]),
            "session_id": state["session_id"],
            "final_text": state.get("final_text", ""),
            "pending_approval": state.get("pending_approval"),
            "pending_clarification": state.get("pending_clarification"),
            "provider_capabilities": state.get("provider_capabilities"),
            "final_contract": state.get("final_contract"),
            "multi_step_contract": state.get("multi_step_contract"),
            "goal_summary": state.get("goal_summary", {}),
            "realization_summary": state.get("realization_summary", {}),
            "goal_gap_summary": state.get("goal_gap_summary", {}),
            "step_progress": state.get("step_progress", {}),
            "evidence_map": state.get("evidence_map", {}),
            "tool_results": state.get("tool_results", []),
            "used_tool_names": state.get("used_tool_names", []),
            "executed_tools": state.get("executed_tools", []),
            "recent_tool_signatures": state.get("recent_tool_signatures", []),
            "repeated_tool_signature_streak": state.get("repeated_tool_signature_streak"),
            "no_progress_turns": state.get("no_progress_turns", 0),
            "repair_attempts": state.get("repair_attempts", 0),
            "stale_final_retry_count": state.get("stale_final_retry_count", 0),
            "last_read_file_path": state.get("last_read_file_path"),
            "last_read_file_content": state.get("last_read_file_content"),
            "last_opened_url": state.get("last_opened_url"),
            "last_opened_file_path": state.get("last_opened_file_path"),
            "written_files": state.get("written_files", []),
            "web_search_last_result_count": state.get("web_search_last_result_count", -1),
            "web_search_last_result_urls": state.get("web_search_last_result_urls", []),
            "rag_lookup_last_status": state.get("rag_lookup_last_status"),
            "rag_lookup_last_hit_count": state.get("rag_lookup_last_hit_count"),
            "rag_lookup_last_hits": state.get("rag_lookup_last_hits", []),
            "rag_lookup_last_structured": state.get("rag_lookup_last_structured", False),
            "final_repair_attempted": state.get("final_repair_attempted", False),
            "multistep_repair_attempted": state.get("multistep_repair_attempted", False),
            "multistep_no_progress_turns": state.get("multistep_no_progress_turns", 0),
            "verify_action": state.get("verify_action"),
            "graph_route": state.get("graph_route"),
            "conversation": state.get("conversation", {}),
            "runtime": state.get("runtime", {}),
            "goal": state.get("goal", {}),
            "evidence": state.get("evidence", {}),
            "decision": state.get("decision", {}),
            "control": state.get("control", {}),
            "output": state.get("output", {}),
            "run_trace": state.get("run_trace", {}),
        }

    def _deserialize_snapshot(self, payload: dict[str, Any]) -> AgentState:
        return {
            "messages": deserialize_messages(payload["messages"]),
            "session_id": payload["session_id"],
            "run_id": payload["run_id"],
            "workspace_root": payload["workspace_root"],
            "profile": payload.get("profile"),
            "model": payload.get("model"),
            "provider_mode": payload["provider_mode"],
            "policy_profile": payload["policy_profile"],
            "tool_toggles": payload.get("tool_toggles"),
            "force_tool_use": payload.get("force_tool_use"),
            "final_text": payload.get("final_text", ""),
            "pending_approval": payload.get("pending_approval"),
            "pending_clarification": payload.get("pending_clarification"),
            "tool_events": [],
            "provider_capabilities": payload.get("provider_capabilities")
            or {
                "supportsNativeTools": True,
                "supportsTextualReplay": False,
                "supportsMultiToolTurn": True,
                "maxToolCallsPerTurn": 8,
                "requiresSequentialToolLoop": False,
                "configSource": "env",
                "providerFamily": "openrouter",
            },
            "final_contract": payload.get("final_contract") or FinalAnswerContract().to_payload(),
            "multi_step_contract": payload.get("multi_step_contract") or MultiStepContract().to_payload(),
            "goal_summary": payload.get("goal_summary") or {},
            "realization_summary": payload.get("realization_summary") or {},
            "goal_gap_summary": payload.get("goal_gap_summary") or {},
            "step_progress": payload.get("step_progress") or {"coveredInputs": [], "missingInputs": [], "coveredAnswerFields": [], "missingAnswerFields": [], "completedStepCount": 0, "completionRatio": 1.0},
            "evidence_map": payload.get("evidence_map") or {"files": {}, "directories": {}, "terminals": [], "facts": {}, "signature": ""},
            "tool_results": payload.get("tool_results", []),
            "used_tool_names": payload.get("used_tool_names", []),
            "executed_tools": payload.get("executed_tools", []),
            "recent_tool_signatures": payload.get("recent_tool_signatures", []),
            "repeated_tool_signature_streak": payload.get("repeated_tool_signature_streak"),
            "no_progress_turns": int(payload.get("no_progress_turns", 0) or 0),
            "repair_attempts": int(payload.get("repair_attempts", 0) or 0),
            "stale_final_retry_count": int(payload.get("stale_final_retry_count", 0) or 0),
            "last_read_file_path": payload.get("last_read_file_path"),
            "last_read_file_content": payload.get("last_read_file_content"),
            "last_opened_url": payload.get("last_opened_url"),
            "last_opened_file_path": payload.get("last_opened_file_path"),
            "written_files": payload.get("written_files", []),
            "web_search_last_result_count": int(payload.get("web_search_last_result_count", -1) or -1),
            "web_search_last_result_urls": payload.get("web_search_last_result_urls", []),
            "rag_lookup_last_status": payload.get("rag_lookup_last_status"),
            "rag_lookup_last_hit_count": payload.get("rag_lookup_last_hit_count"),
            "rag_lookup_last_hits": payload.get("rag_lookup_last_hits", []),
            "rag_lookup_last_structured": bool(payload.get("rag_lookup_last_structured", False)),
            "final_repair_attempted": bool(payload.get("final_repair_attempted", False)),
            "multistep_repair_attempted": bool(payload.get("multistep_repair_attempted", False)),
            "multistep_no_progress_turns": int(payload.get("multistep_no_progress_turns", 0) or 0),
            "verify_action": payload.get("verify_action"),
            "graph_route": payload.get("graph_route"),
            "conversation": dict(payload.get("conversation") or {}),
            "runtime": dict(payload.get("runtime") or {}),
            "goal": dict(payload.get("goal") or {}),
            "evidence": dict(payload.get("evidence") or {}),
            "decision": dict(payload.get("decision") or {}),
            "control": dict(payload.get("control") or {}),
            "output": dict(payload.get("output") or {}),
            "run_trace": dict(payload.get("run_trace") or {}),
        }

    async def stream_chat(self, request: SidecarChatRequest, messages: list[BaseMessage]):
        run_id = request.runId or str(uuid.uuid4())
        state = self._initial_state(request, run_id, messages)
        registry = self.deps.tool_registry_factory(state["workspace_root"], state["session_id"], state["run_id"], state.get("tool_toggles"))
        system_prompt = build_runtime_system_prompt(
            registry,
            selected_group=forced_tool_group(request.forceToolUse),
            provider_capabilities=state.get("provider_capabilities"),
        )
        state["messages"] = [SystemMessage(content=system_prompt)] + list(state["messages"])
        force_instruction = forced_tool_instruction(request.forceToolUse)
        if force_instruction:
            state["messages"] = [SystemMessage(content=force_instruction)] + list(state["messages"])
        yield run_state(run_id, "running")
        await asyncio.sleep(0)
        yield run_phase(run_id, "planning")
        await asyncio.sleep(0)
        yield run_phase(run_id, "execute")
        await asyncio.sleep(0)
        yield run_diagnostic(run_id, "provider_mode", f"Provider mode: {state['provider_mode']}.")
        await asyncio.sleep(0)
        provider_capabilities = state.get("provider_capabilities") or {}
        provider_family = str(provider_capabilities.get("providerFamily") or "unknown")
        config_source = str(provider_capabilities.get("configSource") or "unknown")
        yield run_diagnostic(run_id, "provider_family", f"Provider family: {provider_family}.")
        await asyncio.sleep(0)
        yield run_diagnostic(run_id, "provider_config_source", f"Provider config source: {config_source}.")
        await asyncio.sleep(0)
        config_path = str(provider_capabilities.get("configPath") or "").strip()
        if config_path:
            yield run_diagnostic(run_id, "provider_config_path", f"Provider config path: {config_path}.")
            await asyncio.sleep(0)
        yield run_diagnostic(run_id, "tool_routing_mode", "Tool selection is performed in the main model turn (single-pass).")
        await asyncio.sleep(0)
        contract = FinalAnswerContract.from_payload(state.get("final_contract"))
        if self._should_emit_contract_diagnostic(contract):
            yield build_contract_detected_diagnostic(run_id, contract)
        multi_step_contract = MultiStepContract.from_payload(state.get("multi_step_contract"))
        if is_multistep_contract_active(multi_step_contract):
            yield build_multistep_contract_diagnostic(run_id, multi_step_contract)
            await asyncio.sleep(0)
        clarification_enabled = bool((state.get("tool_toggles") or {}).get("clarification", True))
        early_clarification = build_early_clarification(self._last_user_message(state["messages"]), clarification_enabled)
        if early_clarification:
            clarification_id = str(uuid.uuid4())
            state["pending_clarification"] = {
                "clarificationId": clarification_id,
                "runId": run_id,
                "question": early_clarification["questions"][0],
                "questions": early_clarification["questions"],
                "options": early_clarification.get("options", []),
                "reason": early_clarification["reason"],
            }
            state["tool_events"] = [
                run_diagnostic(run_id, "clarification_needed", f"Clarification requested: {early_clarification['reason']}"),
                {
                    "type": "tool_call",
                    "runId": run_id,
                    "actionId": clarification_id,
                    "name": "request_clarification",
                    "arguments": json.dumps({"question": early_clarification["questions"][0]}, ensure_ascii=False),
                    "riskLevel": "safe",
                },
            ]
            async for event in self._stream_result(run_id, state):
                yield event
            return
        try:
            result = None
            async for event in self._run_graph_with_heartbeats(run_id, state, detail="executing the tool plan"):
                if event.get("type") == "_graph_result":
                    result = event["result"]
                    continue
                yield event
        except Exception as exc:
            failure = self._build_failure_payload("runtime_exception", message=str(exc))
            self._append_run_trace_step(state, kind="finish", status="failed", summary=f"Runtime exception: {failure['message']}.", data={"code": failure["code"]})
            trace = self._finalize_run_trace(state, outcome="failed", failure=failure)
            yield error_event(str(exc))
            yield run_state(run_id, "failed")
            ok, message = self._assert_terminal_consistency(trace)
            if not ok:
                yield run_diagnostic(run_id, "assert_terminal_consistency", message, level="warn")
            yield done(run_id=run_id, run_trace=self._run_trace_summary(trace))
            return
        if result is None:
            failure = self._build_failure_payload("runtime_exception", message="Runtime finished without a graph result.")
            self._append_run_trace_step(state, kind="finish", status="failed", summary="Runtime finished without a graph result.", data={"code": failure["code"]})
            trace = self._finalize_run_trace(state, outcome="failed", failure=failure)
            yield error_event("Runtime finished without a graph result.")
            yield run_state(run_id, "failed")
            ok, message = self._assert_terminal_consistency(trace)
            if not ok:
                yield run_diagnostic(run_id, "assert_terminal_consistency", message, level="warn")
            yield done(run_id=run_id, run_trace=self._run_trace_summary(trace))
            return
        async for event in self._stream_result(run_id, result):
            yield event

    async def stream_approval_decision(self, request: ApprovalDecisionRequest):
        snapshot = self.deps.state_store.pop("approvals", request.approval_id)
        if not snapshot:
            raise KeyError("Approval not found")
        state = self._deserialize_snapshot(snapshot)
        pending = state["pending_approval"]
        run_id = state["run_id"]
        yield approval_decision(run_id, request.approval_id, request.decision)
        if request.decision == "approved":
            registry = self.deps.tool_registry_factory(state["workspace_root"], state["session_id"], state["run_id"], state.get("tool_toggles"))
            registered = registry.get(pending["name"])
            if not registered:
                failure = self._build_failure_payload("tool_failed", message="Approved tool is no longer available.")
                self._append_run_trace_step(state, kind="finish", status="failed", summary="Approved tool is no longer available.", data={"code": failure["code"]})
                trace = self._finalize_run_trace(state, outcome="failed", failure=failure)
                yield error_event("Approved tool is no longer available")
                yield run_state(run_id, "failed")
                ok, message = self._assert_terminal_consistency(trace)
                if not ok:
                    yield run_diagnostic(run_id, "assert_terminal_consistency", message, level="warn")
                yield done(run_id=run_id, run_trace=self._run_trace_summary(trace))
                return
            try:
                normalized_args, normalization_events = self._normalize_tool_args(state, pending["name"], dict(pending["arguments"] or {}))
                pending["arguments"] = normalized_args
                for event in normalization_events:
                    yield event
                result = str(registered.tool.invoke(normalized_args))
                result_events, message_result, _ok, tool_meta = self._materialize_tool_result(run_id, pending["name"], pending["toolCallId"], result)
                for event in result_events:
                    yield event
                if state["provider_mode"] == "textual_replay":
                    state["messages"] = state["messages"] + [build_tool_replay_message(pending["name"], normalized_args, message_result, "succeeded")]
                else:
                    state["messages"] = state["messages"] + [ToolMessage(content=message_result, tool_call_id=pending["toolCallId"])]
                state["messages"] = state["messages"] + [self._build_tool_followup_guidance(state, pending["name"], message_result, "succeeded", tool_meta.get("meta") if tool_meta else None)]
                state["tool_results"] = list(state.get("tool_results", [])) + [
                    {
                        "tool": pending["name"],
                        "result": message_result,
                        "status": "succeeded",
                        **(
                            {
                                "meta": {**(tool_meta.get("meta") or {}), "hits": tool_meta.get("hits", [])},
                                "lookupStatus": tool_meta.get("status"),
                            }
                            if tool_meta
                            else {}
                        ),
                    }
                ]
                state["evidence_map"] = self._update_evidence_map(state, pending["name"], normalized_args, message_result, tool_meta)
                self._note_tool_success(state, pending["name"], normalized_args, message_result, tool_meta)
                post_tool_hint = self._maybe_build_post_tool_reasoning_hint(state, pending["name"], message_result, tool_meta)
                if post_tool_hint:
                    state["messages"] = state["messages"] + [SystemMessage(content=post_tool_hint)]
                state["used_tool_names"] = list(state.get("used_tool_names", [])) + [pending["name"]]
            except Exception as exc:
                failure_events, message_result = self._tool_failure_result(run_id, pending["name"], pending["toolCallId"], exc)
                for event in failure_events:
                    yield event
                if state["provider_mode"] == "textual_replay":
                    state["messages"] = state["messages"] + [build_tool_replay_message(pending["name"], pending["arguments"], message_result, "failed")]
                else:
                    state["messages"] = state["messages"] + [ToolMessage(content=message_result, tool_call_id=pending["toolCallId"])]
                state["messages"] = state["messages"] + [self._build_tool_followup_guidance(state, pending["name"], message_result, "failed")]
                state["tool_results"] = list(state.get("tool_results", [])) + [{"tool": pending["name"], "result": message_result, "status": "failed"}]
                self._note_executed_tool(state, pending["name"])
                state["used_tool_names"] = list(state.get("used_tool_names", [])) + [pending["name"]]
        else:
            rejection = (
                build_tool_replay_message(pending["name"], pending["arguments"], "Rejected by user approval policy.", "rejected")
                if state["provider_mode"] == "textual_replay"
                else SystemMessage(content=f"Tool {pending['name']} was rejected by the user. Continue without this tool.")
            )
            state["messages"] = state["messages"] + [rejection]
        state["pending_approval"] = None
        state["pending_clarification"] = None
        state["final_text"] = ""
        yield run_state(run_id, "running")
        yield run_phase(run_id, "repair")
        try:
            result = None
            async for event in self._run_graph_with_heartbeats(run_id, state, detail="repairing after approval"):
                if event.get("type") == "_graph_result":
                    result = event["result"]
                    continue
                yield event
        except Exception as exc:
            failure = self._build_failure_payload("runtime_exception", message=str(exc))
            self._append_run_trace_step(state, kind="finish", status="failed", summary=f"Runtime exception: {failure['message']}.", data={"code": failure["code"]})
            trace = self._finalize_run_trace(state, outcome="failed", failure=failure)
            yield error_event(str(exc))
            yield run_state(run_id, "failed")
            ok, message = self._assert_terminal_consistency(trace)
            if not ok:
                yield run_diagnostic(run_id, "assert_terminal_consistency", message, level="warn")
            yield done(run_id=run_id, run_trace=self._run_trace_summary(trace))
            return
        if result is None:
            failure = self._build_failure_payload("runtime_exception", message="Runtime finished without a graph result.")
            self._append_run_trace_step(state, kind="finish", status="failed", summary="Runtime finished without a graph result.", data={"code": failure["code"]})
            trace = self._finalize_run_trace(state, outcome="failed", failure=failure)
            yield error_event("Runtime finished without a graph result.")
            yield run_state(run_id, "failed")
            ok, message = self._assert_terminal_consistency(trace)
            if not ok:
                yield run_diagnostic(run_id, "assert_terminal_consistency", message, level="warn")
            yield done(run_id=run_id, run_trace=self._run_trace_summary(trace))
            return
        async for event in self._stream_result(run_id, result):
            yield event

    async def stream_clarification_decision(self, request: ClarificationDecisionRequest):
        snapshot = self.deps.state_store.pop("clarifications", request.clarification_id)
        if not snapshot:
            raise KeyError("Clarification not found")
        state = self._deserialize_snapshot(snapshot)
        run_id = state["run_id"]
        clarification = state.get("pending_clarification")
        state["messages"] = state["messages"] + [
            build_clarification_resume_message(clarification, request.answer),
            HumanMessage(content=request.answer),
        ]
        state["pending_approval"] = None
        state["pending_clarification"] = None
        state["final_text"] = ""
        clarification_enabled = bool((state.get("tool_toggles") or {}).get("clarification", True))
        early_clarification = build_early_clarification(request.answer, clarification_enabled)
        if early_clarification:
            clarification_id = str(uuid.uuid4())
            state["pending_clarification"] = {
                "clarificationId": clarification_id,
                "runId": run_id,
                "question": early_clarification["questions"][0],
                "questions": early_clarification["questions"],
                "options": early_clarification.get("options", []),
                "reason": early_clarification["reason"],
            }
            state["tool_events"] = [
                run_diagnostic(run_id, "clarification_needed", f"Clarification requested: {early_clarification['reason']}"),
                {
                    "type": "tool_call",
                    "runId": run_id,
                    "actionId": clarification_id,
                    "name": "request_clarification",
                    "arguments": json.dumps({"question": early_clarification["questions"][0]}, ensure_ascii=False),
                    "riskLevel": "safe",
                },
            ]
            async for event in self._stream_result(run_id, state):
                yield event
            return
        yield run_state(run_id, "running")
        yield run_phase(run_id, "execute", detail="Clarification received")
        try:
            result = None
            async for event in self._run_graph_with_heartbeats(run_id, state, detail="continuing after clarification"):
                if event.get("type") == "_graph_result":
                    result = event["result"]
                    continue
                yield event
        except Exception as exc:
            failure = self._build_failure_payload("runtime_exception", message=str(exc))
            self._append_run_trace_step(state, kind="finish", status="failed", summary=f"Runtime exception: {failure['message']}.", data={"code": failure["code"]})
            trace = self._finalize_run_trace(state, outcome="failed", failure=failure)
            yield error_event(str(exc))
            yield run_state(run_id, "failed")
            ok, message = self._assert_terminal_consistency(trace)
            if not ok:
                yield run_diagnostic(run_id, "assert_terminal_consistency", message, level="warn")
            yield done(run_id=run_id, run_trace=self._run_trace_summary(trace))
            return
        if result is None:
            failure = self._build_failure_payload("runtime_exception", message="Runtime finished without a graph result.")
            self._append_run_trace_step(state, kind="finish", status="failed", summary="Runtime finished without a graph result.", data={"code": failure["code"]})
            trace = self._finalize_run_trace(state, outcome="failed", failure=failure)
            yield error_event("Runtime finished without a graph result.")
            yield run_state(run_id, "failed")
            ok, message = self._assert_terminal_consistency(trace)
            if not ok:
                yield run_diagnostic(run_id, "assert_terminal_consistency", message, level="warn")
            yield done(run_id=run_id, run_trace=self._run_trace_summary(trace))
            return
        async for event in self._stream_result(run_id, result):
            yield event
