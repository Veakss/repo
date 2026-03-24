from __future__ import annotations

from typing import Any

import re

from langchain_core.messages import AIMessage, SystemMessage

from streamlit_python_only.events import run_diagnostic


class AgentEvaluationEngine:
    def __init__(self, owner: Any) -> None:
        self.owner = owner

    def __getattr__(self, name: str) -> Any:
        return getattr(self.owner, name)

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
        runtime_module = self._runtime_module()
        normalized_code = code if code in runtime_module.FAILURE_CODE_SET else "verification_failed"
        default_message, default_next_action = self._failure_defaults(normalized_code)
        payload: dict[str, Any] = {
            "code": normalized_code,
            "message": runtime_module.normalize_whitespace(message or default_message),
            "missing_evidence": list(missing_evidence or []),
            "next_action": next_action or default_next_action,
        }
        if context:
            payload["context"] = dict(context)
        return payload

    def _failure_from_goal_assessment(
        self,
        state: dict[str, Any],
        assessment: Any,
        violations: list[tuple[str, str]],
        final_text: str,
    ) -> dict[str, Any]:
        runtime_module = self._runtime_module()
        violation_codes = {code for code, _message in violations}
        if not runtime_module.normalize_whitespace(final_text) or {"empty_final_answer", "invalid_final_answer"}.intersection(violation_codes):
            message = "; ".join(message for _code, message in violations if _code in {"empty_final_answer", "invalid_final_answer"}) or None
            return self._build_failure_payload("invalid_final_answer", message=message)
        if assessment.gap.missing_evidence or assessment.gap.missing_facts or "task_completion_unverified" in violation_codes:
            missing = runtime_module._dedupe_strings([*assessment.gap.missing_evidence, *assessment.gap.missing_facts])
            return self._build_failure_payload(
                "missing_evidence",
                message="Goal gap is still open: missing verified evidence.",
                missing_evidence=missing,
            )
        if int(state.get("no_progress_turns", 0) or 0) >= runtime_module.MAX_NO_PROGRESS_TURNS:
            return self._build_failure_payload("no_progress_limit")
        if any(code in {"tool_execution_error", "terminal_tool_error"} for code, _message in violations):
            return self._build_failure_payload("tool_failed")
        message = "; ".join(message for _code, message in violations[:4]) if violations else None
        return self._build_failure_payload("verification_failed", message=message)

    def _finalize_run_trace(self, state: dict[str, Any], *, outcome: str, failure: dict[str, Any] | None = None) -> dict[str, Any]:
        trace = self._ensure_run_trace(state)
        trace["ended_at"] = self._now_iso()
        trace["outcome"] = "completed" if outcome == "completed" else "failed"
        trace["failure"] = None if outcome == "completed" else dict(failure or self._build_failure_payload("verification_failed"))
        return trace

    def _assert_terminal_consistency(self, trace: dict[str, Any]) -> tuple[bool, str]:
        runtime_module = self._runtime_module()
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
            if not isinstance(failure, dict) or str(failure.get("code") or "") not in runtime_module.FAILURE_CODE_SET:
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

    def _evaluate_goal_completion(self, state: dict[str, Any], final_text: str) -> Any:
        runtime_module = self._runtime_module()
        contract = runtime_module.FinalAnswerContract.from_payload(state.get("final_contract"))
        multi_step_contract = runtime_module.MultiStepContract.from_payload(state.get("multi_step_contract"))
        violations: list[tuple[str, str]] = []
        normalized_final = runtime_module.normalize_whitespace(final_text)
        if not normalized_final:
            violations.append(("empty_final_answer", "Model returned an empty final answer."))
        elif self._looks_like_stale_final_answer(state, final_text):
            violations.append(("invalid_final_answer", "Final answer is stale, polluted by intermediate output, or looks like tool JSON."))

        action_claim = self._detect_action_claim_mismatch(state, final_text)
        if action_claim:
            violations.append(action_claim)

        answer_language = runtime_module.detect_message_language(final_text)
        if contract.requested_language and answer_language and answer_language != contract.requested_language:
            violations.append(
                (
                    "wrong_response_language",
                    f"Expected {runtime_module.describe_language(contract.requested_language)}, got {runtime_module.describe_language(answer_language)}.",
                )
            )

        if contract.exact_output_text and normalized_final and runtime_module.normalize_exact_output(final_text) != runtime_module.normalize_exact_output(contract.exact_output_text):
            violations.append(("exact_output_mismatch", f"Expected exact output `{contract.exact_output_text}`."))

        used_tools = set(state.get("used_tool_names", []))
        urls = runtime_module.extract_urls(final_text)
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

        requires_grounding = contract.require_grounding and bool(used_tools.intersection(runtime_module.GROUNDING_TOOL_NAMES))
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
        if int(multi_step_contract.expected_step_count or 0) > 1 and (assessment.gap.missing_evidence or assessment.gap.missing_facts):
            missing_parts: list[str] = []
            if assessment.gap.missing_evidence:
                missing_parts.append("missing evidence: " + ", ".join(assessment.gap.missing_evidence[:6]))
            if assessment.gap.missing_facts:
                missing_parts.append("missing facts: " + ", ".join(assessment.gap.missing_facts[:6]))
            assessment.mismatch_codes.append(("task_completion_unverified", "; ".join(missing_parts)))
            if assessment.gap.answer_defects is not None:
                assessment.gap.answer_defects = runtime_module._dedupe_strings([*assessment.gap.answer_defects, "task_completion_unverified"])
            assessment.action = "gather_more_evidence" if assessment.gap.missing_evidence and assessment.gap.can_gather_more_evidence else "repair_answer"
            state["goal_gap_summary"] = assessment.gap.to_payload()
        return assessment

    def _collect_contract_violations(self, state: dict[str, Any], final_text: str) -> list[tuple[str, str]]:
        assessment = self._evaluate_goal_completion(state, final_text)
        return list(assessment.mismatch_codes)

    def _build_goal_state(self, state: dict[str, Any]) -> Any:
        runtime_module = self._runtime_module()
        prompt = runtime_module.normalize_whitespace(self._last_user_message(state.get("messages", [])))
        final_contract = runtime_module.FinalAnswerContract.from_payload(state.get("final_contract"))
        multi_step_contract = runtime_module.MultiStepContract.from_payload(state.get("multi_step_contract"))
        required_evidence_kinds: list[str] = []
        required_facts = list(multi_step_contract.required_answer_fields)
        if multi_step_contract.required_inputs:
            required_evidence_kinds.extend(
                f"{item.get('kind')}:{runtime_module.normalize_whitespace(str(item.get('target') or ''))}"
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
        return runtime_module.GoalState(
            user_intent=prompt,
            required_evidence_kinds=runtime_module._dedupe_strings(required_evidence_kinds),
            required_facts=runtime_module._dedupe_strings(required_facts),
            required_citations=bool(final_contract.require_sources or final_contract.require_citations_when_rag_used),
            required_output_constraint={
                "language": final_contract.requested_language,
                "exactOutputText": final_contract.exact_output_text,
                "requireGrounding": final_contract.require_grounding,
                "requireGroundingFromRag": final_contract.require_grounding_from_rag,
            },
        )

    def _build_realization_state(self, state: dict[str, Any], candidate_answer: str) -> Any:
        runtime_module = self._runtime_module()
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
        return runtime_module.RealizationState(
            available_evidence=available_evidence,
            proven_facts=dict(facts),
            candidate_answer=runtime_module.normalize_whitespace(candidate_answer),
        )

    def _build_goal_gap(
        self,
        state: dict[str, Any],
        goal: Any,
        realization: Any,
        base_violations: list[tuple[str, str]] | None = None,
    ) -> Any:
        runtime_module = self._runtime_module()
        evidence_map = dict(state.get("evidence_map") or {})
        contract = runtime_module.MultiStepContract.from_payload(state.get("multi_step_contract"))
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
        return runtime_module.GoalGap(
            missing_evidence=runtime_module._dedupe_strings(missing_evidence),
            missing_facts=runtime_module._dedupe_strings(missing_facts),
            answer_defects=runtime_module._dedupe_strings(answer_defects),
            can_gather_more_evidence=bool(missing_evidence),
            is_complete=not missing_evidence and not missing_facts and not answer_defects,
        )

    def _refresh_goal_tracking(
        self,
        state: dict[str, Any],
        candidate_answer: str | None = None,
        base_violations: list[tuple[str, str]] | None = None,
    ) -> Any:
        runtime_module = self._runtime_module()
        goal = self._build_goal_state(state)
        realization = self._build_realization_state(
            state,
            candidate_answer if candidate_answer is not None else str(state.get("final_text") or ""),
        )
        gap = self._build_goal_gap(state, goal, realization, base_violations)
        action = "finish"
        multi_step_contract = runtime_module.MultiStepContract.from_payload(state.get("multi_step_contract"))
        if (
            runtime_module.is_multistep_contract_active(multi_step_contract)
            and gap.missing_evidence
            and gap.can_gather_more_evidence
            and self._has_runtime_actionable_missing_evidence(gap.missing_evidence)
        ):
            action = "gather_more_evidence"
        elif gap.missing_facts or gap.answer_defects:
            action = "repair_answer"
        assessment = runtime_module.GoalAssessment(
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

    def _goal_gap_events(self, run_id: str, state: dict[str, Any], reason: str) -> list[dict[str, Any]]:
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
        contract: Any,
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

    def _build_final_repair_prompt(self, state: dict[str, Any], violations: list[tuple[str, str]]) -> str:
        runtime_module = self._runtime_module()
        contract = runtime_module.FinalAnswerContract.from_payload(state.get("final_contract"))
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
            lines.append(f"Answer in {runtime_module.describe_language(contract.requested_language)}.")
        if contract.exact_output_text:
            lines.append(f"Return exactly this text and nothing else: {contract.exact_output_text}")
        if contract.require_sources or "web_search" in set(state.get("used_tool_names", [])):
            lines.append("Add a separate Sources section with at least two Markdown links like [Title](https://...).")
            lines.append("Keep URLs out of the main body unless the user explicitly asks to display raw links.")
        if contract.require_grounding and set(state.get("used_tool_names", [])).intersection(runtime_module.GROUNDING_TOOL_NAMES):
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

    def _synthesize_fallback_final_answer(self, state: dict[str, Any]) -> str | None:
        runtime_module = self._runtime_module()
        contract = runtime_module.FinalAnswerContract.from_payload(state.get("final_contract"))
        multi_step_contract = runtime_module.MultiStepContract.from_payload(state.get("multi_step_contract"))
        evidence_map = dict(state.get("evidence_map") or {})
        facts = evidence_map.get("facts", {}) if isinstance(evidence_map.get("facts"), dict) else {}
        if contract.exact_output_text:
            return contract.exact_output_text

        if runtime_module.is_multistep_contract_active(multi_step_contract) and multi_step_contract.required_answer_fields:
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

    def _runtime_module(self):
        from streamlit_python_only import runtime as runtime_module

        return runtime_module
