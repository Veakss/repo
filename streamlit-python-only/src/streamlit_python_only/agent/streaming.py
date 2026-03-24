from __future__ import annotations

import asyncio
import json
from typing import Any

from streamlit_python_only.events import (
    assistant_progress,
    approval_required,
    clarification_required,
    done,
    run_diagnostic,
    run_phase,
    run_state,
    run_step,
    token,
)


class AgentRunStreamer:
    def __init__(self, owner: Any) -> None:
        self.owner = owner

    def __getattr__(self, name: str) -> Any:
        return getattr(self.owner, name)

    def _runtime_module(self):
        from streamlit_python_only import runtime as runtime_module

        return runtime_module

    def _build_run_metrics_event(self, state: dict[str, Any]) -> dict[str, Any]:
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

    def _build_initial_run_trace(self, state: dict[str, Any]) -> dict[str, Any]:
        runtime_module = self._runtime_module()
        goal_summary = dict(state.get("goal_summary") or {})
        goal_text = runtime_module.normalize_whitespace(
            str(goal_summary.get("userIntent") or self._last_user_message(state.get("messages", [])) or "")
        )
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
            "_step_emit_cursor": 0,
        }

    def _ensure_run_trace(self, state: dict[str, Any]) -> dict[str, Any]:
        runtime_module = self._runtime_module()
        trace = state.get("run_trace")
        if not isinstance(trace, dict):
            trace = self._build_initial_run_trace(state)
            state["run_trace"] = trace
        trace.setdefault("run_id", state.get("run_id"))
        trace.setdefault("session_id", state.get("session_id"))
        trace.setdefault("started_at", self._now_iso())
        trace.setdefault(
            "goal",
            runtime_module.normalize_whitespace(
                str((state.get("goal_summary") or {}).get("userIntent") or self._last_user_message(state.get("messages", [])) or "")
            ),
        )
        trace.setdefault("steps", [])
        trace.setdefault("outcome", None)
        trace.setdefault("failure", None)
        trace.setdefault("_event_cursor", 0)
        trace.setdefault("_step_emit_cursor", 0)
        return trace

    def _append_run_trace_step(
        self,
        state: dict[str, Any],
        *,
        kind: str,
        status: str,
        summary: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        runtime_module = self._runtime_module()
        trace = self._ensure_run_trace(state)
        steps = trace.get("steps")
        if not isinstance(steps, list):
            steps = []
            trace["steps"] = steps
        step = {
            "index": len(steps) + 1,
            "kind": kind,
            "status": status,
            "summary": runtime_module.normalize_whitespace(summary),
            "timestamp": self._now_iso(),
        }
        if data:
            step["data"] = data
        steps.append(step)

    def _append_trace_steps_from_tool_events(self, state: dict[str, Any]) -> None:
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

    def _record_progress_message(self, state: dict[str, Any], summary: str, *, source: str = "runtime") -> dict[str, Any] | None:
        runtime_module = self._runtime_module()
        cleaned = runtime_module.normalize_whitespace(summary)
        if not cleaned:
            return None
        if len(cleaned) > 280:
            cleaned = cleaned[:277].rstrip() + "..."
        signature = f"{source}:{cleaned.lower()}"
        if signature == str(state.get("last_progress_hash") or ""):
            return None
        progress_messages = list(state.get("progress_messages", []))
        payload = {
            "stepIndex": len(self._ensure_run_trace(state).get("steps", [])),
            "summary": cleaned,
            "source": source,
            "timestamp": self._now_iso(),
        }
        progress_messages.append(payload)
        state["progress_messages"] = progress_messages[-40:]
        state["last_progress_hash"] = signature
        return payload

    def _progress_summary_from_event(self, event: dict[str, Any]) -> str | None:
        event_type = str(event.get("type") or "")
        if event_type == "tool_call":
            tool_name = str(event.get("name") or "tool")
            return f"Action en cours: appel de `{tool_name}`."
        if event_type == "tool_result":
            tool_name = str(event.get("name") or "tool")
            ok = bool(event.get("ok"))
            return f"Action terminée: `{tool_name}` {'réussi' if ok else 'a échoué'}."
        if event_type == "run_diagnostic":
            code = str(event.get("code") or "")
            if code == "goal_gap_summary":
                return "Vérification: mise à jour du gap objectif/réalisation."
            if code in {"task_completion_unverified", "verify_guardrail_repair"}:
                return "Vérification: le run continue pour fermer le gap."
        return None

    def _collect_new_run_step_events(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        trace = self._ensure_run_trace(state)
        steps = trace.get("steps", [])
        if not isinstance(steps, list):
            return []
        cursor = int(trace.get("_step_emit_cursor", 0) or 0)
        if cursor >= len(steps):
            return []
        emitted: list[dict[str, Any]] = []
        for step in steps[cursor:]:
            emitted.append(
                run_step(
                    str(state.get("run_id") or ""),
                    int(step.get("index") or 0),
                    str(step.get("kind") or "step"),
                    str(step.get("status") or "unknown"),
                    str(step.get("summary") or ""),
                )
            )
        trace["_step_emit_cursor"] = len(steps)
        return emitted

    async def _stream_result(self, run_id: str, result: dict[str, Any]):
        runtime_module = self._runtime_module()
        self._append_trace_steps_from_tool_events(result)
        for event in result.get("tool_events", []):
            yield event
            progress_summary = self._progress_summary_from_event(event)
            if progress_summary:
                progress = self._record_progress_message(result, progress_summary, source="runtime")
                if progress:
                    yield assistant_progress(
                        run_id,
                        int(progress.get("stepIndex") or 0),
                        str(progress.get("summary") or ""),
                        str(progress.get("source") or "runtime"),
                    )
        for step_event in self._collect_new_run_step_events(result):
            yield step_event
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
            for step_event in self._collect_new_run_step_events(result):
                yield step_event
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
            for step_event in self._collect_new_run_step_events(result):
                yield step_event
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
        final_text = runtime_module.normalize_final_markdown(str(result.get("final_text", "") or ""))
        if not final_text:
            fallback = self._synthesize_fallback_final_answer(result)
            if fallback:
                final_text = runtime_module.normalize_final_markdown(fallback)
        final_text = runtime_module.normalize_final_markdown(final_text)
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
        verify_progress = self._record_progress_message(
            result,
            "Vérification finale: gap fermé." if verify_ok else "Vérification finale: ajustements encore nécessaires.",
            source="runtime",
        )
        if verify_progress:
            yield assistant_progress(
                run_id,
                int(verify_progress.get("stepIndex") or 0),
                str(verify_progress.get("summary") or ""),
                str(verify_progress.get("source") or "runtime"),
            )
        for step_event in self._collect_new_run_step_events(result):
            yield step_event
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
                runtime_module.normalize_whitespace(final_text)
                and failure.get("code") in {"missing_evidence", "verification_failed", "no_progress_limit"}
                and not violation_codes.intersection({"empty_final_answer", "invalid_final_answer"})
            ):
                visible_text = final_text
            else:
                visible_text = runtime_module.normalize_final_markdown(self._build_failure_final_text(failure))
            self._append_run_trace_step(
                result,
                kind="finish",
                status="failed",
                summary=f"Run failed: {failure['code']}.",
                data={"code": failure["code"]},
            )
            trace = self._finalize_run_trace(result, outcome="failed", failure=failure)
            end_state = "failed"
        for step_event in self._collect_new_run_step_events(result):
            yield step_event
        for piece in runtime_module.split_for_streaming(visible_text):
            yield token(piece)
            await asyncio.sleep(0)
        yield self._build_run_metrics_event(result)
        ok, message = self._assert_terminal_consistency(trace)
        if not ok:
            yield run_diagnostic(run_id, "assert_terminal_consistency", message, level="warn")
        yield run_state(run_id, end_state)
        yield done(run_id=run_id, run_trace=self._run_trace_summary(trace))
