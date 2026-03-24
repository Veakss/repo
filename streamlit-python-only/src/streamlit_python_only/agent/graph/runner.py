from __future__ import annotations

from copy import deepcopy
from typing import Any
import json
import uuid

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from streamlit_python_only.agent.graph.factory import create_graph
from streamlit_python_only.events import run_diagnostic
from streamlit_python_only.graph_state import AgentState, MultiStepContract

GRAPH_MAX_REPAIR_ATTEMPTS = 3


class AgentGraphRunner:
    """Dedicated LangGraph execution engine that owns the LangGraph node logic."""

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.deps = engine.deps
        self.graph = create_graph(self)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.engine, name)

    def _normalize_ws(self, value: str) -> str:
        return " ".join(str(value or "").split()).strip()

    def _parse_text_tool_calls(self, content: Any) -> list[dict[str, Any]]:
        trimmed = str(content or "").strip()
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

    def _sync_graph_state_slices(self, state: AgentState) -> None:
        state["conversation"] = {
            "messages": len(state.get("messages", [])),
            "sessionId": state.get("session_id"),
            "runId": state.get("run_id"),
        }
        state["runtime"] = {
            "workspaceRoot": state.get("workspace_root"),
            "profile": state.get("profile"),
            "model": state.get("model"),
            "providerCapabilities": dict(state.get("provider_capabilities") or {}),
            "policyProfile": state.get("policy_profile"),
        }
        state["goal"] = {
            "summary": dict(state.get("goal_summary") or {}),
            "contract": dict(state.get("final_contract") or {}),
            "multiStep": dict(state.get("multi_step_contract") or {}),
        }
        state["procedure_state"] = dict(self._build_procedure_state(state))
        state["evidence"] = {
            "summary": dict(state.get("realization_summary") or {}),
            "gap": dict(state.get("goal_gap_summary") or {}),
            "map": dict(state.get("evidence_map") or {}),
            "toolResults": len(state.get("tool_results", [])),
            "writtenFiles": list(state.get("written_files", [])),
            "lastReadFilePath": state.get("last_read_file_path"),
            "lastOpenedUrl": state.get("last_opened_url"),
        }
        state["decision"] = {
            "verifyAction": state.get("verify_action"),
            "graphRoute": state.get("graph_route"),
            "pendingApproval": bool(state.get("pending_approval")),
            "pendingClarification": bool(state.get("pending_clarification")),
        }
        state["control"] = {
            "noProgressCount": int(state.get("no_progress_turns", 0) or 0),
            "repairCount": int(state.get("repair_attempts", 0) or 0),
            "loopCount": len(state.get("tool_results", [])),
            "needsClarification": bool(state.get("pending_clarification")),
        }
        state["output"] = {
            "candidateFinal": state.get("final_text", ""),
            "validatedFinal": state.get("final_text", ""),
        }

    def _preflight_node(self, state: AgentState) -> AgentState:
        updated = dict(state)
        self._refresh_goal_tracking(updated, str(updated.get("final_text", "") or ""))
        updated["graph_route"] = None
        self._sync_graph_state_slices(updated)
        return updated

    def _clarify_gate_node(self, state: AgentState) -> AgentState:
        updated = dict(state)
        updated["graph_route"] = "clarify" if updated.get("pending_clarification") else "agent"
        self._sync_graph_state_slices(updated)
        return updated

    def _clarify_node(self, state: AgentState) -> AgentState:
        updated = dict(state)
        updated["graph_route"] = "finish"
        self._sync_graph_state_slices(updated)
        return updated

    def _agent_node(self, state: AgentState) -> AgentState:
        registry = self.deps.tool_registry_factory(state["workspace_root"], state["session_id"], state["run_id"], state.get("tool_toggles"))
        model = self.deps.model_factory(state.get("profile"), state.get("model"))
        provider = self.deps.provider_resolver(state.get("profile"), state.get("model"))
        response = model.bind_tools(registry.enabled_tools()).invoke(state["messages"])
        final_text = response.content if isinstance(response.content, str) else str(response.content or "")
        updated: AgentState = {
            **state,
            "messages": [response],
            "provider_mode": provider.mode,
            "provider_capabilities": self._provider_capabilities_payload(provider),
            "final_text": final_text,
            "graph_route": None,
        }
        self._sync_graph_state_slices(updated)
        return updated

    def _tool_router_node(self, state: AgentState) -> AgentState:
        last_message = state["messages"][-1]
        route = "verify"
        if isinstance(last_message, AIMessage):
            has_calls = (last_message.tool_calls and len(last_message.tool_calls) > 0) or self._parse_text_tool_calls(last_message.content)
            if has_calls:
                route = "tool_executor"
        updated = dict(state)
        updated["graph_route"] = route
        self._sync_graph_state_slices(updated)
        return updated

    def _tools_node(self, state: AgentState) -> AgentState:
        registry = self.deps.tool_registry_factory(state["workspace_root"], state["session_id"], state["run_id"], state.get("tool_toggles"))
        last_message = state["messages"][-1]
        emitted_messages: list[BaseMessage] = []
        tool_events: list[dict[str, Any]] = []
        pending_approval = None
        pending_clarification = None
        tool_results = list(state.get("tool_results", []))
        used_tool_names = list(state.get("used_tool_names", []))
        evidence_map = dict(state.get("evidence_map") or {})
        tool_calls: list[dict[str, Any]] = []

        if isinstance(last_message, AIMessage):
            tool_calls = list(last_message.tool_calls or [])
            if not tool_calls:
                tool_calls = self._parse_text_tool_calls(last_message.content)
        if len(tool_calls) > 1 and self._provider_requires_sequential_tools(state):
            original_count = len(tool_calls)
            tool_calls = tool_calls[:1]
            tool_events.append(
                run_diagnostic(
                    state["run_id"],
                    "provider_tool_calls_collapsed",
                    f"Provider allows only one tool call per turn; collapsed {original_count} tool calls to the first one.",
                    level="warn",
                )
            )

        for call in tool_calls:
            tool_name = str(call["name"])
            tool_args = dict(call.get("args", {}) or {})
            tool_id = str(call.get("id") or uuid.uuid4())
            registered = registry.get(tool_name)
            if not registered or not registered.enabled:
                continue
            normalized_args = tool_args
            try:
                normalized_args, normalization_events = self._normalize_tool_args(state, tool_name, tool_args)
                tool_events.extend(normalization_events)
            except Exception as exc:
                failure_events, message_result = self._tool_failure_result(state["run_id"], tool_name, tool_id, exc)
                tool_events.extend(
                    [
                        {
                            "type": "tool_call",
                            "runId": state["run_id"],
                            "actionId": tool_id,
                            "name": tool_name,
                            "arguments": json.dumps(tool_args, ensure_ascii=False),
                            "riskLevel": registered.risk_level,
                        },
                        *failure_events,
                    ]
                )
                if state["provider_mode"] == "textual_replay":
                    emitted_messages.append(self._build_tool_replay_message(tool_name, tool_args, message_result, "failed"))
                else:
                    emitted_messages.append(ToolMessage(content=message_result, tool_call_id=tool_id))
                emitted_messages.append(self._build_tool_followup_guidance(state, tool_name, message_result, "failed"))
                tool_results.append({"tool": tool_name, "result": message_result, "status": "failed"})
                used_tool_names.append(tool_name)
                continue

            tool_events.append(
                {
                    "type": "tool_call",
                    "runId": state["run_id"],
                    "actionId": tool_id,
                    "name": tool_name,
                    "arguments": json.dumps(normalized_args, ensure_ascii=False),
                    "riskLevel": registered.risk_level,
                }
            )
            loop_guard = self._should_stop_for_repeated_tool_call(state, tool_name, normalized_args)
            if loop_guard.get("stop"):
                tool_events.append(
                    run_diagnostic(state["run_id"], "repeated_tool_call_loop", str(loop_guard.get("reason") or "Repeated tool call loop."), level="warn")
                )
                break
            if tool_name == "request_clarification":
                options = []
                for key in ("option_a", "option_b", "option_c"):
                    value = normalized_args.get(key)
                    if isinstance(value, str) and value.strip():
                        options.append({"label": value.strip(), "value": value.strip()})
                clarification_question = str(normalized_args.get("question", "Could you clarify?")).strip() or "Could you clarify?"
                emitted_messages.append(
                    SystemMessage(
                        content="\n".join(
                            [
                                "Clarification requested and awaiting the user response.",
                                f"Clarification question: {clarification_question}",
                                *(
                                    ["Suggested options: " + ", ".join(str(option.get("label") or "").strip() for option in options if str(option.get("label") or "").strip())]
                                    if options
                                    else []
                                ),
                            ]
                        )
                    )
                )
                pending_clarification = {
                    "clarificationId": str(uuid.uuid4()),
                    "runId": state["run_id"],
                    "question": clarification_question,
                    "options": options,
                }
                break
            if self._should_request_approval(state["policy_profile"], registered.risk_level):
                pending_approval = {
                    "approvalId": str(uuid.uuid4()),
                    "runId": state["run_id"],
                    "name": tool_name,
                    "arguments": normalized_args,
                    "riskLevel": registered.risk_level,
                    "toolCallId": tool_id,
                }
                break
            try:
                result = str(registered.tool.invoke(normalized_args))
                result_events, message_result, _ok, tool_meta = self._materialize_tool_result(state["run_id"], tool_name, tool_id, result)
                tool_events.extend(result_events)
                if state["provider_mode"] == "textual_replay":
                    emitted_messages.append(self._build_tool_replay_message(tool_name, normalized_args, message_result, "succeeded"))
                else:
                    emitted_messages.append(ToolMessage(content=message_result, tool_call_id=tool_id))
                emitted_messages.append(self._build_tool_followup_guidance(state, tool_name, message_result, "succeeded", tool_meta.get("meta") if tool_meta else None))
                tool_results.append(
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
                )
                evidence_map = self._update_evidence_map(state, tool_name, normalized_args, message_result, tool_meta)
                state["evidence_map"] = evidence_map
                self._note_tool_success(state, tool_name, normalized_args, message_result, tool_meta)
                used_tool_names.append(tool_name)
                post_tool_hint = self._maybe_build_post_tool_reasoning_hint(state, tool_name, message_result, tool_meta)
                if post_tool_hint:
                    emitted_messages.append(SystemMessage(content=post_tool_hint))
            except Exception as exc:
                failure_events, message_result = self._tool_failure_result(state["run_id"], tool_name, tool_id, exc)
                tool_events.extend(failure_events)
                if state["provider_mode"] == "textual_replay":
                    emitted_messages.append(self._build_tool_replay_message(tool_name, normalized_args, message_result, "failed"))
                else:
                    emitted_messages.append(ToolMessage(content=message_result, tool_call_id=tool_id))
                emitted_messages.append(self._build_tool_followup_guidance(state, tool_name, message_result, "failed"))
                tool_results.append({"tool": tool_name, "result": message_result, "status": "failed"})
                used_tool_names.append(tool_name)
                self._note_executed_tool(state, tool_name)

        updated_state: AgentState = {
            **state,
            "messages": emitted_messages,
            "pending_approval": pending_approval,
            "pending_clarification": pending_clarification,
            "tool_events": list(state.get("tool_events", [])) + tool_events,
            "evidence_map": evidence_map,
            "tool_results": tool_results,
            "used_tool_names": used_tool_names,
            "executed_tools": state.get("executed_tools", []),
            "recent_tool_signatures": state.get("recent_tool_signatures", []),
            "repeated_tool_signature_streak": state.get("repeated_tool_signature_streak"),
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
            "graph_route": "finish" if pending_approval or pending_clarification else ("verify" if self._provider_requires_sequential_tools(state) else "agent"),
        }
        self._refresh_goal_tracking(updated_state, "")
        self._sync_graph_state_slices(updated_state)
        return updated_state

    def _state_update_node(self, state: AgentState) -> AgentState:
        updated = dict(state)
        self._refresh_goal_tracking(updated, str(updated.get("final_text", "") or ""))
        if updated.get("pending_approval") or updated.get("pending_clarification"):
            updated["graph_route"] = "finish"
        elif self._provider_requires_sequential_tools(updated):
            updated["graph_route"] = "verify"
        else:
            updated["graph_route"] = "agent"
        self._sync_graph_state_slices(updated)
        return updated

    def _verify_node(self, state: AgentState) -> AgentState:
        updated = self._graph_verify_state_for_loop(dict(state))
        route = "finish"
        if updated.get("pending_clarification"):
            route = "clarify"
        elif updated.get("verify_action") == "agent":
            route = "agent"
        elif updated.get("verify_action") == "repair":
            route = "repair"
        elif updated.get("verify_action") == "finish":
            route = "finish"
        elif updated.get("verify_action") == "end":
            violations = self._graph_collect_contract_violations(updated, updated.get("final_text", ""))
            if violations and not updated.get("final_repair_attempted") and int(updated.get("repair_attempts", 0) or 0) < GRAPH_MAX_REPAIR_ATTEMPTS:
                route = "repair"
            else:
                route = "finish"
        updated["graph_route"] = route
        self._sync_graph_state_slices(updated)
        return updated

    def _repair_node(self, state: AgentState) -> AgentState:
        violations = self._graph_collect_contract_violations(state, state.get("final_text", ""))
        if not violations:
            updated = dict(state)
            updated["graph_route"] = "finish"
            self._sync_graph_state_slices(updated)
            return updated
        repaired_state, repair_events = self._graph_invoke_final_repair(state, violations)
        repaired_state["tool_events"] = list(repaired_state.get("tool_events", [])) + repair_events
        repaired_state["verify_action"] = "end"
        repaired_state["graph_route"] = "verify"
        self._sync_graph_state_slices(repaired_state)
        return repaired_state

    def _finish_node(self, state: AgentState) -> AgentState:
        updated = self._graph_finalize_result_contract(dict(state))
        updated["graph_route"] = "finish"
        self._sync_graph_state_slices(updated)
        return updated

    def _build_canonical_tool_replay_message(
        self,
        *,
        tool_name: str,
        tool_args: dict[str, Any],
        result_text: str,
        status: str,
        task: str,
        done: list[str],
        remaining: list[str],
    ) -> SystemMessage:
        done_text = "\n".join(done) if done else "- No verified completed items recorded yet."
        remaining_text = (
            "\n".join(remaining) if remaining else "- Either answer now if the goal is satisfied, or choose one next useful action."
        )
        return SystemMessage(
            content=(
                "Tool execution replay for provider compatibility.\n\n"
                "TASK\n"
                f"{task.strip()}\n\n"
                "LAST_ACTION\n"
                f"tool={tool_name}\n"
                f"status={status}\n"
                f"arguments={json.dumps(tool_args, ensure_ascii=False)}\n\n"
                "OBSERVATION\n"
                f"{result_text}\n\n"
                "DONE\n"
                f"{done_text}\n\n"
                "REMAINING\n"
                f"{remaining_text}\n\n"
                "NEXT_ACTION_RULE\n"
                "Choose exactly one next action. Either call one tool, answer finally, or ask for clarification. "
                "Do not rely on OpenAI tool-role replay."
            )
        )

    def _procedure_done_items(self, state: dict[str, Any]) -> list[str]:
        items: list[str] = []
        for result in state.get("tool_results", []):
            tool = str(result.get("tool") or "").strip()
            status = str(result.get("status") or "").strip()
            if tool and status == "succeeded":
                items.append(f"- {tool} succeeded")
        return items[-6:]

    def _procedure_remaining_items(self, state: dict[str, Any]) -> list[str]:
        pending: list[str] = []
        if state.get("pending_approval"):
            pending.append("- Approval is required before the next action.")
        if state.get("pending_clarification"):
            pending.append("- Clarification is required before continuing.")
        contract = dict(state.get("final_contract") or {})
        if contract.get("requireSources"):
            pending.append("- If answering now, include the required sources.")
        if self._provider_requires_sequential_tools(state):
            pending.append("- Provider is sequential: emit at most one tool call in the next turn.")
        return pending[:6]

    def _canonicalize_textual_replay_messages(self, state: dict[str, Any]) -> None:
        if state.get("provider_mode") != "textual_replay":
            return
        messages = list(state.get("messages", []))
        if not messages:
            return
        last_message = messages[-1]
        content = str(getattr(last_message, "content", "") or "")
        if not content.strip():
            return
        tool_results = list(state.get("tool_results", []))
        if not tool_results:
            return
        last_result = tool_results[-1]
        tool_name = str(last_result.get("tool") or "").strip()
        if not tool_name:
            return
        status = str(last_result.get("status") or "succeeded").strip() or "succeeded"
        result_text = str(last_result.get("result") or content)
        replacement = self._build_canonical_tool_replay_message(
            tool_name=tool_name,
            tool_args=dict((last_result.get("meta") or {}).get("arguments") or {}),
            result_text=result_text,
            status=status,
            task=str(self._last_user_message(state.get("messages", [])) or "Continue the current user request."),
            done=self._procedure_done_items(state),
            remaining=self._procedure_remaining_items(state),
        )
        messages[-1] = replacement
        state["messages"] = messages

    def _decision_validate_node(self, state: dict[str, Any]) -> dict[str, Any]:
        updated = dict(state)
        updated["graph_route"] = "tool_router"
        last_message = updated.get("messages", [])[-1] if updated.get("messages") else None
        if isinstance(last_message, AIMessage):
            tool_calls = list(last_message.tool_calls or [])
            parsed_text_calls = self._parse_text_tool_calls(last_message.content)
            final_candidate = str(last_message.content or "").strip()
            if not tool_calls and not parsed_text_calls and not final_candidate:
                evidence_map = dict(updated.get("evidence_map") or {})
                has_structured_evidence = any(
                    bool(evidence_map.get(key))
                    for key in ("files", "directories", "terminals", "facts")
                )
                has_recoverable_context = bool(updated.get("tool_results")) or has_structured_evidence or len(updated.get("messages", [])) > 2
                if has_recoverable_context:
                    updated["graph_route"] = "verify"
                else:
                    updated["graph_route"] = "fail"
                    updated["verify_action"] = "fail"
        self._sync_graph_state_slices(updated)
        return updated

    def _fail_node(self, state: dict[str, Any]) -> dict[str, Any]:
        updated = dict(state)
        updated["graph_route"] = "fail"
        updated["verify_action"] = "fail"
        self._sync_graph_state_slices(updated)
        return updated

    def _state_reduce_node(self, state: dict[str, Any]) -> dict[str, Any]:
        updated = self._state_update_node(state)
        self._canonicalize_textual_replay_messages(updated)
        updated["graph_route"] = updated.get("graph_route")
        self._sync_graph_state_slices(updated)
        return updated

    def prepare_initial_chat_state(self, state: dict[str, Any]) -> dict[str, Any]:
        updated = deepcopy(state)
        clarification_enabled = bool((updated.get("tool_toggles") or {}).get("clarification", True))
        early_clarification = self._build_early_clarification(self._last_user_message(updated["messages"]), clarification_enabled)
        if early_clarification:
            clarification_id = str(uuid.uuid4())
            updated["pending_clarification"] = {
                "clarificationId": clarification_id,
                "runId": updated["run_id"],
                "question": early_clarification["questions"][0],
                "questions": early_clarification["questions"],
                "options": early_clarification.get("options", []),
                "reason": early_clarification["reason"],
            }
            updated["tool_events"] = [
                run_diagnostic(updated["run_id"], "clarification_needed", f"Clarification requested: {early_clarification['reason']}"),
                {
                    "type": "tool_call",
                    "runId": updated["run_id"],
                    "actionId": clarification_id,
                    "name": "request_clarification",
                    "arguments": json.dumps({"question": early_clarification["questions"][0]}, ensure_ascii=False),
                    "riskLevel": "safe",
                },
            ]
        self._sync_graph_state_slices(updated)
        return updated

    def prepare_approval_resume_state(self, state: dict[str, Any], decision: str) -> dict[str, Any]:
        updated = deepcopy(state)
        pending = dict(updated.get("pending_approval") or {})
        if not pending:
            return updated
        if decision == "approved":
            registry = self.deps.tool_registry_factory(
                updated["workspace_root"],
                updated["session_id"],
                updated["run_id"],
                updated.get("tool_toggles"),
            )
            registered = registry.get(pending["name"])
            if not registered:
                updated["resume_failure"] = {"code": "tool_failed", "message": "Approved tool is no longer available."}
                return updated
            normalized_args, normalization_events = self._normalize_tool_args(updated, pending["name"], dict(pending["arguments"] or {}))
            pending["arguments"] = normalized_args
            result = str(registered.tool.invoke(normalized_args))
            result_events, message_result, _ok, tool_meta = self._materialize_tool_result(updated["run_id"], pending["name"], pending["toolCallId"], result)
            updated["tool_events"] = list(updated.get("tool_events", [])) + list(normalization_events) + list(result_events)
            if updated["provider_mode"] == "textual_replay":
                replay = self._build_canonical_tool_replay_message(
                    tool_name=pending["name"],
                    tool_args=normalized_args,
                    result_text=message_result,
                    status="succeeded",
                    task=str(self._last_user_message(updated.get("messages", [])) or "Continue the current user request."),
                    done=self._procedure_done_items(updated),
                    remaining=self._procedure_remaining_items(updated),
                )
                updated["messages"] = updated["messages"] + [replay]
            else:
                updated["messages"] = updated["messages"] + [ToolMessage(content=message_result, tool_call_id=pending["toolCallId"])]
            updated["messages"] = updated["messages"] + [
                self._build_tool_followup_guidance(updated, pending["name"], message_result, "succeeded", tool_meta.get("meta") if tool_meta else None)
            ]
            updated["tool_results"] = list(updated.get("tool_results", [])) + [
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
            updated["evidence_map"] = self._update_evidence_map(updated, pending["name"], normalized_args, message_result, tool_meta)
            self._note_tool_success(updated, pending["name"], normalized_args, message_result, tool_meta)
            post_tool_hint = self._maybe_build_post_tool_reasoning_hint(updated, pending["name"], message_result, tool_meta)
            if post_tool_hint:
                updated["messages"] = updated["messages"] + [SystemMessage(content=post_tool_hint)]
            updated["used_tool_names"] = list(updated.get("used_tool_names", [])) + [pending["name"]]
        else:
            rejection = (
                self._build_canonical_tool_replay_message(
                    tool_name=pending["name"],
                    tool_args=dict(pending.get("arguments") or {}),
                    result_text="Rejected by user approval policy.",
                    status="rejected",
                    task=str(self._last_user_message(updated.get("messages", [])) or "Continue the current user request."),
                    done=self._procedure_done_items(updated),
                    remaining=self._procedure_remaining_items(updated),
                )
                if updated["provider_mode"] == "textual_replay"
                else SystemMessage(content=f"Tool {pending['name']} was rejected by the user. Continue without this tool.")
            )
            updated["messages"] = updated["messages"] + [rejection]
        updated["pending_approval"] = None
        updated["pending_clarification"] = None
        updated["final_text"] = ""
        self._sync_graph_state_slices(updated)
        return updated

    def prepare_clarification_resume_state(self, state: dict[str, Any], answer: str) -> dict[str, Any]:
        updated = deepcopy(state)
        clarification = updated.get("pending_clarification")
        updated["messages"] = updated["messages"] + [
            self._build_clarification_resume_message(clarification, answer),
            HumanMessage(content=answer),
        ]
        updated["pending_approval"] = None
        updated["pending_clarification"] = None
        updated["final_text"] = ""
        clarification_enabled = bool((updated.get("tool_toggles") or {}).get("clarification", True))
        early_clarification = self._build_early_clarification(answer, clarification_enabled)
        if early_clarification:
            clarification_id = str(uuid.uuid4())
            updated["pending_clarification"] = {
                "clarificationId": clarification_id,
                "runId": updated["run_id"],
                "question": early_clarification["questions"][0],
                "questions": early_clarification["questions"],
                "options": early_clarification.get("options", []),
                "reason": early_clarification["reason"],
            }
            updated["tool_events"] = [
                run_diagnostic(updated["run_id"], "clarification_needed", f"Clarification requested: {early_clarification['reason']}"),
                {
                    "type": "tool_call",
                    "runId": updated["run_id"],
                    "actionId": clarification_id,
                    "name": "request_clarification",
                    "arguments": json.dumps({"question": early_clarification["questions"][0]}, ensure_ascii=False),
                    "riskLevel": "safe",
                },
            ]
        self._sync_graph_state_slices(updated)
        return updated

    def _graph_collect_contract_violations(self, state: dict[str, Any], final_text: str) -> list[tuple[str, str]]:
        return self.engine._collect_contract_violations(state, final_text)

    def _graph_invoke_final_repair(self, state: dict[str, Any], violations: list[tuple[str, str]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
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
            ),
        ]
        repair_prompt = self.engine._build_final_repair_prompt(state, violations)
        model = self.deps.model_factory(state.get("profile"), state.get("model"))
        response = model.invoke(list(state["messages"]) + [SystemMessage(content=repair_prompt)])
        repaired_text = response.content if isinstance(response.content, str) else str(response.content or "")
        repaired_text = str(repaired_text or "").strip()
        if not repaired_text:
            self._bump_no_progress(state)
            fallback = self.engine._synthesize_fallback_final_answer(state)
            if fallback:
                repaired_text = fallback
        repaired_state: dict[str, Any] = {
            **state,
            "messages": list(state["messages"]) + [SystemMessage(content=repair_prompt), AIMessage(content=repaired_text)],
            "final_text": repaired_text,
            "final_repair_attempted": True,
        }
        remaining = self._graph_collect_contract_violations(repaired_state, repaired_text)
        if remaining:
            fallback = self.engine._synthesize_fallback_final_answer(state)
            if fallback and self._normalize_ws(fallback) != self._normalize_ws(repaired_text):
                repaired_text = fallback
                repaired_state = {
                    **state,
                    "messages": list(state["messages"]) + [SystemMessage(content=repair_prompt), AIMessage(content=repaired_text)],
                    "final_text": repaired_text,
                    "final_repair_attempted": True,
                }
                remaining = self._graph_collect_contract_violations(repaired_state, repaired_text)
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

    def _graph_finalize_result_contract(self, state: dict[str, Any]) -> dict[str, Any]:
        if state.get("pending_approval") or state.get("pending_clarification"):
            return state
        assessment = self.engine._evaluate_goal_completion(state, state.get("final_text", ""))
        multistep_violations = self.engine._collect_multistep_violations(state, state.get("final_text", ""))
        multistep_events: list[dict[str, Any]] = []
        multistep_events.extend(self.engine._goal_gap_events(state["run_id"], state, "finalize_result"))
        multi_step_contract = MultiStepContract.from_payload(state.get("multi_step_contract"))
        if int(multi_step_contract.expected_step_count or 0) > 1:
            multistep_events.extend(
                self.engine._build_multistep_progress_events(
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
        if int(multi_step_contract.expected_step_count or 0) > 1 and getattr(assessment.gap, "is_complete", False) and not violations:
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
        if violations and not state.get("final_repair_attempted") and int(state.get("repair_attempts", 0) or 0) < GRAPH_MAX_REPAIR_ATTEMPTS:
            repaired_state, repair_events = self._graph_invoke_final_repair(state, violations)
            repaired_state["tool_events"] = list(repaired_state.get("tool_events", [])) + extra_events + repair_events
            return repaired_state
        if extra_events:
            state["tool_events"] = list(state.get("tool_events", [])) + extra_events
        return state

    def _graph_verify_state_for_loop(self, state: dict[str, Any]) -> dict[str, Any]:
        if state.get("pending_approval") or state.get("pending_clarification"):
            return {**state, "verify_action": "end"}

        assessment = self.engine._evaluate_goal_completion(state, state.get("final_text", ""))
        multistep_violations = self.engine._collect_multistep_violations(state, state.get("final_text", ""))
        tool_events = list(state.get("tool_events", [])) + self.engine._goal_gap_events(state["run_id"], state, "verify_loop")
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
            if missing_inputs and self.engine._has_runtime_actionable_missing_evidence(missing_inputs):
                prior_signature = str((state.get("evidence_map") or {}).get("signature") or "")
                acquired_state, _events, acquired = self.engine._run_explicit_evidence_acquisition(state, missing_inputs[0])
                if acquired:
                    updated_signature = str((acquired_state.get("evidence_map") or {}).get("signature") or "")
                    acquired_state["multistep_no_progress_turns"] = 0 if updated_signature != prior_signature else int(state.get("multistep_no_progress_turns", 0) or 0) + 1
                    acquired_state["multistep_repair_attempted"] = updated_signature == prior_signature
                    acquired_state["verify_action"] = "agent"
                    acquired_state["tool_events"] = list(acquired_state.get("tool_events", [])) + tool_events
                    return acquired_state
            continue_prompt = self.engine._build_multistep_continue_prompt(
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

        if violations and not state.get("final_repair_attempted") and int(state.get("repair_attempts", 0) or 0) < GRAPH_MAX_REPAIR_ATTEMPTS:
            repaired_state, repair_events = self._graph_invoke_final_repair(state, violations)
            repaired_state["tool_events"] = list(repaired_state.get("tool_events", [])) + tool_events + repair_events
            repaired_state["verify_action"] = "end"
            return repaired_state

        return {
            **state,
            "tool_events": tool_events,
            "verify_action": "end",
        }
