from __future__ import annotations

import json
import uuid
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage

from streamlit_python_only.events import run_diagnostic
from streamlit_python_only.graph_state import AgentState

MAX_REPAIR_ATTEMPTS = 3


class RuntimeGraphOrchestrationMixin:
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
        updated = self._verify_state_for_loop(dict(state))
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
            violations = self._collect_contract_violations(updated, updated.get("final_text", ""))
            if violations and not updated.get("final_repair_attempted") and int(updated.get("repair_attempts", 0) or 0) < MAX_REPAIR_ATTEMPTS:
                route = "repair"
            else:
                route = "finish"
        updated["graph_route"] = route
        self._sync_graph_state_slices(updated)
        return updated

    def _repair_node(self, state: AgentState) -> AgentState:
        violations = self._collect_contract_violations(state, state.get("final_text", ""))
        if not violations:
            updated = dict(state)
            updated["graph_route"] = "finish"
            self._sync_graph_state_slices(updated)
            return updated
        repaired_state, repair_events = self._invoke_final_repair(state, violations)
        repaired_state["tool_events"] = list(repaired_state.get("tool_events", [])) + repair_events
        repaired_state["verify_action"] = "end"
        repaired_state["graph_route"] = "verify"
        self._sync_graph_state_slices(repaired_state)
        return repaired_state

    def _finish_node(self, state: AgentState) -> AgentState:
        updated = self._finalize_result_contract(dict(state))
        updated["graph_route"] = "finish"
        self._sync_graph_state_slices(updated)
        return updated
