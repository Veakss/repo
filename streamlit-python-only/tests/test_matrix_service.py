from __future__ import annotations

import json
from pathlib import Path

import httpx
import mongomock

from streamlit_python_only.matrix import MatrixService, _aggregate_results, _grade_scenario, _summarize_events
from streamlit_python_only.matrix_catalog import build_matrix_scenarios
from streamlit_python_only.store import MongoStore


def encode_sse(events: list[dict]) -> str:
    return "".join(f"data: {json.dumps(event, ensure_ascii=False)}\n\n" for event in events)


def build_service(tmp_path: Path) -> MatrixService:
    store = MongoStore(
        client=mongomock.MongoClient(),
        database_name="streamlit_python_only_matrix_test",
        artifacts_root=tmp_path,
    )
    session_messages: dict[str, list[dict[str, str]]] = {}
    approval_sessions: dict[str, str] = {}
    clarification_sessions: dict[str, str] = {}
    rag_jobs: dict[str, dict[str, str]] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/sessions" and request.method == "POST":
            session_id = f"matrix-session-{len(session_messages) + 1}"
            session_messages[session_id] = []
            return httpx.Response(200, json={"session_id": session_id})
        if request.url.path.startswith("/v1/sessions/") and request.url.path.endswith("/messages") and request.method == "GET":
            session_id = request.url.path.split("/")[3]
            return httpx.Response(200, json={"messages": session_messages.get(session_id, [])})
        if request.url.path.startswith("/v1/rag/session/") and request.url.path.endswith("/files/import") and request.method == "POST":
            return httpx.Response(200, json={"ok": True})
        if request.url.path.startswith("/v1/rag/session/") and request.url.path.endswith("/index/jobs") and request.method == "POST":
            job_id = "rag-job-1"
            rag_jobs[job_id] = {"job_id": job_id, "status": "done"}
            return httpx.Response(200, json={"job": rag_jobs[job_id]})
        if request.url.path.startswith("/v1/rag/index/jobs/") and request.method == "GET":
            job_id = request.url.path.rsplit("/", 1)[-1]
            return httpx.Response(200, json={"job": rag_jobs[job_id]})
        if request.url.path == "/v1/chat/stream":
            payload = json.loads(request.content.decode("utf-8"))
            prompt = payload.get("message")
            if prompt is None:
                messages = payload.get("messages") or []
                prompt = messages[-1]["content"] if messages else ""
            session_id = payload.get("session_id") or payload.get("sessionId")
            session_messages.setdefault(session_id, []).append({"role": "user", "content": prompt})
            if "food application" in prompt:
                clarification_sessions["clar-1"] = session_id
                session_messages[session_id].append({"role": "assistant", "content": "I need clarification before continuing.\n1. Which platform?\nQuick options:\n- iPhone"})
                return httpx.Response(
                    200,
                    text=encode_sse(
                        [
                            {"type": "run_state", "runId": "run-clar", "state": "awaiting_clarification", "timestamp": "2026-03-11T12:00:00+00:00"},
                            {
                                "type": "tool_call",
                                "runId": "run-clar",
                                "actionId": "call-clar-1",
                                "name": "request_clarification",
                                "arguments": "{\"question\":\"Which platform?\"}",
                                "timestamp": "2026-03-11T12:00:00+00:00",
                            },
                            {
                                "type": "clarification_required",
                                "runId": "run-clar",
                                "clarificationId": "clar-1",
                                "question": "Which platform?",
                                "questions": ["Which platform?"],
                                "options": [{"label": "iPhone"}],
                                "timestamp": "2026-03-11T12:00:01+00:00",
                            },
                            {"type": "done"},
                        ]
                    ),
                    headers={"content-type": "text/event-stream"},
                )
            if "Inspect the workspace to find the likely UI state file" in prompt:
                clarification_sessions["clar-2"] = session_id
                session_messages[session_id].append({"role": "assistant", "content": "I need clarification before continuing.\n1. Which file should I update?\nQuick options:\n- frontend/ui_state.py\n- frontend/app.py"})
                return httpx.Response(
                    200,
                    text=encode_sse(
                        [
                            {"type": "run_state", "runId": "run-explore", "state": "planning", "timestamp": "2026-03-11T12:00:10+00:00"},
                            {
                                "type": "tool_call",
                                "runId": "run-explore",
                                "actionId": "call-list-1",
                                "name": "list_directory",
                                "arguments": "{\"path\":\"frontend\"}",
                                "timestamp": "2026-03-11T12:00:11+00:00",
                            },
                            {
                                "type": "tool_call",
                                "runId": "run-explore",
                                "actionId": "call-clar-2",
                                "name": "request_clarification",
                                "arguments": "{\"question\":\"Which file should I update?\"}",
                                "timestamp": "2026-03-11T12:00:12+00:00",
                            },
                            {"type": "run_state", "runId": "run-explore", "state": "awaiting_clarification", "timestamp": "2026-03-11T12:00:13+00:00"},
                            {
                                "type": "clarification_required",
                                "runId": "run-explore",
                                "clarificationId": "clar-2",
                                "question": "Which file should I update?",
                                "questions": ["Which file should I update?"],
                                "options": [{"label": "frontend/ui_state.py"}, {"label": "frontend/app.py"}],
                                "timestamp": "2026-03-11T12:00:14+00:00",
                            },
                            {"type": "done"},
                        ]
                    ),
                    headers={"content-type": "text/event-stream"},
                )
            if "Create the file artifacts/matrix_sandbox/hello.txt" in prompt:
                approval_sessions["approval-1"] = session_id
                return httpx.Response(
                    200,
                    text=encode_sse(
                        [
                            {"type": "run_state", "runId": "run-file", "state": "awaiting_approval", "timestamp": "2026-03-11T12:01:00+00:00"},
                            {
                                "type": "tool_call",
                                "runId": "run-file",
                                "actionId": "call-1",
                                "name": "write_file",
                                "arguments": "{\"path\":\"artifacts/matrix_sandbox/hello.txt\",\"content\":\"bonjour-matrice\"}",
                                "timestamp": "2026-03-11T12:01:01+00:00",
                            },
                            {
                                "type": "approval_required",
                                "runId": "run-file",
                                "actionId": "approval-1",
                                "name": "write_file",
                                "riskLevel": "risky",
                                "arguments": "{\"path\":\"artifacts/matrix_sandbox/hello.txt\"}",
                                "timestamp": "2026-03-11T12:01:02+00:00",
                            },
                            {"type": "done"},
                        ]
                    ),
                    headers={"content-type": "text/event-stream"},
                )
            if "Read matrix_fixtures/roadmap_status.md" in prompt:
                session_messages[session_id].append({"role": "assistant", "content": "AURORA_PHASE4"})
                return httpx.Response(
                    200,
                    text=encode_sse(
                        [
                            {"type": "run_state", "runId": "run-read", "state": "planning", "timestamp": "2026-03-11T12:02:00+00:00"},
                            {
                                "type": "tool_call",
                                "runId": "run-read",
                                "actionId": "call-2",
                                "name": "read_file",
                                "arguments": "{\"path\":\"matrix_fixtures/roadmap_status.md\"}",
                                "timestamp": "2026-03-11T12:02:01+00:00",
                            },
                            {"type": "token", "token": "AURORA_PHASE4"},
                            {"type": "run_state", "runId": "run-read", "state": "completed", "timestamp": "2026-03-11T12:02:02+00:00"},
                            {"type": "done"},
                        ]
                    ),
                    headers={"content-type": "text/event-stream"},
                )
            if prompt == "Now read matrix_fixtures/project_stack.md and answer in one short sentence with the checkpoint plus the main stack.":
                assistant_text = "AURORA_PHASE4 uses a Python stack with Streamlit and FastAPI."
                session_messages[session_id].append({"role": "assistant", "content": assistant_text})
                return httpx.Response(
                    200,
                    text=encode_sse(
                        [
                            {"type": "run_state", "runId": "run-multistep-2", "state": "planning", "timestamp": "2026-03-11T12:02:02+00:00"},
                            {
                                "type": "tool_call",
                                "runId": "run-multistep-2",
                                "actionId": "call-multistep-2",
                                "name": "read_file",
                                "arguments": "{\"path\":\"matrix_fixtures/project_stack.md\"}",
                                "timestamp": "2026-03-11T12:02:03+00:00",
                            },
                            {"type": "token", "token": assistant_text},
                            {"type": "run_state", "runId": "run-multistep-2", "state": "completed", "timestamp": "2026-03-11T12:02:04+00:00"},
                            {"type": "done"},
                        ]
                    ),
                    headers={"content-type": "text/event-stream"},
                )
            if prompt == "Now read matrix_fixtures/project_stack.md and tell me only the main stack signals.":
                assistant_text = "Python, Streamlit, FastAPI."
                session_messages[session_id].append({"role": "assistant", "content": assistant_text})
                return httpx.Response(
                    200,
                    text=encode_sse(
                        [
                            {"type": "run_state", "runId": "run-multistep-stack-only", "state": "planning", "timestamp": "2026-03-11T12:02:04+00:00"},
                            {
                                "type": "tool_call",
                                "runId": "run-multistep-stack-only",
                                "actionId": "call-multistep-stack-only",
                                "name": "read_file",
                                "arguments": "{\"path\":\"matrix_fixtures/project_stack.md\"}",
                                "timestamp": "2026-03-11T12:02:05+00:00",
                            },
                            {"type": "token", "token": assistant_text},
                            {"type": "run_state", "runId": "run-multistep-stack-only", "state": "completed", "timestamp": "2026-03-11T12:02:06+00:00"},
                            {"type": "done"},
                        ]
                    ),
                    headers={"content-type": "text/event-stream"},
                )
            if prompt == "Now read matrix_fixtures/product_identity.md and answer in one short sentence with the product name, checkpoint, and stack.":
                assistant_text = "AI Technical Assistant is at AURORA_PHASE4 and uses Python with Streamlit and FastAPI."
                session_messages[session_id].append({"role": "assistant", "content": assistant_text})
                return httpx.Response(
                    200,
                    text=encode_sse(
                        [
                            {"type": "run_state", "runId": "run-multistep-3", "state": "planning", "timestamp": "2026-03-11T12:02:07+00:00"},
                            {
                                "type": "tool_call",
                                "runId": "run-multistep-3",
                                "actionId": "call-multistep-3",
                                "name": "read_file",
                                "arguments": "{\"path\":\"matrix_fixtures/product_identity.md\"}",
                                "timestamp": "2026-03-11T12:02:08+00:00",
                            },
                            {"type": "token", "token": assistant_text},
                            {"type": "run_state", "runId": "run-multistep-3", "state": "completed", "timestamp": "2026-03-11T12:02:09+00:00"},
                            {"type": "done"},
                        ]
                    ),
                    headers={"content-type": "text/event-stream"},
                )
            if prompt == "Now read matrix_fixtures/product_identity.md and tell me only the product name.":
                assistant_text = "AI Technical Assistant."
                session_messages[session_id].append({"role": "assistant", "content": assistant_text})
                return httpx.Response(
                    200,
                    text=encode_sse(
                        [
                            {"type": "run_state", "runId": "run-multistep-product-only", "state": "planning", "timestamp": "2026-03-11T12:02:10+00:00"},
                            {
                                "type": "tool_call",
                                "runId": "run-multistep-product-only",
                                "actionId": "call-multistep-product-only",
                                "name": "read_file",
                                "arguments": "{\"path\":\"matrix_fixtures/product_identity.md\"}",
                                "timestamp": "2026-03-11T12:02:11+00:00",
                            },
                            {"type": "token", "token": assistant_text},
                            {"type": "run_state", "runId": "run-multistep-product-only", "state": "completed", "timestamp": "2026-03-11T12:02:12+00:00"},
                            {"type": "done"},
                        ]
                    ),
                    headers={"content-type": "text/event-stream"},
                )
            if prompt == "Now read matrix_fixtures/operational_notes.md and tell me only the preferred editor.":
                assistant_text = "VS Code."
                session_messages[session_id].append({"role": "assistant", "content": assistant_text})
                return httpx.Response(
                    200,
                    text=encode_sse(
                        [
                            {"type": "run_state", "runId": "run-multistep-editor-only", "state": "planning", "timestamp": "2026-03-11T12:02:13+00:00"},
                            {
                                "type": "tool_call",
                                "runId": "run-multistep-editor-only",
                                "actionId": "call-multistep-editor-only",
                                "name": "read_file",
                                "arguments": "{\"path\":\"matrix_fixtures/operational_notes.md\"}",
                                "timestamp": "2026-03-11T12:02:14+00:00",
                            },
                            {"type": "token", "token": assistant_text},
                            {"type": "run_state", "runId": "run-multistep-editor-only", "state": "completed", "timestamp": "2026-03-11T12:02:15+00:00"},
                            {"type": "done"},
                        ]
                    ),
                    headers={"content-type": "text/event-stream"},
                )
            if prompt == "Now inspect the matrix_fixtures directory and answer in one short sentence with the product name, checkpoint, stack, and one extra fixture file that exists there.":
                assistant_text = "AI Technical Assistant is at AURORA_PHASE4, uses Python with Streamlit and FastAPI, and matrix_fixtures includes operational_notes.md."
                session_messages[session_id].append({"role": "assistant", "content": assistant_text})
                return httpx.Response(
                    200,
                    text=encode_sse(
                        [
                            {"type": "run_state", "runId": "run-multistep-list-4", "state": "planning", "timestamp": "2026-03-11T12:02:16+00:00"},
                            {
                                "type": "tool_call",
                                "runId": "run-multistep-list-4",
                                "actionId": "call-multistep-list-4",
                                "name": "list_directory",
                                "arguments": "{\"path\":\"matrix_fixtures\"}",
                                "timestamp": "2026-03-11T12:02:17+00:00",
                            },
                            {"type": "token", "token": assistant_text},
                            {"type": "run_state", "runId": "run-multistep-list-4", "state": "completed", "timestamp": "2026-03-11T12:02:18+00:00"},
                            {"type": "done"},
                        ]
                    ),
                    headers={"content-type": "text/event-stream"},
                )
            if prompt == "Now inspect the matrix_fixtures directory and answer in one short sentence with the product name, checkpoint, stack, preferred editor, and one extra fixture file that exists there.":
                assistant_text = "AI Technical Assistant is at AURORA_PHASE4, uses Python with Streamlit and FastAPI, prefers VS Code, and matrix_fixtures includes operational_notes.md."
                session_messages[session_id].append({"role": "assistant", "content": assistant_text})
                return httpx.Response(
                    200,
                    text=encode_sse(
                        [
                            {"type": "run_state", "runId": "run-multistep-list-5", "state": "planning", "timestamp": "2026-03-11T12:02:19+00:00"},
                            {
                                "type": "tool_call",
                                "runId": "run-multistep-list-5",
                                "actionId": "call-multistep-list-5",
                                "name": "list_directory",
                                "arguments": "{\"path\":\"matrix_fixtures\"}",
                                "timestamp": "2026-03-11T12:02:20+00:00",
                            },
                            {"type": "token", "token": assistant_text},
                            {"type": "run_state", "runId": "run-multistep-list-5", "state": "completed", "timestamp": "2026-03-11T12:02:21+00:00"},
                            {"type": "done"},
                        ]
                    ),
                    headers={"content-type": "text/event-stream"},
                )
            if prompt == "Now remind me of that checkpoint in one short line.":
                session_messages[session_id].append({"role": "assistant", "content": "AURORA_PHASE4"})
                return httpx.Response(
                    200,
                    text=encode_sse(
                        [
                            {"type": "run_state", "runId": "run-read-recall", "state": "completed", "timestamp": "2026-03-11T12:02:03+00:00"},
                            {"type": "token", "token": "AURORA_PHASE4"},
                            {"type": "done"},
                        ]
                    ),
                    headers={"content-type": "text/event-stream"},
                )
            if prompt == "Repeat that checkpoint only, in a compact line.":
                session_messages[session_id].append({"role": "assistant", "content": "AURORA_PHASE4"})
                return httpx.Response(
                    200,
                    text=encode_sse(
                        [
                            {"type": "run_state", "runId": "run-read-recall-compact", "state": "completed", "timestamp": "2026-03-11T12:02:04+00:00"},
                            {"type": "token", "token": "AURORA_PHASE4"},
                            {"type": "done"},
                        ]
                    ),
                    headers={"content-type": "text/event-stream"},
                )
            if "latest OpenAI news with two source links" in prompt:
                session_messages[session_id].append({"role": "assistant", "content": "OpenAI shipped a platform update. Sources: https://openai.com/index/one https://openai.com/index/two"})
                return httpx.Response(
                    200,
                    text=encode_sse(
                        [
                            {"type": "run_state", "runId": "run-web", "state": "planning", "timestamp": "2026-03-11T12:02:05+00:00"},
                            {
                                "type": "tool_call",
                                "runId": "run-web",
                                "actionId": "call-web-1",
                                "name": "web_search",
                                "arguments": "{\"query\":\"latest OpenAI news\"}",
                                "timestamp": "2026-03-11T12:02:06+00:00",
                            },
                            {"type": "token", "token": "OpenAI shipped a platform update. Sources: https://openai.com/index/one https://openai.com/index/two"},
                            {"type": "run_state", "runId": "run-web", "state": "completed", "timestamp": "2026-03-11T12:02:07+00:00"},
                            {"type": "done"},
                        ]
                    ),
                    headers={"content-type": "text/event-stream"},
                )
            if "one of those updates is about" in prompt:
                session_messages[session_id].append({"role": "assistant", "content": "One update is about platform tooling. Sources: https://openai.com/index/one https://openai.com/index/two"})
                return httpx.Response(
                    200,
                    text=encode_sse(
                        [
                            {"type": "run_state", "runId": "run-web-followup", "state": "planning", "timestamp": "2026-03-11T12:02:08+00:00"},
                            {
                                "type": "tool_call",
                                "runId": "run-web-followup",
                                "actionId": "call-web-2",
                                "name": "web_search",
                                "arguments": "{\"query\":\"latest OpenAI news follow-up\"}",
                                "timestamp": "2026-03-11T12:02:09+00:00",
                            },
                            {"type": "token", "token": "One update is about platform tooling. Sources: https://openai.com/index/one https://openai.com/index/two"},
                            {"type": "run_state", "runId": "run-web-followup", "state": "completed", "timestamp": "2026-03-11T12:02:10+00:00"},
                            {"type": "done"},
                        ]
                    ),
                    headers={"content-type": "text/event-stream"},
                )
            if prompt == "Say exactly MESSAGE_ORDER_OK.":
                session_messages[session_id].append({"role": "assistant", "content": "MESSAGE_ORDER_OK."})
                return httpx.Response(
                    200,
                    text=encode_sse(
                        [
                            {"type": "run_state", "runId": "run-order", "state": "planning", "timestamp": "2026-03-11T11:58:00+00:00"},
                            {"type": "token", "token": "MESSAGE_ORDER_OK."},
                            {"type": "run_state", "runId": "run-order", "state": "completed", "timestamp": "2026-03-11T11:58:01+00:00"},
                            {"type": "done"},
                        ]
                    ),
                    headers={"content-type": "text/event-stream"},
                )
            if "Use the terminal to print the current directory and list the workspace root." in prompt:
                session_messages[session_id].append({"role": "assistant", "content": "Current directory inspected; workspace looks like AI Technical Assistant."})
                return httpx.Response(
                    200,
                    text=encode_sse(
                        [
                            {"type": "run_state", "runId": "run-terminal-multi-1", "state": "planning", "timestamp": "2026-03-11T12:02:20+00:00"},
                            {
                                "type": "tool_call",
                                "runId": "run-terminal-multi-1",
                                "actionId": "call-terminal-multi-1",
                                "name": "run_terminal",
                                "arguments": "{\"command\":\"pwd\"}",
                                "timestamp": "2026-03-11T12:02:21+00:00",
                            },
                            {
                                "type": "tool_call",
                                "runId": "run-terminal-multi-1",
                                "actionId": "call-terminal-multi-2",
                                "name": "run_terminal",
                                "arguments": "{\"command\":\"ls\"}",
                                "timestamp": "2026-03-11T12:02:22+00:00",
                            },
                            {"type": "token", "token": "Current directory inspected; workspace looks like AI Technical Assistant."},
                            {"type": "run_state", "runId": "run-terminal-multi-1", "state": "completed", "timestamp": "2026-03-11T12:02:23+00:00"},
                            {"type": "done"},
                        ]
                    ),
                    headers={"content-type": "text/event-stream"},
                )
            if prompt == "And now summarize it in one line.":
                session_messages[session_id].append({"role": "assistant", "content": "AI Technical Assistant is a Streamlit Python project."})
                return httpx.Response(
                    200,
                    text=encode_sse(
                        [
                            {"type": "run_state", "runId": "run-terminal-multi-2", "state": "completed", "timestamp": "2026-03-11T12:02:24+00:00"},
                            {"type": "token", "token": "AI Technical Assistant is a Streamlit Python project."},
                            {"type": "done"},
                        ]
                    ),
                    headers={"content-type": "text/event-stream"},
                )
            if prompt == "Now give me a very short summary of that workspace.":
                session_messages[session_id].append({"role": "assistant", "content": "AI Technical Assistant is a Streamlit Python workspace."})
                return httpx.Response(
                    200,
                    text=encode_sse(
                        [
                            {"type": "run_state", "runId": "run-terminal-multi-compact", "state": "completed", "timestamp": "2026-03-11T12:02:25+00:00"},
                            {"type": "token", "token": "AI Technical Assistant is a Streamlit Python workspace."},
                            {"type": "done"},
                        ]
                    ),
                    headers={"content-type": "text/event-stream"},
                )
            if prompt == "What stands out from that workspace? Keep it brief.":
                session_messages[session_id].append({"role": "assistant", "content": "The workspace stands out for its backend/, artifacts/, and README.md structure."})
                return httpx.Response(
                    200,
                    text=encode_sse(
                        [
                            {"type": "run_state", "runId": "run-terminal-multi-standout", "state": "completed", "timestamp": "2026-03-11T12:02:26+00:00"},
                            {"type": "token", "token": "The workspace stands out for its backend/, artifacts/, and README.md structure."},
                            {"type": "done"},
                        ]
                    ),
                    headers={"content-type": "text/event-stream"},
                )
            exact_match = None
            for marker in [
                "MATRIX_TERMINAL_OK",
                "MATRIX_VARIANT_OK",
                "MATRIX_ONLY_OK",
                "MATRIX_STRICT_OK",
                "MATRIX_RETURN_OK",
            ]:
                if marker in prompt:
                    exact_match = marker
                    break
            if exact_match:
                session_messages[session_id].append({"role": "assistant", "content": exact_match})
                return httpx.Response(
                    200,
                    text=encode_sse(
                        [
                            {"type": "run_state", "runId": "run-terminal-exact", "state": "planning", "timestamp": "2026-03-11T12:02:10+00:00"},
                            {
                                "type": "tool_call",
                                "runId": "run-terminal-exact",
                                "actionId": "call-terminal-1",
                                "name": "run_terminal",
                                "arguments": json.dumps({"command": f"printf {exact_match}"}),
                                "timestamp": "2026-03-11T12:02:11+00:00",
                            },
                            {"type": "token", "token": exact_match},
                            {"type": "run_state", "runId": "run-terminal-exact", "state": "completed", "timestamp": "2026-03-11T12:02:12+00:00"},
                            {"type": "done"},
                        ]
                    ),
                    headers={"content-type": "text/event-stream"},
                )
            french_language_match = None
            for marker in ["LANGUE_FR_OK", "LANGUE_FR_VARIANT_OK"]:
                if marker in prompt:
                    french_language_match = marker
                    break
            if french_language_match:
                session_messages[session_id].append({"role": "assistant", "content": f"Le résultat affiché est {french_language_match}."})
                return httpx.Response(
                    200,
                    text=encode_sse(
                        [
                            {"type": "run_state", "runId": "run-lang-fr", "state": "planning", "timestamp": "2026-03-11T12:02:16+00:00"},
                            {
                                "type": "tool_call",
                                "runId": "run-lang-fr",
                                "actionId": "call-lang-fr",
                                "name": "run_terminal",
                                "arguments": json.dumps({"command": f"printf {french_language_match}"}),
                                "timestamp": "2026-03-11T12:02:17+00:00",
                            },
                            {"type": "token", "token": f"Le résultat affiché est {french_language_match}."},
                            {"type": "run_state", "runId": "run-lang-fr", "state": "completed", "timestamp": "2026-03-11T12:02:18+00:00"},
                            {"type": "done"},
                        ]
                    ),
                    headers={"content-type": "text/event-stream"},
                )
            english_language_match = None
            for marker in ["LANGUAGE_EN_OK", "LANGUAGE_EN_VARIANT_OK", "LANGUAGE_EN_COMPACT_OK"]:
                if marker in prompt:
                    english_language_match = marker
                    break
            if english_language_match:
                session_messages[session_id].append({"role": "assistant", "content": f"The result is {english_language_match}."})
                return httpx.Response(
                    200,
                    text=encode_sse(
                        [
                            {"type": "run_state", "runId": "run-lang-en", "state": "planning", "timestamp": "2026-03-11T12:02:19+00:00"},
                            {
                                "type": "tool_call",
                                "runId": "run-lang-en",
                                "actionId": "call-lang-en",
                                "name": "run_terminal",
                                "arguments": json.dumps({"command": f"printf {english_language_match}"}),
                                "timestamp": "2026-03-11T12:02:20+00:00",
                            },
                            {"type": "token", "token": f"The result is {english_language_match}."},
                            {"type": "run_state", "runId": "run-lang-en", "state": "completed", "timestamp": "2026-03-11T12:02:21+00:00"},
                            {"type": "done"},
                        ]
                    ),
                    headers={"content-type": "text/event-stream"},
                )
            if "what checkpoint is recorded in the indexed document" in prompt.lower():
                session_messages[session_id].append({"role": "assistant", "content": "AURORA_PHASE4"})
                return httpx.Response(
                    200,
                    text=encode_sse(
                        [
                            {"type": "run_state", "runId": "run-rag-1", "state": "planning", "timestamp": "2026-03-11T12:03:00+00:00"},
                            {
                                "type": "tool_call",
                                "runId": "run-rag-1",
                                "actionId": "call-rag-1",
                                "name": "rag_lookup",
                                "arguments": "{\"question\":\"checkpoint\"}",
                                "timestamp": "2026-03-11T12:03:01+00:00",
                            },
                            {"type": "token", "token": "AURORA_PHASE4"},
                            {"type": "run_state", "runId": "run-rag-1", "state": "completed", "timestamp": "2026-03-11T12:03:02+00:00"},
                            {"type": "done"},
                        ]
                    ),
                    headers={"content-type": "text/event-stream"},
                )
            if "compact citation" in prompt.lower():
                session_messages[session_id].append({"role": "assistant", "content": "Very advanced and close to closure. (matrix_fixtures/roadmap_status.md:3)"})
                return httpx.Response(
                    200,
                    text=encode_sse(
                        [
                            {"type": "run_state", "runId": "run-rag-compact", "state": "completed", "timestamp": "2026-03-11T12:03:05+00:00"},
                            {"type": "token", "token": "Very advanced and close to closure. (matrix_fixtures/roadmap_status.md:3)"},
                            {"type": "done"},
                        ]
                    ),
                    headers={"content-type": "text/event-stream"},
                )
            if "which checkpoint is in the indexed document" in prompt.lower():
                session_messages[session_id].append({"role": "assistant", "content": "AURORA_PHASE4"})
                return httpx.Response(
                    200,
                    text=encode_sse(
                        [
                            {"type": "run_state", "runId": "run-rag-compact-1", "state": "planning", "timestamp": "2026-03-11T12:03:00+00:00"},
                            {
                                "type": "tool_call",
                                "runId": "run-rag-compact-1",
                                "actionId": "call-rag-compact-1",
                                "name": "rag_lookup",
                                "arguments": "{\"question\":\"checkpoint\"}",
                                "timestamp": "2026-03-11T12:03:01+00:00",
                            },
                            {"type": "token", "token": "AURORA_PHASE4"},
                            {"type": "run_state", "runId": "run-rag-compact-1", "state": "completed", "timestamp": "2026-03-11T12:03:02+00:00"},
                            {"type": "done"},
                        ]
                    ),
                    headers={"content-type": "text/event-stream"},
                )
            if "inspect the workspace root and tell me briefly what kind of project this is" in prompt.lower():
                session_messages[session_id].append({"role": "assistant", "content": "This is a Streamlit Python assistant project."})
                return httpx.Response(
                    200,
                    text=encode_sse(
                        [
                            {"type": "run_state", "runId": "run-single-tool-2", "state": "planning", "timestamp": "2026-03-11T12:04:00+00:00"},
                            {
                                "type": "tool_call",
                                "runId": "run-single-tool-2",
                                "actionId": "call-list-2",
                                "name": "list_directory",
                                "arguments": "{\"path\":\".\"}",
                                "timestamp": "2026-03-11T12:04:01+00:00",
                            },
                            {"type": "token", "token": "This is a Streamlit Python assistant project."},
                            {"type": "run_state", "runId": "run-single-tool-2", "state": "completed", "timestamp": "2026-03-11T12:04:02+00:00"},
                            {"type": "done"},
                        ]
                    ),
                    headers={"content-type": "text/event-stream"},
                )
            if "what is the phase 4 status" in prompt.lower():
                session_messages[session_id].append({"role": "assistant", "content": "Very advanced and close to closure."})
                return httpx.Response(
                    200,
                    text=encode_sse(
                        [
                            {"type": "run_state", "runId": "run-rag-2", "state": "completed", "timestamp": "2026-03-11T12:03:04+00:00"},
                            {"type": "token", "token": "Very advanced and close to closure."},
                            {"type": "done"},
                        ]
                    ),
                    headers={"content-type": "text/event-stream"},
                )
            session_messages[session_id].append({"role": "assistant", "content": "Salut"})
            return httpx.Response(
                200,
                text=encode_sse(
                    [
                        {"type": "run_state", "runId": "run-social", "state": "planning", "timestamp": "2026-03-11T11:59:00+00:00"},
                        {"type": "token", "token": "Salut"},
                        {"type": "run_state", "runId": "run-social", "state": "completed", "timestamp": "2026-03-11T11:59:01+00:00"},
                        {"type": "done"},
                    ]
                ),
                headers={"content-type": "text/event-stream"},
            )
        if request.url.path == "/v1/approvals/stream":
            matrix_file = Path(__file__).resolve().parents[1].joinpath("artifacts", "matrix_sandbox", "hello.txt")
            matrix_file.parent.mkdir(parents=True, exist_ok=True)
            matrix_file.write_text("bonjour-matrice", encoding="utf-8")
            payload = json.loads(request.content.decode("utf-8"))
            session_id = approval_sessions[payload["approval_id"]]
            session_messages[session_id].append({"role": "assistant", "content": "bonjour-matrice"})
            return httpx.Response(
                200,
                text=encode_sse(
                    [
                        {"type": "approval_decision", "runId": "run-file", "actionId": "approval-1", "decision": "approved", "timestamp": "2026-03-11T12:01:03+00:00"},
                        {"type": "tool_result", "runId": "run-file", "name": "write_file", "ok": True, "timestamp": "2026-03-11T12:01:04+00:00"},
                        {
                            "type": "tool_call",
                            "runId": "run-file",
                            "actionId": "call-3",
                            "name": "read_file",
                            "arguments": "{\"path\":\"artifacts/matrix_sandbox/hello.txt\"}",
                            "timestamp": "2026-03-11T12:01:05+00:00",
                        },
                        {"type": "token", "token": "bonjour-matrice"},
                        {"type": "run_state", "runId": "run-file", "state": "completed", "timestamp": "2026-03-11T12:01:06+00:00"},
                        {"type": "done"},
                    ]
                ),
                headers={"content-type": "text/event-stream"},
            )
        if request.url.path in {"/v1/clarifications/stream", "/v1/clarifications/respond/stream"}:
            payload = json.loads(request.content.decode("utf-8"))
            session_id = clarification_sessions[payload["clarification_id"]]
            session_messages[session_id].append({"role": "user", "content": payload["answer"]})
            lowered_answer = str(payload["answer"]).lower()
            if "web" in lowered_answer:
                assistant_text = "Web MVP with restaurant search and favorites"
                events = [
                    {"type": "clarification_answered", "runId": "run-clar", "clarificationId": "clar-1", "answer": payload["answer"], "timestamp": "2026-03-11T12:00:02+00:00"},
                    {"type": "token", "token": assistant_text},
                    {"type": "run_state", "runId": "run-clar", "state": "completed", "timestamp": "2026-03-11T12:00:03+00:00"},
                    {"type": "done"},
                ]
            elif lowered_answer.strip() == "a mobile app.":
                assistant_text = "I need clarification before continuing.\n1. Do you want iPhone or Android?\nQuick options:\n- iPhone\n- Android"
                events = [
                    {"type": "clarification_answered", "runId": "run-clar", "clarificationId": "clar-1", "answer": payload["answer"], "timestamp": "2026-03-11T12:00:02+00:00"},
                    {
                        "type": "tool_call",
                        "runId": "run-clar",
                        "actionId": "call-clar-2",
                        "name": "request_clarification",
                        "arguments": "{\"question\":\"Do you want iPhone or Android?\"}",
                        "timestamp": "2026-03-11T12:00:03+00:00",
                    },
                    {
                        "type": "clarification_required",
                        "runId": "run-clar",
                        "clarificationId": "clar-2",
                        "question": "Do you want iPhone or Android?",
                        "questions": ["Do you want iPhone or Android?"],
                        "options": [{"label": "iPhone"}, {"label": "Android"}],
                        "timestamp": "2026-03-11T12:00:04+00:00",
                    },
                    {"type": "run_state", "runId": "run-clar", "state": "awaiting_clarification", "timestamp": "2026-03-11T12:00:05+00:00"},
                    {"type": "done"},
                ]
            else:
                assistant_text = "iPhone MVP with restaurants and favorites"
                events = [
                    {"type": "clarification_answered", "runId": "run-clar", "clarificationId": "clar-1", "answer": payload["answer"], "timestamp": "2026-03-11T12:00:02+00:00"},
                    {"type": "token", "token": assistant_text},
                    {"type": "run_state", "runId": "run-clar", "state": "completed", "timestamp": "2026-03-11T12:00:03+00:00"},
                    {"type": "done"},
                ]
            session_messages[session_id].append({"role": "assistant", "content": assistant_text})
            return httpx.Response(
                200,
                text=encode_sse(events),
                headers={"content-type": "text/event-stream"},
            )
        raise AssertionError(f"Unexpected path {request.method} {request.url.path}")

    transport = httpx.MockTransport(handler)
    return MatrixService(
        store=store,
        backend_base_url="http://backend.test",
        sidecar_base_url="http://sidecar.test",
        backend_transport=transport,
        sidecar_transport=transport,
    )


def test_matrix_service_runs_and_persists_report(tmp_path: Path):
    service = build_service(tmp_path)

    report = service.run_matrix(
        {
            "models": ["test-model"],
            "scenario_ids": [
                "social_no_clarify",
                "clarification_resume_food_app_multiturn",
                "session_file_exploration_before_clarification",
                "files_write_then_readback",
                "read_file_phase_status",
                "terminal_exact_output",
                "web_latest_with_sources",
                "rag_session_docs_multiturn_backend",
            ],
            "profiles": ["baseline_current"],
            "surfaces": ["backend_relay"],
            "repeat": 1,
        }
    )

    assert report["report_id"].startswith("matrix-")
    assert len(report["results"]) == 8
    assert report["aggregate"]["bySurface"]["backend_relay"]["runs"] == 8
    assert any(row["scenarioId"] == "files_write_then_readback" and row["grade"]["overall"] == "pass" for row in report["results"])
    assert any(
        row["scenarioId"] == "session_file_exploration_before_clarification"
        and row["grade"]["dimensions"]["flow"]["status"] == "pass"
        and row["grade"]["dimensions"]["messageOrder"]["status"] == "pass"
        for row in report["results"]
    )
    assert any(
        row["scenarioId"] == "terminal_exact_output"
        and row["grade"]["dimensions"]["finalContract"]["status"] == "pass"
        and row["grade"]["dimensions"]["exactOutput"]["status"] == "pass"
        for row in report["results"]
    )
    assert any(
        row["scenarioId"] == "web_latest_with_sources"
        and row["grade"]["dimensions"]["sourceQuality"]["status"] == "pass"
        for row in report["results"]
    )
    assert any(
        row["scenarioId"] == "rag_session_docs_multiturn_backend"
        and row["grade"]["dimensions"]["ragQuality"]["status"] == "pass"
        for row in report["results"]
    )
    assert service.get_report(report["report_id"]) is not None
    assert tmp_path.joinpath("matrix", f"{report['report_id']}.json").exists()


def test_matrix_service_compare_returns_summary(tmp_path: Path):
    service = build_service(tmp_path)
    report_a = service.run_matrix(
        {
            "models": ["test-model"],
            "scenario_ids": ["social_no_clarify"],
            "profiles": ["baseline_current"],
            "surfaces": ["backend_relay"],
        }
    )
    report_b = service.run_matrix(
        {
            "models": ["test-model"],
            "scenario_ids": ["read_file_phase_status"],
            "profiles": ["baseline_current"],
            "surfaces": ["backend_relay"],
        }
    )

    comparison = service.compare(report_b["report_id"], report_a["report_id"])
    assert comparison["summary"]["currentRuns"] == 1
    assert comparison["summary"]["baselineRuns"] == 1
    assert isinstance(comparison["byModel"], list)


def test_matrix_catalog_contains_live_hardening_scenarios(tmp_path: Path):
    service = build_service(tmp_path)
    scenario_ids = {scenario["id"] for scenario in service.list_scenarios()}
    assert {
        "message_order_basic",
        "web_followup_with_sources",
        "terminal_multiturn_inspect_then_summarize",
        "exact_output_after_tool_variant",
        "reply_in_user_language_after_tool_fr",
        "reply_in_user_language_after_tool_en",
        "single_tool_per_turn_sequential_runtime",
        "rag_followup_grounded_compact",
        "multistep_read_three_files_synthesis_explicit",
        "multistep_mixed_five_step_workspace_explicit",
        "mini_project_marker_parser",
        "mini_project_marker_parser_guided",
        "mini_project_marker_parser_verified",
        "mini_project_marker_parser_roadmap",
        "mini_project_marker_parser_ordered",
        "mini_project_marker_parser_levels_1",
        "mini_project_marker_parser_levels_2",
        "mini_project_marker_parser_levels_3",
        "mini_project_marker_parser_levels_4",
        "mini_project_marker_parser_levels_5",
    }.issubset(scenario_ids)
    by_id = {scenario["id"]: scenario for scenario in service.list_scenarios()}
    assert by_id["terminal_exact_output"]["variantGroup"] == "exact_output_after_tool"
    assert by_id["exact_output_after_tool_variant"]["variantGroup"] == "exact_output_after_tool"
    exact_output_variants = [scenario for scenario in service.list_scenarios() if scenario.get("variantGroup") == "exact_output_after_tool"]
    language_variants = [scenario for scenario in service.list_scenarios() if scenario.get("variantGroup") == "reply_in_user_language_after_tool"]
    provider_variants = [scenario for scenario in service.list_scenarios() if scenario.get("variantGroup") == "provider_sequential_runtime"]
    message_order_variants = [scenario for scenario in service.list_scenarios() if scenario.get("variantGroup") == "message_order_runtime"]
    terminal_variants = [scenario for scenario in service.list_scenarios() if scenario.get("variantGroup") == "terminal_sequential_followup"]
    multistep_variants = [scenario for scenario in service.list_scenarios() if scenario.get("variantGroup") == "multistep_runtime"]
    multistep_explicit_variants = [scenario for scenario in service.list_scenarios() if scenario.get("variantGroup") == "multistep_runtime_explicit"]
    mini_project_variants = [scenario for scenario in service.list_scenarios() if scenario.get("variantGroup") == "mini_project_code_loop"]
    mini_project_guided_variants = [scenario for scenario in service.list_scenarios() if scenario.get("variantGroup") == "mini_project_code_loop_guided"]
    mini_project_verified_variants = [scenario for scenario in service.list_scenarios() if scenario.get("variantGroup") == "mini_project_code_loop_verified"]
    mini_project_roadmap_variants = [scenario for scenario in service.list_scenarios() if scenario.get("variantGroup") == "mini_project_code_loop_roadmap"]
    mini_project_ordered_variants = [scenario for scenario in service.list_scenarios() if scenario.get("variantGroup") == "mini_project_code_loop_ordered"]
    mini_project_levels_variants = [scenario for scenario in service.list_scenarios() if scenario.get("variantGroup") == "mini_project_code_loop_levels"]
    assert len(exact_output_variants) >= 5
    assert len(language_variants) >= 5
    assert len(provider_variants) >= 4
    assert len(message_order_variants) >= 8
    assert len(terminal_variants) >= 4
    assert len(multistep_variants) >= 4
    assert len(multistep_explicit_variants) >= 2
    assert len(mini_project_variants) >= 1
    assert len(mini_project_guided_variants) >= 1
    assert len(mini_project_verified_variants) >= 1
    assert len(mini_project_roadmap_variants) >= 1
    assert len(mini_project_ordered_variants) >= 1
    assert len(mini_project_levels_variants) >= 5
    assert {scenario.get("stepCount") for scenario in multistep_variants} == {2, 3, 4, 5}


def test_matrix_grading_assigns_failure_classes_and_clusters(tmp_path: Path):
    scenarios = {scenario["id"]: scenario for scenario in build_matrix_scenarios()}
    workspace_root = tmp_path

    source_fail = _grade_scenario(
        workspace_root,
        scenarios["web_followup_with_sources"],
        {
            "state": "completed",
            "tools": ["web_search"],
            "diagnostics": [{"code": "sources_missing_in_final_answer", "message": "Missing URLs."}],
            "finalText": "Short answer without links",
            "eventCount": 4,
            "messages": {"tailRoles": ["user", "assistant"]},
            "turns": [
                {"state": "completed", "tools": ["web_search"], "finalText": "Short answer without links", "messages": {"tailRoles": ["user", "assistant"]}},
                {"state": "completed", "tools": [], "finalText": "Short answer without links", "messages": {"tailRoles": ["user", "assistant"]}},
            ],
            "eventSequence": ["tool:web_search", "state:completed"],
        },
    )
    assert source_fail["suspectedFailureClass"] == "source_quality"

    language_fail = _grade_scenario(
        workspace_root,
        scenarios["reply_in_user_language_after_tool_fr"],
        {
            "state": "completed",
            "tools": ["run_terminal"],
            "diagnostics": [{"code": "wrong_response_language", "message": "Expected French."}],
            "finalText": "The result is LANGUAGE_EN_OK.",
            "eventCount": 4,
            "messages": {"tailRoles": ["user", "assistant"]},
            "eventSequence": ["tool:run_terminal", "state:completed"],
        },
    )
    assert language_fail["suspectedFailureClass"] == "runtime_contract"

    sequential_fail = _grade_scenario(
        workspace_root,
        scenarios["single_tool_per_turn_sequential_runtime"],
        {
            "state": "completed",
            "tools": ["read_file", "list_directory"],
            "diagnostics": [{"code": "provider_tool_calls_collapsed", "message": "Collapsed."}],
            "finalText": "Done.",
            "eventCount": 6,
            "messages": {"tailRoles": ["user", "assistant"]},
            "turns": [
                {"state": "completed", "tools": ["read_file"], "finalText": "AURORA_PHASE4", "messages": {"tailRoles": ["user", "assistant"]}},
                {"state": "completed", "tools": ["list_directory"], "finalText": "Done.", "messages": {"tailRoles": ["user", "assistant"]}},
            ],
            "eventSequence": ["tool:read_file", "tool:list_directory", "state:completed"],
        },
    )
    assert sequential_fail["suspectedFailureClass"] == "provider_sequentiality"

    rag_fail = _grade_scenario(
        workspace_root,
        scenarios["rag_followup_grounded_compact"],
        {
            "state": "completed",
            "tools": ["rag_lookup"],
            "diagnostics": [{"code": "rag_grounding_weak", "message": "Weak grounding."}],
            "finalText": "Probably fine.",
            "eventCount": 4,
            "messages": {"tailRoles": ["user", "assistant"]},
            "turns": [
                {"state": "completed", "tools": ["rag_lookup"], "finalText": "AURORA_PHASE4", "messages": {"tailRoles": ["user", "assistant"]}},
                {"state": "completed", "tools": [], "finalText": "Probably fine.", "messages": {"tailRoles": ["user", "assistant"]}},
            ],
            "eventSequence": ["tool:rag_lookup", "state:completed"],
        },
    )
    assert rag_fail["suspectedFailureClass"] == "rag_quality"

    aggregate = _aggregate_results(
        [
            {
                "scenarioId": "terminal_exact_output",
                "variantGroup": "exact_output_after_tool",
                "model": "m1",
                "profile": "baseline_current",
                "surface": "backend_relay",
                "grade": {**language_fail, "overall": "hard_fail", "suspectedFailureClass": "runtime_contract"},
            },
            {
                "scenarioId": "exact_output_after_tool_variant",
                "variantGroup": "exact_output_after_tool",
                "model": "m1",
                "profile": "baseline_current",
                "surface": "direct_runtime",
                "grade": {**language_fail, "overall": "hard_fail", "suspectedFailureClass": "runtime_contract"},
            },
        ],
        repeat=1,
    )
    cluster = aggregate["failureClusters"][0]
    assert cluster["class"] == "runtime_contract"
    assert cluster["sameFailureAcrossVariants"] is True
    assert cluster["surfaceScope"] == "all_surfaces"
    trend_signal = next(item for item in aggregate["trendSignals"] if item["variantGroup"] == "exact_output_after_tool")
    assert trend_signal["signalStrength"] == "weak"
    assert trend_signal["dominantFailureClass"] == "runtime_contract"


def test_matrix_aggregate_exposes_confirmed_trend_signals(tmp_path: Path):
    aggregate = _aggregate_results(
        [
            {
                "scenarioId": f"exact-output-{index}",
                "variantGroup": "exact_output_after_tool",
                "model": "m1",
                "profile": "baseline_current",
                "surface": "backend_relay",
                "grade": {
                    "overall": "hard_fail" if index < 5 else "pass",
                    "score": 10,
                    "maxScore": 16,
                    "suspectedFailureClass": "runtime_contract" if index < 5 else None,
                    "dimensions": {},
                },
                "summary": {"latencyMs": 10},
            }
            for index in range(1, 6)
        ],
        repeat=1,
    )
    trend_signal = next(item for item in aggregate["trendSignals"] if item["variantGroup"] == "exact_output_after_tool")
    assert trend_signal["signalStrength"] == "confirmed"
    assert trend_signal["likelyCause"] == "logic_or_contract_gap"
    assert trend_signal["sameFailureAcrossMajority"] is True


def test_matrix_aggregate_exposes_by_step_count(tmp_path: Path):
    aggregate = _aggregate_results(
        [
            {
                "scenarioId": "multistep-read-two",
                "variantGroup": "multistep_runtime",
                "stepCount": 2,
                "model": "m1",
                "profile": "baseline_current",
                "surface": "backend_relay",
                "grade": {
                    "overall": "pass",
                    "score": 16,
                    "maxScore": 16,
                    "suspectedFailureClass": None,
                    "dimensions": {},
                },
                "summary": {"latencyMs": 10},
            },
            {
                "scenarioId": "multistep-read-five",
                "variantGroup": "multistep_runtime",
                "stepCount": 5,
                "model": "m1",
                "profile": "baseline_current",
                "surface": "backend_relay",
                "grade": {
                    "overall": "hard_fail",
                    "score": 10,
                    "maxScore": 16,
                    "suspectedFailureClass": "final_answer_discipline",
                    "dimensions": {},
                },
                "summary": {"latencyMs": 20},
            },
        ],
        repeat=1,
    )
    assert aggregate["byStepCount"]["2"]["runs"] == 1
    assert aggregate["byStepCount"]["2"]["passRate"] == 1.0
    assert aggregate["byStepCount"]["5"]["runs"] == 1
    assert aggregate["byStepCount"]["5"]["hard_fail"] == 1


def test_matrix_service_filters_by_variant_group(tmp_path: Path):
    service = build_service(tmp_path)
    report = service.run_matrix(
        {
            "models": ["test-model"],
            "variant_groups": ["exact_output_after_tool"],
            "profiles": ["baseline_current"],
            "surfaces": ["backend_relay"],
            "repeat": 1,
        }
    )
    assert report["results"]
    assert all(row["variantGroup"] == "exact_output_after_tool" for row in report["results"])


def test_matrix_service_runs_new_live_hardening_subset(tmp_path: Path):
    service = build_service(tmp_path)
    report = service.run_matrix(
        {
            "models": ["test-model"],
            "scenario_ids": [
                "message_order_basic",
                "web_followup_with_sources",
                "terminal_multiturn_inspect_then_summarize",
                "terminal_multiturn_inspect_then_compact_summary",
                "terminal_multiturn_inspect_then_key_entries",
                "exact_output_after_tool_variant",
                "reply_in_user_language_after_tool_fr",
                "reply_in_user_language_after_tool_en",
                "single_tool_per_turn_sequential_runtime",
                "single_tool_per_turn_project_stack_runtime",
                "single_tool_per_turn_project_signals_runtime",
                "single_tool_per_turn_project_classification_runtime",
                "rag_followup_grounded_compact",
            ],
            "profiles": ["baseline_current"],
            "surfaces": ["backend_relay"],
            "repeat": 1,
        }
    )
    assert len(report["results"]) == 13
    assert any(row["scenarioId"] == "web_followup_with_sources" and row["grade"]["overall"] == "pass" for row in report["results"])
    assert any(row["scenarioId"] == "reply_in_user_language_after_tool_fr" and row["grade"]["dimensions"]["finalContract"]["status"] == "pass" for row in report["results"])
    assert any(row["scenarioId"] == "single_tool_per_turn_project_stack_runtime" for row in report["results"])
    assert any(row["scenarioId"] == "terminal_multiturn_inspect_then_compact_summary" for row in report["results"])
    assert "failureClusters" in report["aggregate"]


def test_matrix_service_runs_multistep_subset_and_reports_step_depth(tmp_path: Path):
    service = build_service(tmp_path)
    report = service.run_matrix(
        {
            "models": ["test-model"],
            "scenario_ids": [
                "multistep_read_two_files_synthesis",
                "multistep_read_three_files_synthesis",
                "multistep_mixed_four_step_workspace",
                "multistep_mixed_five_step_workspace",
            ],
            "profiles": ["baseline_current"],
            "surfaces": ["backend_relay"],
            "repeat": 1,
        }
    )
    assert len(report["results"]) == 4
    assert {row["stepCount"] for row in report["results"]} == {2, 3, 4, 5}
    assert report["aggregate"]["byStepCount"]["2"]["runs"] == 1
    assert report["aggregate"]["byStepCount"]["5"]["runs"] == 1
    assert all(row["grade"]["overall"] == "pass" for row in report["results"])


def test_matrix_service_can_compare_message_order_across_surfaces(tmp_path: Path):
    service = build_service(tmp_path)
    report = service.run_matrix(
        {
            "models": ["test-model"],
            "scenario_ids": [
                "message_order_basic",
                "message_order_basic_variant",
                "message_order_context_recall_multiturn",
                "clarification_resume_needs_second_question_multiturn",
            ],
            "profiles": ["baseline_current"],
            "surfaces": ["backend_relay", "direct_runtime"],
            "repeat": 1,
        }
    )
    assert len(report["results"]) == 8
    assert report["aggregate"]["bySurface"]["backend_relay"]["runs"] == 4
    assert report["aggregate"]["bySurface"]["direct_runtime"]["runs"] == 4
    assert any(row["surface"] == "backend_relay" for row in report["results"])
    assert any(row["surface"] == "direct_runtime" for row in report["results"])


def test_matrix_grading_accepts_escaped_markdown_and_nonfatal_rag_diagnostics(tmp_path: Path):
    spec = next(item for item in build_matrix_scenarios() if item["id"] == "rag_session_docs_multiturn_backend")
    summary = {
        "state": "completed",
        "tools": ["rag_lookup"],
        "diagnostics": [{"code": "rag_citations_missing", "level": "warn", "message": "repair requested earlier"}],
        "eventCount": 4,
        "error": None,
        "finalText": "AURORA\\_PHASE4 very advanced and close to closure. (matrix/roadmap_status.md:3)",
        "messages": {"tailRoles": ["user", "assistant"]},
        "turns": [
            {
                "state": "completed",
                "tools": ["rag_lookup"],
                "diagnostics": [],
                "messages": {"tailRoles": ["user", "assistant"]},
                "finalText": "AURORA\\_PHASE4 (matrix/roadmap_status.md:2)",
            },
            {
                "state": "completed",
                "tools": ["rag_lookup"],
                "diagnostics": [{"code": "rag_citations_missing", "level": "warn", "message": "repair requested earlier"}],
                "messages": {"tailRoles": ["user", "assistant"]},
                "finalText": "very advanced and close to closure. (matrix/roadmap_status.md:3)",
            },
        ],
    }
    grade = _grade_scenario(tmp_path, spec, summary)
    assert grade["dimensions"]["turns"]["status"] == "pass"
    assert grade["dimensions"]["ragQuality"]["status"] == "pass"
    assert grade["dimensions"]["truth"]["status"] == "pass"
    assert grade["overall"] == "pass"


def test_matrix_event_summary_preserves_structured_diagnostic_payloads():
    summary = _summarize_events(
        [
            {
                "type": "run_diagnostic",
                "code": "multistep_progress_snapshot",
                "level": "info",
                "message": "snapshot",
                "data": {"expectedStepCount": 3, "missingInputs": ["file:a.md"]},
            }
        ]
    )
    assert summary["diagnostics"][0]["code"] == "multistep_progress_snapshot"
    assert summary["diagnostics"][0]["data"]["expectedStepCount"] == 3
    assert summary["diagnostics"][0]["data"]["missingInputs"] == ["file:a.md"]


def test_matrix_service_tracks_clarification_resume_message_order_without_backend_message_persistence(tmp_path: Path):
    store = MongoStore(
        client=mongomock.MongoClient(),
        database_name="streamlit_python_only_matrix_test_resume_order",
        artifacts_root=tmp_path,
    )
    session_messages: dict[str, list[dict[str, str]]] = {}
    clarification_sessions: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/sessions" and request.method == "POST":
            session_id = "matrix-session-clarify"
            session_messages[session_id] = []
            return httpx.Response(200, json={"session_id": session_id})
        if request.url.path.startswith("/v1/sessions/") and request.url.path.endswith("/messages") and request.method == "GET":
            session_id = request.url.path.split("/")[3]
            return httpx.Response(200, json={"messages": session_messages.get(session_id, [])})
        if request.url.path == "/v1/chat/stream":
            payload = json.loads(request.content.decode("utf-8"))
            prompt = payload["message"]
            session_id = payload["session_id"]
            session_messages.setdefault(session_id, []).append({"role": "user", "content": prompt})
            session_messages[session_id].append({"role": "assistant", "content": "I need clarification before continuing.\n1. Which platform?\nQuick options:\n- iPhone"})
            clarification_sessions["clar-1"] = session_id
            return httpx.Response(
                200,
                text=encode_sse(
                    [
                        {"type": "run_state", "runId": "run-clar", "state": "awaiting_clarification", "timestamp": "2026-03-11T12:00:00+00:00"},
                        {"type": "tool_call", "runId": "run-clar", "actionId": "call-clar-1", "name": "request_clarification", "arguments": "{\"question\":\"Which platform?\"}", "timestamp": "2026-03-11T12:00:00+00:00"},
                        {"type": "clarification_required", "runId": "run-clar", "clarificationId": "clar-1", "question": "Which platform?", "questions": ["Which platform?"], "options": [{"label": "iPhone"}], "timestamp": "2026-03-11T12:00:01+00:00"},
                        {"type": "done"},
                    ]
                ),
                headers={"content-type": "text/event-stream"},
            )
        if request.url.path == "/v1/clarifications/stream":
            return httpx.Response(
                200,
                text=encode_sse(
                    [
                        {"type": "clarification_answered", "runId": "run-clar", "clarificationId": "clar-1", "answer": "For iPhone", "timestamp": "2026-03-11T12:00:02+00:00"},
                        {"type": "token", "token": "iPhone MVP with restaurants and favorites"},
                        {"type": "run_state", "runId": "run-clar", "state": "completed", "timestamp": "2026-03-11T12:00:03+00:00"},
                        {"type": "done"},
                    ]
                ),
                headers={"content-type": "text/event-stream"},
            )
        raise AssertionError(f"Unexpected path {request.method} {request.url.path}")

    service = MatrixService(
        store=store,
        backend_base_url="http://backend.test",
        sidecar_base_url="http://sidecar.test",
        backend_transport=httpx.MockTransport(handler),
        sidecar_transport=httpx.MockTransport(handler),
    )

    report = service.run_matrix(
        {
            "models": ["test-model"],
            "scenario_ids": ["clarification_resume_food_app_multiturn"],
            "profiles": ["baseline_current"],
            "surfaces": ["backend_relay"],
            "repeat": 1,
        }
    )

    result = report["results"][0]
    assert result["grade"]["dimensions"]["messageOrder"]["status"] == "pass"
    assert result["grade"]["dimensions"]["turns"]["status"] == "pass"
    assert result["summary"]["messages"]["tailRoles"][-2:] == ["user", "assistant"]


def test_matrix_service_accepts_second_clarification_when_followup_answer_stays_broad(tmp_path: Path):
    service = build_service(tmp_path)
    report = service.run_matrix(
        {
            "models": ["test-model"],
            "scenario_ids": ["clarification_resume_needs_second_question_multiturn"],
            "profiles": ["baseline_current"],
            "surfaces": ["backend_relay", "direct_runtime"],
            "repeat": 1,
        }
    )

    assert len(report["results"]) == 2
    for result in report["results"]:
        assert result["grade"]["overall"] == "pass"
        assert result["grade"]["dimensions"]["turns"]["status"] == "pass"
        assert result["grade"]["dimensions"]["messageOrder"]["status"] == "pass"


def test_matrix_service_turn_timeout_becomes_hard_fail_instead_of_blocking_job(tmp_path: Path):
    store = MongoStore(
        client=mongomock.MongoClient(),
        database_name="streamlit_python_only_matrix_test_timeout",
        artifacts_root=tmp_path,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/sessions" and request.method == "POST":
            return httpx.Response(200, json={"session_id": "matrix-timeout-session"})
        if request.url.path.startswith("/v1/sessions/") and request.url.path.endswith("/messages") and request.method == "GET":
            return httpx.Response(200, json={"messages": [{"role": "user", "content": "Help me create a food application."}]})
        if request.url.path == "/v1/chat/stream":
            raise httpx.ReadTimeout("timed out while waiting for the stream to finish", request=request)
        raise AssertionError(f"Unexpected path {request.method} {request.url.path}")

    service = MatrixService(
        store=store,
        backend_base_url="http://backend.test",
        sidecar_base_url="http://sidecar.test",
        backend_transport=httpx.MockTransport(handler),
        sidecar_transport=httpx.MockTransport(handler),
    )
    service.stream_timeout_s = 0.01

    report = service.run_matrix(
        {
            "models": ["test-model"],
            "scenario_ids": ["clarification_food_app"],
            "profiles": ["baseline_current"],
            "surfaces": ["backend_relay"],
            "repeat": 1,
        }
    )

    result = report["results"][0]
    assert result["grade"]["overall"] == "hard_fail"
    assert result["grade"]["dimensions"]["infra"]["status"] == "hard_fail"
    assert "Matrix turn 1 failed" in (result["summary"].get("error") or "")
