from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from continue_better_py.settings import ensure_runtime_dirs, get_settings
from frontend.ui_state import derive_pending_items


TASKS: list[dict[str, Any]] = [
    {
        "id": "terminal_single_shot",
        "prompt": (
            "Use the terminal to inspect the current working directory. "
            "Answer exactly in the form TERMINAL_SINGLE_SHOT_OK:<basename>."
        ),
        "policy_profile": "always_allow",
        "expect_tools": {"run_terminal"},
        "expect_text": "TERMINAL_SINGLE_SHOT_OK:",
    },
    {
        "id": "interactive_terminal",
        "prompt": (
            "Use open_terminal, terminal_write, terminal_wait_for_output, and terminal_close. "
            "Run pwd in the interactive terminal, verify the output, close the terminal, "
            "then answer exactly INTERACTIVE_TERMINAL_OK."
        ),
        "policy_profile": "always_allow",
        "expect_tools": {"open_terminal", "terminal_write", "terminal_wait_for_output", "terminal_close"},
        "expect_text": "INTERACTIVE_TERMINAL_OK",
    },
    {
        "id": "approval_write_file",
        "prompt": (
            "Create the file agent_playground/llm_eval_artifact.txt with the exact content APPROVAL_FILE_OK "
            "and then answer exactly APPROVAL_FILE_OK."
        ),
        "policy_profile": "ask_when_necessary",
        "expect_tools": {"write_file"},
        "expect_text": "APPROVAL_FILE_OK",
    },
]


def iter_sse_events(response: httpx.Response) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    buffer = ""
    for chunk in response.iter_text():
        buffer += chunk
        parts = buffer.split("\n\n")
        buffer = parts.pop() or ""
        for part in parts:
            if not part.startswith("data: "):
                continue
            payload = part[6:].strip()
            if payload:
                events.append(json.loads(payload))
    return events


def stream_with_timeout(base_url: str, path: str, json_payload: dict[str, Any], read_timeout: float = 45.0) -> list[dict[str, Any]]:
    timeout = httpx.Timeout(connect=10.0, read=read_timeout, write=read_timeout, pool=read_timeout)
    with httpx.Client(base_url=base_url, timeout=timeout) as client:
        with client.stream("POST", path, json=json_payload) as response:
            response.raise_for_status()
            return iter_sse_events(response)


def collect_events(base_url: str, payload: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    all_events = stream_with_timeout(base_url, "/v1/chat/stream", payload)
    assistant_text = "".join(event.get("token", "") for event in all_events if event.get("type") == "token").strip()
    while True:
        approvals, clarification = derive_pending_items(all_events)
        if approvals:
            next_events = stream_with_timeout(base_url, "/v1/approvals/stream", {"approval_id": approvals[0]["approval_id"], "decision": "approved"})
            all_events.extend(next_events)
            assistant_text += "".join(event.get("token", "") for event in next_events if event.get("type") == "token").strip()
            continue
        if clarification:
            options = clarification.get("options") or []
            answer = options[0]["label"] if options and isinstance(options[0], dict) and options[0].get("label") else "Proceed with the most direct safe option."
            next_events = stream_with_timeout(base_url, "/v1/clarifications/stream", {"clarification_id": clarification["clarification_id"], "answer": answer})
            all_events.extend(next_events)
            assistant_text += "".join(event.get("token", "") for event in next_events if event.get("type") == "token").strip()
            continue
        break
    return all_events, assistant_text.strip()


def evaluate_single_task(task_id: str) -> dict[str, Any]:
    ensure_runtime_dirs()
    settings = get_settings()
    task = next(item for item in TASKS if item["id"] == task_id)
    workspace_root = str(settings.project_root.parent)
    with httpx.Client(base_url=settings.backend_base_url, timeout=20.0) as client:
        session = client.post("/v1/sessions", json={"title": f"Capability Evaluation {task_id}"})
        session.raise_for_status()
        session_id = session.json()["session_id"]

    payload = {
        "session_id": session_id,
        "message": task["prompt"],
        "workspace_root": workspace_root,
        "model": settings.llm_model,
        "policy_profile": task["policy_profile"],
        "allow_writes": True,
    }
    events, assistant_text = collect_events(settings.backend_base_url, payload)
    called_tools = [event.get("name") for event in events if event.get("type") == "tool_call"]
    approvals = [event for event in events if event.get("type") == "approval_required"]
    terminal_events = [event.get("type") for event in events if str(event.get("type", "")).startswith("terminal_")]
    passed = task["expect_text"] in assistant_text and task["expect_tools"].issubset(set(called_tools))
    file_exists = None
    if task["id"] == "approval_write_file":
        target = Path(workspace_root).joinpath("agent_playground/llm_eval_artifact.txt")
        file_exists = target.exists() and target.read_text(encoding="utf-8") == "APPROVAL_FILE_OK"
        passed = passed and bool(file_exists)
    return {
        "id": task["id"],
        "session_id": session_id,
        "model": settings.llm_model,
        "workspace_root": workspace_root,
        "passed": passed,
        "assistant_text": assistant_text,
        "called_tools": called_tools,
        "approvals": len(approvals),
        "terminal_events": terminal_events,
        "file_check": file_exists,
        "error": None,
    }


def evaluate_all() -> dict[str, Any]:
    ensure_runtime_dirs()
    settings = get_settings()
    results: list[dict[str, Any]] = []
    for task in TASKS:
        command = [sys.executable, str(Path(__file__).resolve()), "--task", task["id"]]
        try:
            completed = subprocess.run(
                command,
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=90,
                check=True,
            )
            result = json.loads(completed.stdout)
        except subprocess.TimeoutExpired:
            result = {"id": task["id"], "passed": False, "error": "task_timeout", "assistant_text": "", "called_tools": [], "approvals": 0, "terminal_events": [], "file_check": None}
        except subprocess.CalledProcessError as exc:
            result = {
                "id": task["id"],
                "passed": False,
                "error": exc.stderr.strip() or exc.stdout.strip() or f"task_failed:{exc.returncode}",
                "assistant_text": "",
                "called_tools": [],
                "approvals": 0,
                "terminal_events": [],
                "file_check": None,
            }
        results.append(result)

    summary = {
        "model": settings.llm_model,
        "passed": sum(1 for row in results if row["passed"]),
        "total": len(results),
        "results": results,
    }
    report_dir = settings.resolved_artifact_root.joinpath("evaluations")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_dir.joinpath("capability_eval.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=[task["id"] for task in TASKS], default=None)
    args = parser.parse_args()
    if args.task:
        print(json.dumps(evaluate_single_task(args.task), ensure_ascii=False))
        return
    print(json.dumps(evaluate_all(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
