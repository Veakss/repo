from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv
from pymongo import MongoClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIDECAR_PORT = 14011
BACKEND_PORT = 18020


def wait_for(url: str, timeout: float = 45.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            response = httpx.get(url, timeout=2.0)
            if response.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(0.5)
    raise SystemExit(f"Timed out waiting for {url}")


def parse_sse(text: str) -> list[dict]:
    payloads: list[dict] = []
    for packet in text.split("\n\n"):
        if packet.startswith("data: "):
            payloads.append(json.loads(packet[6:]))
    return payloads


def start_process(module_app: str, port: int, env: dict[str, str]) -> subprocess.Popen[str]:
    python = str(PROJECT_ROOT.joinpath(".venv", "bin", "python"))
    return subprocess.Popen(
        [python, "-m", "uvicorn", module_app, "--host", "127.0.0.1", "--port", str(port)],
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )


def stop_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def wait_for_job_done(client: httpx.Client, job_id: str, timeout: float = 30.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        response = client.get(f"/v1/rag/index/jobs/{job_id}")
        response.raise_for_status()
        job = response.json()["job"]
        if job["status"] in {"done", "failed"}:
            return job
        time.sleep(0.5)
    raise SystemExit(f"Timed out waiting for job {job_id}")


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    run_stamp = str(int(time.time()))
    workspace_root = PROJECT_ROOT.joinpath("artifacts", "live_phase4_workspace")
    artifact_root = PROJECT_ROOT.joinpath("artifacts", "live_phase4_artifacts")
    workspace_root.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)
    source_file = workspace_root.joinpath("phase4_rag_note.md")
    codename = "AURORA_PHASE4"
    source_file.write_text(
        f"# Phase 4\nThe codename for this validation is {codename}.\nUse MongoDB as the canonical store.\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    database_name = f"streamlit_python_only_phase4_live_{run_stamp}"
    env.update(
        {
            "PYTHONPATH": f"{PROJECT_ROOT}:{PROJECT_ROOT / 'src'}:{PROJECT_ROOT / 'frontend'}:{env.get('PYTHONPATH', '')}",
            "ORCHESTRATOR_SIDECAR_URL": f"http://127.0.0.1:{SIDECAR_PORT}",
            "BACKEND_BASE_URL": f"http://127.0.0.1:{BACKEND_PORT}",
            "WORKSPACE_ROOT": str(workspace_root),
            "ARTIFACT_ROOT": str(artifact_root),
            "MONGODB_DATABASE": database_name,
        }
    )

    sidecar = start_process("sidecar.main:app", SIDECAR_PORT, env)
    backend = start_process("backend.main:app", BACKEND_PORT, env)
    try:
        wait_for(f"http://127.0.0.1:{SIDECAR_PORT}/health")
        wait_for(f"http://127.0.0.1:{BACKEND_PORT}/health")

        summary: dict[str, object] = {"checks": []}
        with httpx.Client(base_url=f"http://127.0.0.1:{BACKEND_PORT}", timeout=60.0) as client:
            session = client.post("/v1/sessions", json={"title": "Phase 4 Live"})
            session.raise_for_status()
            session_id = session.json()["session_id"]
            summary["checks"].append(("session_created", True))

            imported = client.post(f"/v1/rag/session/{session_id}/files/import", json={"path": source_file.name})
            imported.raise_for_status()
            summary["checks"].append(("session_doc_imported", imported.json()["imported"] is True))

            job_resp = client.post(f"/v1/rag/session/{session_id}/index/jobs")
            job_resp.raise_for_status()
            job_id = job_resp.json()["job"]["job_id"]
            job = wait_for_job_done(client, job_id)
            summary["checks"].append(("index_job_done", job["status"] == "done"))

            lookup = client.post("/v1/rag/lookup", json={"question": "What is the codename?", "session_id": session_id})
            lookup.raise_for_status()
            lookup_payload = lookup.json()
            summary["checks"].append(("lookup_ok", lookup_payload["status"] == "ok"))
            summary["checks"].append(("lookup_contains_codename", any(codename in hit.get("snippet", "") for hit in lookup_payload.get("hits", []))))

            memory = client.get(f"/v1/rag/session/{session_id}/memory")
            memory.raise_for_status()
            summary["checks"].append(("memory_available", "config" in memory.json()))

            stream = client.post(
                "/v1/chat/stream",
                json={
                    "session_id": session_id,
                    "message": f"Use the rag_lookup tool before answering. What is the codename in the indexed session doc? Reply with only the codename.",
                    "model": os.getenv("LLM_MODEL", "google/gemini-2.5-flash-lite-preview-09-2025"),
                    "allow_writes": False,
                },
            )
            stream.raise_for_status()
            events = parse_sse(stream.text)
            assistant = "".join(event.get("token", "") for event in events if event.get("type") == "token").strip()
            used_rag_tool = any(event.get("type") == "tool_call" and event.get("name") == "rag_lookup" for event in events)
            summary["checks"].append(("live_run_completed", any(event.get("type") == "run_state" and event.get("state") == "completed" for event in events)))
            summary["checks"].append(("live_run_used_rag_or_answered", used_rag_tool or codename.lower() in assistant.lower()))
            summary["assistant"] = assistant
            summary["used_rag_tool"] = used_rag_tool

        print(json.dumps(summary, indent=2))
        return 0 if all(ok for _, ok in summary["checks"]) else 1
    finally:
        stop_process(backend)
        stop_process(sidecar)
        MongoClient("mongodb://127.0.0.1:27017").drop_database(database_name)


if __name__ == "__main__":
    raise SystemExit(main())
