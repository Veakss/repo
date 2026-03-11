from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv
from pymongo import MongoClient
from streamlit.testing.v1 import AppTest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIDECAR_PORT = 14001
BACKEND_PORT = 18010


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


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    run_stamp = str(int(time.time()))
    workspace_root = PROJECT_ROOT.joinpath("artifacts", "live_phase3_workspace")
    artifact_root = PROJECT_ROOT.joinpath("artifacts", "live_phase3_artifacts")
    workspace_root.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)
    workspace_root.joinpath("phase3_notes.txt").write_text("streamlit phase 3 live smoke", encoding="utf-8")

    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": f"{PROJECT_ROOT}:{PROJECT_ROOT / 'src'}:{env.get('PYTHONPATH', '')}",
            "ORCHESTRATOR_SIDECAR_URL": f"http://127.0.0.1:{SIDECAR_PORT}",
            "BACKEND_BASE_URL": f"http://127.0.0.1:{BACKEND_PORT}",
            "WORKSPACE_ROOT": str(workspace_root),
            "ARTIFACT_ROOT": str(artifact_root),
            "MONGODB_DATABASE": f"continue_better_python_phase3_live_{run_stamp}",
        }
    )
    database_name = env["MONGODB_DATABASE"]

    sidecar = start_process("sidecar.main:app", SIDECAR_PORT, env)
    backend = start_process("backend.main:app", BACKEND_PORT, env)

    try:
        wait_for(f"http://127.0.0.1:{SIDECAR_PORT}/health")
        wait_for(f"http://127.0.0.1:{BACKEND_PORT}/health")

        with httpx.Client(base_url=f"http://127.0.0.1:{BACKEND_PORT}", timeout=30.0) as client:
            created = client.post("/v1/sessions", json={"title": "Phase 3 Live Session"})
            created.raise_for_status()
            session_id = created.json()["session_id"]

            script = f"""
import sys
from pathlib import Path
import streamlit as st

project_root = Path(r"{PROJECT_ROOT}")
for entry in [project_root, project_root / "src", project_root / "frontend"]:
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

st.session_state["backend_url"] = "http://127.0.0.1:{BACKEND_PORT}"
from frontend.app import run_app
run_app()
"""
            app_test = AppTest.from_string(script, default_timeout=60)
            app_test.run(timeout=60)
            app_test.chat_input[0].set_value("Reply with exactly: streamlit live ok").run(timeout=120)

            messages = client.get(f"/v1/sessions/{session_id}/messages", params={"limit": 20})
            messages.raise_for_status()
            body = messages.json()["messages"]
            assert any(message["role"] == "assistant" and "streamlit live ok" in message["content"].lower() for message in body)

            runs = client.get("/v1/runs", params={"session_id": session_id})
            runs.raise_for_status()
            assert len(runs.json()["runs"]) >= 1

            fs_tree = client.get("/v1/fs/tree")
            fs_tree.raise_for_status()
            assert any(node["path"] == "phase3_notes.txt" for node in fs_tree.json()["children"])

            print(
                {
                    "session_id": session_id,
                    "assistant_messages": [message["content"] for message in body if message["role"] == "assistant"],
                    "run_count": len(runs.json()["runs"]),
                }
            )
        return 0
    finally:
        stop_process(backend)
        stop_process(sidecar)
        MongoClient("mongodb://127.0.0.1:27017").drop_database(database_name)


if __name__ == "__main__":
    raise SystemExit(main())
