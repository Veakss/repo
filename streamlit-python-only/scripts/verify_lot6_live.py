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
SIDECAR_PORT = 14031
BACKEND_PORT = 18040


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


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    run_stamp = str(int(time.time()))
    artifact_root = PROJECT_ROOT.joinpath("artifacts", "live_phase6_artifacts")
    artifact_root.mkdir(parents=True, exist_ok=True)
    database_name = f"streamlit_python_only_phase6_live_{run_stamp}"

    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": f"{PROJECT_ROOT}:{PROJECT_ROOT / 'src'}:{PROJECT_ROOT / 'frontend'}:{env.get('PYTHONPATH', '')}",
            "ORCHESTRATOR_SIDECAR_URL": f"http://127.0.0.1:{SIDECAR_PORT}",
            "BACKEND_BASE_URL": f"http://127.0.0.1:{BACKEND_PORT}",
            "WORKSPACE_ROOT": str(PROJECT_ROOT),
            "ARTIFACT_ROOT": str(artifact_root),
            "MONGODB_DATABASE": database_name,
        }
    )

    sidecar = start_process("sidecar.main:app", SIDECAR_PORT, env)
    backend = start_process("backend.main:app", BACKEND_PORT, env)
    try:
        wait_for(f"http://127.0.0.1:{SIDECAR_PORT}/health")
        wait_for(f"http://127.0.0.1:{BACKEND_PORT}/health")

        provider_probe = subprocess.run(
            [str(PROJECT_ROOT.joinpath(".venv", "bin", "python")), "scripts/probe_provider.py"],
            cwd=PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        provider_payload = json.loads(provider_probe.stdout)

        with httpx.Client(base_url=f"http://127.0.0.1:{BACKEND_PORT}", timeout=120.0) as client:
            created = client.post("/v1/sessions", json={"title": "Phase 6 Live Session"})
            created.raise_for_status()
            session_id = created.json()["session_id"]

            stream = client.post(
                "/v1/chat/stream",
                json={
                    "session_id": session_id,
                    "message": "Use the run_terminal tool to execute pwd and then reply with the absolute path only.",
                    "model": os.getenv("LLM_MODEL", "google/gemini-2.5-flash-lite-preview-09-2025"),
                    "allow_writes": False,
                },
            )
            stream.raise_for_status()
            events = parse_sse(stream.text)
            assistant = "".join(event.get("token", "") for event in events if event.get("type") == "token").strip()

            checks = {
                "provider_ok": "PROVIDER_OK" in provider_payload.get("chat_probe", {}).get("text", ""),
                "provider_mode_reported": bool(provider_payload.get("provider", {}).get("mode")),
                "terminal_events_or_answer": any(event.get("type") == "terminal_opened" for event in events) or str(PROJECT_ROOT) in assistant,
                "run_completed": any(event.get("type") == "run_state" and event.get("state") == "completed" for event in events),
            }
            print(json.dumps({"checks": checks, "assistant": assistant}, indent=2))
            return 0 if all(checks.values()) else 1
    finally:
        stop_process(backend)
        stop_process(sidecar)
        MongoClient("mongodb://127.0.0.1:27017").drop_database(database_name)


if __name__ == "__main__":
    raise SystemExit(main())
