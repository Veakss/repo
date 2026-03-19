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
SIDECAR_PORT = 14021
BACKEND_PORT = 18030


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


def wait_for_job_done(client: httpx.Client, job_id: str, timeout: float = 90.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        response = client.get(f"/v1/matrix/jobs/{job_id}")
        response.raise_for_status()
        job = response.json()["job"]
        if job["status"] in {"done", "failed"}:
            return job
        time.sleep(1)
    raise SystemExit(f"Timed out waiting for matrix job {job_id}")


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    run_stamp = str(int(time.time()))
    artifact_root = PROJECT_ROOT.joinpath("artifacts", "live_phase5_artifacts")
    artifact_root.mkdir(parents=True, exist_ok=True)
    database_name = f"streamlit_python_only_phase5_live_{run_stamp}"

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

        with httpx.Client(base_url=f"http://127.0.0.1:{BACKEND_PORT}", timeout=120.0) as client:
            catalog = client.get("/v1/matrix/catalog")
            catalog.raise_for_status()
            scenarios = catalog.json()["scenarios"]
            assert any(row["id"] == "social_no_clarify" for row in scenarios)

            job_resp = client.post(
                "/v1/matrix/jobs",
                json={
                    "models": [os.getenv("LLM_MODEL", "google/gemini-2.5-flash-lite-preview-09-2025")],
                    "scenario_ids": ["social_no_clarify", "clarification_food_app"],
                    "profiles": ["baseline_current"],
                    "surfaces": ["backend_relay"],
                    "repeat": 1,
                },
            )
            job_resp.raise_for_status()
            job_id = job_resp.json()["job"]["job_id"]
            job = wait_for_job_done(client, job_id)
            if job["status"] != "done":
                raise SystemExit(f"Matrix job failed: {job}")

            report_id = job["report_id"]
            report_resp = client.get(f"/v1/matrix/reports/{report_id}")
            report_resp.raise_for_status()
            report = report_resp.json()["report"]
            compare = client.get("/v1/matrix/compare", params={"current_report_id": report_id, "baseline_report_id": report_id})
            compare.raise_for_status()

            checks = {
                "report_id": report_id,
                "results_count": len(report["results"]) == 2,
                "artifact_exists": Path(report["artifact_path"]).exists(),
                "reports_listed": any(item["report_id"] == report_id for item in client.get("/v1/matrix/reports").json()["reports"]),
                "compare_runs": compare.json()["comparison"]["summary"]["currentRuns"] == 2,
                "infra_dimension_present": "infra" in report["results"][0]["grade"]["dimensions"] if report["results"] else False,
                "message_order_dimension_present": "messageOrder" in report["results"][0]["grade"]["dimensions"] if report["results"] else False,
            }
            print(json.dumps(checks, indent=2))
            return 0 if all(value is True or isinstance(value, str) for value in checks.values()) else 1
    finally:
        stop_process(backend)
        stop_process(sidecar)
        MongoClient("mongodb://127.0.0.1:27017").drop_database(database_name)


if __name__ == "__main__":
    raise SystemExit(main())
