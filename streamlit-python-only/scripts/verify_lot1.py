from __future__ import annotations

import json
import os
import sys

import requests
from dotenv import load_dotenv


def main() -> int:
    load_dotenv()
    sidecar_url = os.getenv("SIDECAR_URL", "http://127.0.0.1:4001")
    model = os.getenv("LOT1_MODEL", "google/gemini-2.5-flash-lite-preview-09-2025")

    checks = []
    checks.append(("health", requests.get(f"{sidecar_url}/health", timeout=10).status_code == 200))
    checks.append(("capabilities", requests.get(f"{sidecar_url}/v1/capabilities", timeout=10).status_code == 200))
    checks.append(("models", requests.get(f"{sidecar_url}/v1/models", timeout=20).status_code == 200))

    llm_api_key = os.getenv("LLM_API_KEY", "").strip()
    if not llm_api_key:
        print(json.dumps({"checks": checks, "live_chat": "skipped", "reason": "LLM_API_KEY missing"}, indent=2))
        return 0

    payload = {
        "sessionId": "lot1-live-smoke",
        "messages": [{"role": "user", "content": "Reply with a short hello."}],
        "workspaceRoot": os.getenv("WORKSPACE_ROOT", os.getcwd()),
        "model": model,
    }
    response = requests.post(f"{sidecar_url}/v1/chat/stream", json=payload, stream=True, timeout=120)
    stream_ok = response.status_code == 200
    chunks = []
    for line in response.iter_lines(decode_unicode=True):
        if line:
            chunks.append(line)
    checks.append(("live_chat_http", stream_ok))
    checks.append(("live_chat_completed", any('"type": "done"' in line or '"type":"done"' in line for line in chunks)))
    checks.append(("live_chat_has_token", any('"type": "token"' in line or '"type":"token"' in line for line in chunks)))
    print(json.dumps({"checks": checks, "stream": chunks[:20]}, indent=2))
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
