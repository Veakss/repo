from __future__ import annotations

from fastapi.testclient import TestClient

from streamlit_python_only.mcp.gateway.app import create_gateway_app


def test_gateway_health_and_invoke(tmp_path) -> None:
    app = create_gateway_app(str(tmp_path))
    client = TestClient(app)
    health = client.get("/healthz")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    invoke = client.post("/invoke", json={"tool": "list_directory", "arguments": {"path": "."}})
    assert invoke.status_code == 200
    payload = invoke.json()
    assert payload["status"] in {"ok", "error"}
