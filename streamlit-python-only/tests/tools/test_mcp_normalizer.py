from streamlit_python_only.tools.mcp.normalizer import normalize_mcp_result


def test_normalize_mcp_result_dict_payload() -> None:
    result = normalize_mcp_result({"status": "ok", "payload": {"value": 1}, "citations": ["a"]}, evidence_kind="web")
    assert result["status"] == "ok"
    assert result["evidence_kind"] == "web"
    assert result["payload"] == {"value": 1}
    assert result["citations"] == ["a"]


def test_normalize_mcp_result_raw_value() -> None:
    result = normalize_mcp_result("raw", evidence_kind="file")
    assert result["status"] == "ok"
    assert result["evidence_kind"] == "file"
    assert result["payload"] == "raw"
