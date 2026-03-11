from continue_better_py.providers import infer_mode, is_thales_url, simplify_schema_for_thales
from continue_better_py.runtime import parse_text_tool_calls
from continue_better_py.settings import ProviderProfile
from continue_better_py.tooling import build_tool_lookup


def test_is_thales_url_detects_corporate_endpoint():
    assert is_thales_url("https://api.corp.thales/corp/genai-llm-small/v1") is True
    assert is_thales_url("https://openrouter.ai/api/v1") is False


def test_infer_mode_prefers_textual_replay_for_thales():
    profile = ProviderProfile(
        provider="thales",
        label="Thales",
        base_url="https://api.corp.thales/corp/genai-llm-small/v1",
        api_key_env="THALES_API_KEY",
        model="mistral",
        mode="auto",
    )
    assert infer_mode(profile, profile.base_url, profile.provider, "auto") == "textual_replay"


def test_simplify_schema_for_thales_flattens_nested_types():
    schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "filters": {"type": "object", "description": "Filter object"},
            "tags": {"type": "array", "description": "Tag list"},
        },
        "required": ["query", "filters", "tags"],
    }
    simplified = simplify_schema_for_thales(schema)
    assert simplified["required"] == ["query", "filters_json", "tags_json"]
    assert "filters_json" in simplified["properties"]
    assert "tags_json" in simplified["properties"]


def test_write_file_is_marked_risky():
    tools = build_tool_lookup("/tmp")
    assert tools["write_file"].risk_level == "risky"
    assert tools["request_clarification"].module_id == "clarification"


def test_parse_text_tool_calls_supports_json_lines():
    calls = parse_text_tool_calls('{"name":"read_file","arguments":{"path":"README.md"}}')
    assert len(calls) == 1
    assert calls[0]["name"] == "read_file"
