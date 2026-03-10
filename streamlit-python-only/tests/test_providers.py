from continue_better_py.providers import infer_mode, is_thales_url, simplify_schema_for_thales
from continue_better_py.settings import ProviderProfile


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
