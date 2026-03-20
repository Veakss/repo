from pathlib import Path

import pytest

from streamlit_python_only.providers import infer_mode, is_thales_url, resolve_provider, simplify_schema_for_thales
from streamlit_python_only.runtime import parse_text_tool_calls
from streamlit_python_only.settings import ProviderProfile, get_settings, load_enterprise_config, load_provider_catalog
from streamlit_python_only.tooling import build_tool_lookup


def _reset_provider_caches():
    get_settings.cache_clear()
    load_provider_catalog.cache_clear()
    load_enterprise_config.cache_clear()


@pytest.fixture(autouse=True)
def _provider_cache_guard():
    _reset_provider_caches()
    yield
    _reset_provider_caches()


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


def test_infer_mode_honors_profile_textual_replay_when_explicit_auto():
    profile = ProviderProfile(
        provider="openrouter",
        label="OpenRouter textual replay profile",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="LLM_API_KEY",
        model="google/gemini-2.5-flash-lite-preview-09-2025",
        mode="textual_replay",
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


def test_resolve_provider_prefers_enterprise_yaml_config(tmp_path: Path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text(
        """
llm:
  default:
    provider: mistralai
    name: mistral prod
  models:
    - name: mistral prod
      provider: mistralai
      base_url: https://api.corp.thales/corp/genai-llm-small/v1/
      api_key: YAML_KEY
      model: mistral
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("ENTERPRISE_CONFIG_YAML", str(config))
    monkeypatch.setenv("LLM_API_KEY", "")
    monkeypatch.setenv("THALES_API_KEY", "")
    _reset_provider_caches()
    provider = resolve_provider()
    assert provider.provider == "thales"
    assert provider.api_key == "YAML_KEY"
    assert provider.capabilities.config_source == "enterprise_yaml"
    assert provider.capabilities.provider_family == "thales"
    assert provider.capabilities.max_tool_calls_per_turn == 1
    assert provider.capabilities.supports_multi_tool_turn is False
    assert provider.capabilities.requires_sequential_tool_loop is True
    assert provider.capabilities.config_path == str(config)


def test_enterprise_yaml_env_override_wins_for_api_key(tmp_path: Path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text(
        """
llm:
  models:
    - name: mistral prod
      provider: mistralai
      base_url: https://api.corp.thales/corp/genai-llm-small/v1/
      api_key: YAML_KEY
      model: mistral
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("ENTERPRISE_CONFIG_YAML", str(config))
    monkeypatch.setenv("THALES_API_KEY", "ENV_THALES_KEY")
    _reset_provider_caches()
    provider = resolve_provider()
    assert provider.api_key == "ENV_THALES_KEY"


def test_enterprise_yaml_missing_models_raises(tmp_path: Path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text("llm: {}", encoding="utf-8")
    monkeypatch.setenv("ENTERPRISE_CONFIG_YAML", str(config))
    _reset_provider_caches()
    try:
        resolve_provider()
    except ValueError as exc:
        assert "llm.models" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing llm.models")
