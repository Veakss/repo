from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx
from langchain_openai import ChatOpenAI

from streamlit_python_only.settings import EnterpriseLlmModel, ProviderProfile, env_value, get_settings, load_enterprise_config, load_provider_catalog


@dataclass(slots=True)
class ProviderCapabilities:
    supports_native_tools: bool
    supports_textual_replay: bool
    supports_multi_tool_turn: bool
    max_tool_calls_per_turn: int
    requires_sequential_tool_loop: bool
    config_source: str
    provider_family: str
    config_path: str | None = None

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "supportsNativeTools": self.supports_native_tools,
            "supportsTextualReplay": self.supports_textual_replay,
            "supportsMultiToolTurn": self.supports_multi_tool_turn,
            "maxToolCallsPerTurn": self.max_tool_calls_per_turn,
            "requiresSequentialToolLoop": self.requires_sequential_tool_loop,
            "configSource": self.config_source,
            "providerFamily": self.provider_family,
        }
        if self.config_path:
            payload["configPath"] = self.config_path
        return payload


@dataclass(slots=True)
class ResolvedProvider:
    name: str
    provider: str
    label: str
    base_url: str
    api_key: str
    model: str
    mode: str
    capabilities: ProviderCapabilities

    @property
    def is_thales(self) -> bool:
        return self.provider == "thales" or "corp.thales" in self.base_url.lower()


def is_thales_url(base_url: str) -> bool:
    return "corp.thales" in base_url.lower()


def normalize_provider_name(provider: str) -> str:
    normalized = str(provider or "").strip().lower()
    if normalized in {"mistralai", "thales", "thalesapi"}:
        return "thales"
    return normalized or "openrouter"


def infer_mode(profile: ProviderProfile | None, base_url: str, provider: str, explicit_mode: str | None) -> str:
    mode = (explicit_mode or profile.mode if profile else explicit_mode or "auto").strip().lower()
    if mode in {"native", "textual_replay"}:
        return mode
    if provider == "thales" or is_thales_url(base_url):
        return "textual_replay"
    return "native"


def build_provider_capabilities(
    mode: str,
    provider: str,
    base_url: str,
    *,
    config_source: str,
    config_path: str | None = None,
) -> ProviderCapabilities:
    normalized_provider = normalize_provider_name(provider)
    is_thales = normalized_provider == "thales" or is_thales_url(base_url)
    textual_replay = mode == "textual_replay"
    native_tools = mode in {"native", "textual_replay"}
    max_tool_calls_per_turn = 1 if textual_replay or is_thales else 8
    return ProviderCapabilities(
        supports_native_tools=native_tools,
        supports_textual_replay=textual_replay or is_thales,
        supports_multi_tool_turn=max_tool_calls_per_turn > 1,
        max_tool_calls_per_turn=max_tool_calls_per_turn,
        requires_sequential_tool_loop=max_tool_calls_per_turn <= 1,
        config_source=config_source,
        provider_family=normalized_provider,
        config_path=config_path,
    )


def _resolve_enterprise_model(profile_name: str | None, requested_model: str | None) -> ResolvedProvider | None:
    settings = get_settings()
    loaded = load_enterprise_config()
    if loaded is None or loaded.config.llm is None:
        return None

    llm = loaded.config.llm
    models = list(llm.models or [])
    if not models:
        raise ValueError(f"Enterprise config {loaded.path} does not define any llm.models entries.")

    by_name = {model.name: model for model in models}
    chosen_name = profile_name or settings.llm_profile
    chosen_model: EnterpriseLlmModel | None = by_name.get(chosen_name) if chosen_name else None

    default_name = llm.default.name if llm.default and llm.default.name else None
    if chosen_model is None and default_name:
        chosen_model = by_name.get(default_name)
    if chosen_model is None:
        chosen_model = models[0]

    provider = normalize_provider_name(chosen_model.provider)
    base_url = str(chosen_model.base_url or settings.thales_base_url or settings.llm_base_url or "").strip()
    if not base_url:
        raise ValueError(f"Enterprise model '{chosen_model.name}' in {loaded.path} is missing base_url.")

    if provider == "thales":
        api_key = settings.thales_api_key or chosen_model.api_key or settings.llm_api_key
    else:
        api_key = settings.llm_api_key or chosen_model.api_key or settings.thales_api_key

    model = requested_model or chosen_model.model
    mode = infer_mode(None, base_url, provider, settings.llm_provider_mode)
    return ResolvedProvider(
        name=chosen_model.name,
        provider=provider,
        label=chosen_model.name,
        base_url=base_url,
        api_key=api_key or "",
        model=model,
        mode=mode,
        capabilities=build_provider_capabilities(
            mode,
            provider,
            base_url,
            config_source="enterprise_yaml",
            config_path=loaded.path,
        ),
    )


def resolve_provider(profile_name: str | None = None, requested_model: str | None = None) -> ResolvedProvider:
    settings = get_settings()
    enterprise_provider = _resolve_enterprise_model(profile_name=profile_name, requested_model=requested_model)
    if enterprise_provider is not None:
        return enterprise_provider

    catalog = load_provider_catalog()
    chosen_name = profile_name or settings.llm_profile or catalog.default_profile
    profile = catalog.profiles.get(chosen_name) if chosen_name else None

    if profile:
        api_key = env_value(profile.api_key_env) or settings.llm_api_key
        base_url = profile.base_url
        model = requested_model or profile.model
        provider = normalize_provider_name(profile.provider)
        label = profile.label or chosen_name or profile.model
        mode = infer_mode(profile, base_url, provider, settings.llm_provider_mode)
        return ResolvedProvider(
            name=chosen_name or "catalog-default",
            provider=provider,
            label=label,
            base_url=base_url,
            api_key=api_key or "",
            model=model,
            mode=mode,
            capabilities=build_provider_capabilities(mode, provider, base_url, config_source="provider_catalog", config_path=str(settings.resolved_provider_catalog_path)),
        )

    base_url = settings.llm_base_url
    provider = "thales" if is_thales_url(base_url) else normalize_provider_name("openrouter")
    label = "env-default"
    mode = infer_mode(None, base_url, provider, settings.llm_provider_mode)
    return ResolvedProvider(
        name=chosen_name or "env-default",
        provider=provider,
        label=label,
        base_url=base_url,
        api_key=settings.llm_api_key,
        model=requested_model or settings.llm_model,
        mode=mode,
        capabilities=build_provider_capabilities(mode, provider, base_url, config_source="env"),
    )


def build_chat_model(profile_name: str | None = None, requested_model: str | None = None) -> ChatOpenAI:
    provider = resolve_provider(profile_name=profile_name, requested_model=requested_model)
    extra_headers: dict[str, str] = {}
    if provider.provider == "openrouter":
        extra_headers["HTTP-Referer"] = "https://local.streamlit-python-only"
        extra_headers["X-Title"] = "Streamlit Python Only"
    return ChatOpenAI(
        base_url=provider.base_url,
        api_key=provider.api_key or "missing-api-key",
        model=provider.model,
        temperature=0,
        default_headers=extra_headers or None,
    )


def simplify_schema_for_thales(schema: Any) -> dict[str, Any]:
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}}

    properties = {}
    source_properties = schema.get("properties", {})
    if not isinstance(source_properties, dict):
        source_properties = {}

    for key, value in source_properties.items():
        if not isinstance(value, dict):
            properties[key] = {"type": "string"}
            continue
        value_type = value.get("type", "string")
        description = value.get("description")
        if value_type == "object":
            properties[f"{key}_json"] = {
                "type": "string",
                "description": f"{description or key} as compact JSON object string.",
            }
            continue
        if value_type == "array":
            properties[f"{key}_json"] = {
                "type": "string",
                "description": f"{description or key} as compact JSON array string.",
            }
            continue
        item = {"type": value_type}
        if description:
            item["description"] = description
        if "enum" in value and isinstance(value["enum"], list):
            item["enum"] = value["enum"]
        properties[key] = item

    required = []
    for name in schema.get("required", []) if isinstance(schema.get("required"), list) else []:
        if name in properties:
            required.append(name)
        elif f"{name}_json" in properties:
            required.append(f"{name}_json")

    output = {"type": "object", "properties": properties}
    if required:
        output["required"] = required
    return output


def list_available_models(provider: ResolvedProvider | None = None) -> dict[str, Any]:
    settings = get_settings()
    provider = provider or resolve_provider()
    models: list[dict[str, Any]] = []

    enterprise_config = load_enterprise_config()
    if enterprise_config and enterprise_config.config.llm and enterprise_config.config.llm.models:
        models = [
            {"id": profile.model, "profile": profile.name, "provider": normalize_provider_name(profile.provider)}
            for profile in enterprise_config.config.llm.models
        ]
    else:
        catalog = load_provider_catalog()
        profiles = catalog.profiles
        models = [{"id": profile.model, "profile": name, "provider": normalize_provider_name(profile.provider)} for name, profile in profiles.items()]
        if not models:
            models.append({"id": settings.llm_model, "profile": settings.llm_profile or "env-default", "provider": provider.provider})

    if provider.capabilities.config_source != "enterprise_yaml":
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(
                    f"{provider.base_url.rstrip('/')}/models",
                    headers={"Authorization": f"Bearer {provider.api_key}"} if provider.api_key else {},
                )
                response.raise_for_status()
                payload = response.json()
                remote_models = payload.get("data", payload.get("models", []))
                remote_ids = []
                for item in remote_models:
                    if isinstance(item, dict) and item.get("id"):
                        remote_ids.append({"id": str(item["id"]), "profile": None, "provider": provider.provider})
                if remote_ids:
                    models = remote_ids
        except Exception:
            pass

    return {
        "models": models,
        "defaultModel": provider.model,
        "providerMode": provider.mode,
        "providerCapabilities": provider.capabilities.to_payload(),
    }
