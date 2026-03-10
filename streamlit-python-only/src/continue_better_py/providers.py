from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx
from langchain_openai import ChatOpenAI

from continue_better_py.settings import ProviderProfile, env_value, get_settings, load_provider_catalog


@dataclass(slots=True)
class ResolvedProvider:
    name: str
    provider: str
    label: str
    base_url: str
    api_key: str
    model: str
    mode: str

    @property
    def is_thales(self) -> bool:
        return self.provider == "thales" or "corp.thales" in self.base_url.lower()


def is_thales_url(base_url: str) -> bool:
    return "corp.thales" in base_url.lower()


def infer_mode(profile: ProviderProfile | None, base_url: str, provider: str, explicit_mode: str | None) -> str:
    mode = (explicit_mode or profile.mode if profile else explicit_mode or "auto").strip().lower()
    if mode in {"native", "textual_replay"}:
        return mode
    if provider == "thales" or is_thales_url(base_url):
        return "textual_replay"
    return "native"


def resolve_provider(profile_name: str | None = None, requested_model: str | None = None) -> ResolvedProvider:
    settings = get_settings()
    catalog = load_provider_catalog()
    chosen_name = profile_name or settings.llm_profile or catalog.default_profile
    profile = catalog.profiles.get(chosen_name) if chosen_name else None

    if profile:
        api_key = env_value(profile.api_key_env) or settings.llm_api_key
        base_url = profile.base_url
        model = requested_model or profile.model
        provider = profile.provider
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
        )

    base_url = settings.llm_base_url
    provider = "thales" if is_thales_url(base_url) else "openrouter"
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
    )


def build_chat_model(profile_name: str | None = None, requested_model: str | None = None) -> ChatOpenAI:
    provider = resolve_provider(profile_name=profile_name, requested_model=requested_model)
    extra_headers: dict[str, str] = {}
    if provider.provider == "openrouter":
        extra_headers["HTTP-Referer"] = "https://local.continue-better"
        extra_headers["X-Title"] = "Continue Better Python"
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


def list_available_models() -> dict[str, Any]:
    settings = get_settings()
    catalog = load_provider_catalog()
    profiles = catalog.profiles
    provider = resolve_provider()
    models = [{"id": profile.model, "profile": name, "provider": profile.provider} for name, profile in profiles.items()]

    if not models:
        models.append({"id": settings.llm_model, "profile": settings.llm_profile or "env-default", "provider": provider.provider})

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

    return {"models": models, "defaultModel": provider.model, "providerMode": provider.mode}
