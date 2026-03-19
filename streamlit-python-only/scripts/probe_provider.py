from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from streamlit_python_only.providers import build_chat_model, list_available_models, resolve_provider
from streamlit_python_only.settings import get_settings, load_enterprise_config, load_provider_catalog


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe the configured provider profile.")
    parser.add_argument("--profile", default=None, help="Optional provider profile name.")
    parser.add_argument("--model", default=None, help="Optional model override.")
    parser.add_argument("--config-path", default=None, help="Optional enterprise YAML config path override.")
    parser.add_argument("--skip-chat", action="store_true", help="Only resolve provider metadata and models.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env")
    if args.config_path:
        os.environ["ENTERPRISE_CONFIG_YAML"] = args.config_path
        get_settings.cache_clear()
        load_provider_catalog.cache_clear()
        load_enterprise_config.cache_clear()

    provider = resolve_provider(profile_name=args.profile, requested_model=args.model)
    payload: dict[str, object] = {
        "provider": {
            "name": provider.name,
            "provider": provider.provider,
            "base_url": provider.base_url,
            "model": provider.model,
            "mode": provider.mode,
            "is_thales": provider.is_thales,
            "has_api_key": bool(provider.api_key),
            "capabilities": provider.capabilities.to_payload(),
        },
        "models": list_available_models(),
    }
    if args.skip_chat:
        print(json.dumps(payload, indent=2))
        return 0

    model = build_chat_model(profile_name=args.profile, requested_model=args.model)
    response = model.invoke([HumanMessage(content="Reply with exactly: PROVIDER_OK")])
    text = response.content if isinstance(response.content, str) else str(response.content)
    payload["chat_probe"] = {"text": text}
    print(json.dumps(payload, indent=2))
    return 0 if "PROVIDER_OK" in text else 1


if __name__ == "__main__":
    raise SystemExit(main())
