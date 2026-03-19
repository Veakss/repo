from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ProviderProfile(BaseModel):
    provider: str
    label: str | None = None
    base_url: str
    api_key_env: str | None = None
    model: str
    mode: str = "auto"


class ProviderCatalog(BaseModel):
    default_profile: str | None = None
    profiles: dict[str, ProviderProfile] = Field(default_factory=dict)


class EnterpriseLlmDefault(BaseModel):
    provider: str | None = None
    name: str | None = None


class EnterpriseLlmModel(BaseModel):
    name: str
    provider: str
    type: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    model: str
    temperature: float | None = None
    max_tokens: int | None = None
    logprobs: bool | None = None
    max_retries: int | None = None
    timeout: float | None = None


class EnterpriseLlmSection(BaseModel):
    default: EnterpriseLlmDefault | None = None
    models: list[EnterpriseLlmModel] = Field(default_factory=list)


class EnterpriseConfig(BaseModel):
    llm: EnterpriseLlmSection | None = None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    backend_port: int = 8010
    sidecar_port: int = 4001
    streamlit_port: int = 8501

    orchestrator_sidecar_url: str = "http://127.0.0.1:4001"
    backend_base_url: str = "http://127.0.0.1:8010"

    mongodb_uri: str = "mongodb://127.0.0.1:27017"
    mongodb_database: str = "streamlit_python_only"

    llm_provider_mode: str = "auto"
    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_api_key: str = ""
    llm_model: str = "google/gemini-2.5-flash-lite-preview-09-2025"
    llm_profile: str | None = None

    thales_base_url: str | None = None
    thales_api_key: str | None = None
    thales_model: str | None = None
    thales_profile: str | None = None

    enterprise_config_path: str | None = None
    provider_catalog_path: str = "config/providers.yaml"
    workspace_root: str = ""
    artifact_root: str | None = None

    enable_file_tools: bool = True
    enable_terminal_tools: bool = True
    enable_web_search: bool = True
    enable_rag: bool = True
    enable_app_actions: bool = True
    enable_tool_clarification: bool = True
    default_language: str = "auto"

    enable_searxng_web_search: bool = True
    searxng_base_url: str = ""
    searxng_timeout_ms: int = 8000

    mcp_mode: str = "disabled"
    mcp_gateway_url: str = "http://127.0.0.1:4010"
    mcp_enable_files: bool = True
    mcp_enable_web: bool = True
    mcp_enable_terminal: bool = False
    mcp_enable_resources: bool = True
    mcp_timeout_ms: int = 10000
    mcp_retry_count: int = 1

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parents[2]

    @property
    def resolved_workspace_root(self) -> Path:
        return Path(self.workspace_root or self.project_root).resolve()

    @property
    def resolved_artifact_root(self) -> Path:
        root = Path(self.artifact_root) if self.artifact_root else self.project_root.joinpath("artifacts")
        return root.resolve()

    @property
    def resolved_provider_catalog_path(self) -> Path:
        return (self.project_root / self.provider_catalog_path).resolve()

    @property
    def resolved_enterprise_config_override_path(self) -> Path | None:
        candidate = (
            self.enterprise_config_path
            or os.getenv("ENTERPRISE_CONFIG_YAML")
            or os.getenv("SEAMLESS_CONFIG_YAML")
            or ""
        ).strip()
        if not candidate:
            return None
        path = Path(candidate)
        if not path.is_absolute():
            path = self.project_root / path
        return path.resolve()

    @property
    def candidate_enterprise_config_paths(self) -> list[Path]:
        candidates: list[Path] = []
        override = self.resolved_enterprise_config_override_path
        if override is not None:
            candidates.append(override)
        for relative in ("tools/config.yaml", "config.yaml"):
            path = (self.project_root / relative).resolve()
            if path not in candidates:
                candidates.append(path)
        return candidates


class LoadedEnterpriseConfig(BaseModel):
    path: str
    config: EnterpriseConfig


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


@lru_cache(maxsize=1)
def load_provider_catalog() -> ProviderCatalog:
    settings = get_settings()
    path = settings.resolved_provider_catalog_path
    if not path.exists():
        return ProviderCatalog()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return ProviderCatalog.model_validate(raw)


@lru_cache(maxsize=1)
def load_enterprise_config() -> LoadedEnterpriseConfig | None:
    settings = get_settings()
    explicit_override = settings.resolved_enterprise_config_override_path
    for path in settings.candidate_enterprise_config_paths:
        if not path.exists():
            continue
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return LoadedEnterpriseConfig(path=str(path), config=EnterpriseConfig.model_validate(raw))
    if explicit_override is not None:
        raise FileNotFoundError(f"Enterprise config file not found: {explicit_override}")
    return None


def env_value(name: str | None) -> str | None:
    if not name:
        return None
    value = os.getenv(name)
    return value if value else None


def ensure_runtime_dirs() -> None:
    settings = get_settings()
    settings.resolved_artifact_root.mkdir(parents=True, exist_ok=True)
    settings.resolved_artifact_root.joinpath("runs").mkdir(parents=True, exist_ok=True)
    settings.resolved_artifact_root.joinpath("matrix").mkdir(parents=True, exist_ok=True)
    settings.resolved_workspace_root.mkdir(parents=True, exist_ok=True)
