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


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    backend_port: int = 8010
    sidecar_port: int = 4001
    streamlit_port: int = 8501

    orchestrator_sidecar_url: str = "http://127.0.0.1:4001"
    backend_base_url: str = "http://127.0.0.1:8010"

    mongodb_uri: str = "mongodb://127.0.0.1:27017"
    mongodb_database: str = "continue_better_python"

    llm_provider_mode: str = "auto"
    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_api_key: str = ""
    llm_model: str = "google/gemini-2.5-flash-lite-preview-09-2025"
    llm_profile: str | None = None

    thales_base_url: str | None = None
    thales_api_key: str | None = None
    thales_model: str | None = None
    thales_profile: str | None = None

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

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parents[2]

    @property
    def resolved_workspace_root(self) -> Path:
        return Path(self.workspace_root or self.project_root.parent.joinpath("agent_playground")).resolve()

    @property
    def resolved_artifact_root(self) -> Path:
        root = Path(self.artifact_root) if self.artifact_root else self.project_root.joinpath("artifacts")
        return root.resolve()

    @property
    def resolved_provider_catalog_path(self) -> Path:
        return (self.project_root / self.provider_catalog_path).resolve()


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
