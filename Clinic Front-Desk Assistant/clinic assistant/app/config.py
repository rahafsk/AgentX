"""Validated application settings loaded from environment and the local .env file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, Self
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_ENV_FILE = _PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    """Application configuration with safe defaults and validated secrets."""

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        frozen=True,
    )

    app_env: Literal["development", "test", "production"] = "production"
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    log_level: Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"] = "INFO"

    database_url: SecretStr

    llm_mode: Literal["fake", "real"] = "fake"
    llm_provider: Literal["openai", "openai-compatible"] = "openai-compatible"
    llm_model: str | None = None
    llm_api_key: SecretStr | None = None
    llm_base_url: str | None = None
    llm_timeout_seconds: float = Field(default=30.0, gt=0, allow_inf_nan=False)

    clinic_documents_path: Path = Path("data/clinic_documents")

    @field_validator("app_env", mode="before")
    @classmethod
    def normalize_app_env(cls, value: object) -> object:
        """Normalize the environment name while keeping production the safe default."""
        if isinstance(value, str):
            return value.strip().lower() or "production"
        return value

    @field_validator("log_level", mode="before")
    @classmethod
    def normalize_log_level(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().upper()
        return value

    @field_validator("llm_mode", mode="before")
    @classmethod
    def normalize_llm_mode(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().lower() or "fake"
        return value

    @field_validator("llm_provider", mode="before")
    @classmethod
    def normalize_llm_provider(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().lower() or "openai-compatible"
        return value

    @field_validator("llm_model", "llm_api_key", mode="before")
    @classmethod
    def normalize_optional_llm_values(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip() or None
        return value

    @field_validator("llm_base_url", mode="before")
    @classmethod
    def normalize_llm_base_url(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().rstrip("/") or None
        return value

    @field_validator("database_url")
    @classmethod
    def validate_database_url(cls, value: SecretStr) -> SecretStr:
        """Require a PostgreSQL DSN without exposing its credentials in errors."""
        parsed = urlsplit(value.get_secret_value())
        if (
            parsed.scheme not in {"postgres", "postgresql"}
            or not parsed.hostname
            or not parsed.path.strip("/")
        ):
            raise ValueError("DATABASE_URL must be a PostgreSQL URL with a database name.")
        return value

    @field_validator("host")
    @classmethod
    def validate_host(cls, value: str) -> str:
        host = value.strip()
        if not host:
            raise ValueError("HOST cannot be empty.")
        return host

    @field_validator("clinic_documents_path")
    @classmethod
    def resolve_clinic_documents_path(cls, value: Path) -> Path:
        """Resolve relative document paths from the project root, not the cwd."""
        path = value.expanduser()
        if not path.is_absolute():
            path = _PROJECT_ROOT / path
        return path.resolve()

    @model_validator(mode="after")
    def validate_real_llm_settings(self) -> Self:
        """Require a complete, supported provider configuration in real mode."""
        if self.llm_mode == "real":
            if not self.llm_model:
                raise ValueError("LLM_MODEL is required when LLM_MODE=real.")
            if not self.llm_base_url:
                raise ValueError("LLM_BASE_URL is required when LLM_MODE=real.")

            parsed_url = urlsplit(self.llm_base_url)
            if parsed_url.scheme not in {"http", "https"} or not parsed_url.hostname:
                raise ValueError("LLM_BASE_URL must be a valid HTTP or HTTPS URL.")
            if parsed_url.query or parsed_url.fragment:
                raise ValueError("LLM_BASE_URL must not include a query string or fragment.")
            if self.llm_provider == "openai" and not self.llm_api_key:
                raise ValueError("LLM_API_KEY is required when LLM_PROVIDER=openai.")

        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached validated settings instance."""
    return Settings()  # type: ignore[call-arg]  # Values are loaded from env/.env.
