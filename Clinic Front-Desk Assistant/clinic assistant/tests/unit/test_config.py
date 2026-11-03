"""Unit tests for application configuration validation."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from app.config import Settings, get_settings

_DATABASE_URL = "postgresql://clinic_user:test-password@localhost:5432/clinic_assistant_test"


def test_settings_normalize_environment_and_resolve_document_path() -> None:
    env = {
        "DATABASE_URL": _DATABASE_URL,
        "APP_ENV": " DEVELOPMENT ",
        "HOST": " 127.0.0.1 ",
        "PORT": "8081",
        "LOG_LEVEL": "debug",
        "LLM_MODE": "FAKE",
        "LLM_PROVIDER": "",
        "LLM_API_KEY": "",
        "LLM_BASE_URL": "",
        "LLM_TIMEOUT_SECONDS": "12.5",
        "CLINIC_DOCUMENTS_PATH": "approved_docs",
    }
    with patch.dict(os.environ, env, clear=True):
        settings = Settings()

    project_root = Path(__file__).resolve().parents[2]
    assert settings.app_env == "development"
    assert settings.host == "127.0.0.1"
    assert settings.port == 8081
    assert settings.log_level == "DEBUG"
    assert settings.llm_mode == "fake"
    assert settings.llm_provider == "openai-compatible"
    assert settings.llm_api_key is None
    assert settings.llm_base_url is None
    assert settings.llm_timeout_seconds == 12.5
    assert settings.clinic_documents_path == project_root / "approved_docs"
    assert "test-password" not in repr(settings)


@pytest.mark.parametrize(
    "database_url",
    ["sqlite:///clinic.db", "postgresql:///missing-host", "postgresql://db"],
)
def test_settings_reject_non_postgresql_or_incomplete_database_urls(
    database_url: str,
) -> None:
    with pytest.raises(ValidationError, match="DATABASE_URL"):
        Settings(database_url=database_url)


def test_settings_reject_empty_host() -> None:
    with pytest.raises(ValidationError, match="HOST cannot be empty"):
        Settings(database_url=_DATABASE_URL, host="  ")


def test_settings_reject_out_of_range_port() -> None:
    with pytest.raises(ValidationError):
        Settings(database_url=_DATABASE_URL, port=70000)


def test_real_llm_mode_requires_model_and_base_url() -> None:
    with pytest.raises(ValidationError, match="LLM_MODEL is required"):
        Settings(database_url=_DATABASE_URL, llm_mode="real")

    with pytest.raises(ValidationError, match="LLM_BASE_URL is required"):
        Settings(database_url=_DATABASE_URL, llm_mode="real", llm_model="test-model")


@pytest.mark.parametrize(
    "base_url",
    [
        "ftp://provider.example/v1",
        "https://provider.example/v1?tenant=clinic",
        "https://provider.example/v1#fragment",
    ],
)
def test_real_llm_mode_rejects_invalid_base_urls(base_url: str) -> None:
    with pytest.raises(ValidationError, match="LLM_BASE_URL"):
        Settings(
            database_url=_DATABASE_URL,
            llm_mode="real",
            llm_model="test-model",
            llm_base_url=base_url,
        )


def test_openai_provider_requires_api_key() -> None:
    with pytest.raises(ValidationError, match="LLM_API_KEY is required"):
        Settings(
            database_url=_DATABASE_URL,
            llm_mode="real",
            llm_provider="openai",
            llm_model="test-model",
            llm_base_url="https://api.openai.com/v1",
        )


def test_get_settings_caches_validated_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    get_settings.cache_clear()
    monkeypatch.setenv("DATABASE_URL", _DATABASE_URL)
    monkeypatch.setenv("HOST", "localhost")
    try:
        first = get_settings()
        monkeypatch.setenv("HOST", "127.0.0.1")
        second = get_settings()
        assert second is first
        assert second.host == "localhost"
    finally:
        get_settings.cache_clear()
