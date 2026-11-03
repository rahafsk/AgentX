"""Unit tests for the fake and OpenAI-compatible LLM adapters."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable

import httpx
import pytest

from app.llm import (
    ChatMessage,
    FakeLLM,
    LLMError,
    LLMSettings,
    OpenAICompatibleLLM,
    create_llm,
)


def _messages() -> list[ChatMessage]:
    return [{"role": "user", "content": "Find an appointment."}]


def _real_settings(**overrides: object) -> LLMSettings:
    values: dict[str, object] = {
        "mode": "real",
        "provider": "openai-compatible",
        "model": "test-model",
        "base_url": "https://llm.example.test/v1",
        "api_key": "test-key",
        "timeout_seconds": 2,
    }
    values.update(overrides)
    return LLMSettings(**values)  # type: ignore[arg-type]


async def test_fake_llm_returns_configured_deterministic_response() -> None:
    llm = FakeLLM(response="fixed test response")

    assert await llm.complete(_messages()) == "fixed test response"
    assert await llm.complete(_messages()) == "fixed test response"


async def test_fake_llm_rejects_empty_message_list() -> None:
    with pytest.raises(ValueError, match="At least one chat message"):
        await FakeLLM().complete([])


async def test_fake_llm_rejects_unsupported_roles() -> None:
    invalid_messages = [{"role": "tool", "content": "not supported"}]  # type: ignore[list-item]

    with pytest.raises(ValueError, match="Unsupported chat message role"):
        await FakeLLM().complete(invalid_messages)


def test_from_env_uses_safe_fake_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "LLM_MODE",
        "LLM_PROVIDER",
        "LLM_MODEL",
        "LLM_BASE_URL",
        "LLM_API_KEY",
        "LLM_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = LLMSettings.from_env()

    assert settings.mode == "fake"
    assert settings.provider == "openai-compatible"
    assert settings.timeout_seconds == 30
    assert settings.api_key is None


@pytest.mark.parametrize("timeout", ["0", "-1", "nan", "not-a-number"])
def test_from_env_rejects_invalid_timeouts(
    monkeypatch: pytest.MonkeyPatch,
    timeout: str,
) -> None:
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", timeout)

    with pytest.raises(LLMError, match="LLM_TIMEOUT_SECONDS"):
        LLMSettings.from_env()


def test_from_env_rejects_unknown_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_MODE", "unknown")

    with pytest.raises(LLMError, match="LLM_MODE"):
        LLMSettings.from_env()


async def test_real_adapter_posts_to_configured_endpoint() -> None:
    async def responder(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://llm.example.test/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer test-key"
        assert json.loads(request.content) == {
            "model": "test-model",
            "messages": _messages(),
            "temperature": 0,
        }
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "slot details"}}]},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(responder)) as client:
        llm = OpenAICompatibleLLM(_real_settings(), client=client)

        assert await llm.complete(_messages()) == "slot details"


@pytest.mark.parametrize(
    ("handler", "message"),
    [
        (
            lambda request: _raise_timeout(request),
            "timed out",
        ),
        (
            lambda request: _raise_connection_error(request),
            "Could not reach",
        ),
    ],
)
async def test_real_adapter_maps_transport_errors(
    handler: Callable[[httpx.Request], Awaitable[httpx.Response]],
    message: str,
) -> None:
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        llm = OpenAICompatibleLLM(_real_settings(), client=client)

        with pytest.raises(LLMError, match=message):
            await llm.complete(_messages())


async def _raise_timeout(request: httpx.Request) -> httpx.Response:
    raise httpx.ReadTimeout("provider timed out", request=request)


async def _raise_connection_error(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("provider unavailable", request=request)


async def test_real_adapter_maps_http_status_errors() -> None:
    async def responder(_: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    async with httpx.AsyncClient(transport=httpx.MockTransport(responder)) as client:
        llm = OpenAICompatibleLLM(_real_settings(), client=client)

        with pytest.raises(LLMError, match="HTTP 503"):
            await llm.complete(_messages())


@pytest.mark.parametrize(
    "response_data",
    [{}, {"choices": []}, {"choices": [{"message": {"content": ""}}]}],
)
async def test_real_adapter_rejects_invalid_or_empty_responses(
    response_data: dict[str, object],
) -> None:
    async def responder(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=response_data)

    async with httpx.AsyncClient(transport=httpx.MockTransport(responder)) as client:
        llm = OpenAICompatibleLLM(_real_settings(), client=client)

        with pytest.raises(LLMError):
            await llm.complete(_messages())


def test_create_llm_uses_fake_adapter_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_MODE", raising=False)

    assert isinstance(create_llm(), FakeLLM)


def test_real_openai_provider_requires_api_key() -> None:
    settings = _real_settings(provider="openai", api_key=None)

    with pytest.raises(LLMError, match="LLM_API_KEY"):
        OpenAICompatibleLLM(settings)
