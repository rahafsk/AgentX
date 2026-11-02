"""Deterministic fake and OpenAI-compatible chat model adapters."""

from __future__ import annotations

import importlib
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypedDict

ChatRole = Literal["system", "user", "assistant"]


class ChatMessage(TypedDict):
    """A text-only chat message accepted by the adapter."""

    role: ChatRole
    content: str


class ChatModel(Protocol):
    """Minimal async interface used by application workflows."""

    async def complete(self, messages: Sequence[ChatMessage]) -> str:
        """Return one assistant response for the supplied conversation."""
        ...


class LLMError(RuntimeError):
    """Raised when model configuration or a provider request fails."""


def _http_client_library() -> Any:
    """Load the optional HTTP client only when real provider access is needed."""
    try:
        return importlib.import_module("httpx")
    except ImportError as exc:
        raise LLMError("The httpx package is required for real LLM mode.") from exc


@dataclass(frozen=True, slots=True)
class LLMSettings:
    """Configuration for the fake or OpenAI-compatible model adapter."""

    mode: Literal["fake", "real"] = "fake"
    provider: str = "openai-compatible"
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    timeout_seconds: float = 30.0

    @classmethod
    def from_env(cls) -> LLMSettings:
        """Create validated settings from the project's LLM environment variables."""
        mode = (os.getenv("LLM_MODE") or "fake").strip().lower()
        if mode not in {"fake", "real"}:
            raise LLMError("LLM_MODE must be either 'fake' or 'real'.")

        timeout_value = os.getenv("LLM_TIMEOUT_SECONDS") or "30"
        try:
            timeout_seconds = float(timeout_value)
        except ValueError as exc:
            raise LLMError("LLM_TIMEOUT_SECONDS must be a positive number.") from exc
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise LLMError("LLM_TIMEOUT_SECONDS must be a positive number.")

        return cls(
            mode=mode,  # type: ignore[arg-type]
            provider=(os.getenv("LLM_PROVIDER") or "openai-compatible").strip().lower(),
            model=(os.getenv("LLM_MODEL") or "").strip() or None,
            base_url=(os.getenv("LLM_BASE_URL") or "").strip().rstrip("/") or None,
            api_key=(os.getenv("LLM_API_KEY") or "").strip() or None,
            timeout_seconds=timeout_seconds,
        )


def _validated_messages(messages: Sequence[ChatMessage]) -> list[dict[str, str]]:
    """Validate and copy messages so provider requests contain only text fields."""
    if not messages:
        raise ValueError("At least one chat message is required.")

    result: list[dict[str, str]] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"Unsupported chat message role: {role!r}.")
        if not isinstance(content, str):
            raise ValueError("Chat message content must be text.")
        result.append({"role": role, "content": content})
    return result


class FakeLLM:
    """A deterministic fake model for local development and automated tests."""

    def __init__(
        self,
        response: str = (
            "I can help with appointments and general clinic information. "
            "I cannot diagnose or recommend treatment."
        ),
    ) -> None:
        self._response = response

    async def complete(self, messages: Sequence[ChatMessage]) -> str:
        _validated_messages(messages)
        return self._response


class OpenAICompatibleLLM:
    """Call a text-chat endpoint compatible with the OpenAI chat-completions API."""

    def __init__(
        self,
        settings: LLMSettings,
        *,
        client: Any | None = None,
    ) -> None:
        if settings.mode != "real":
            raise LLMError("OpenAICompatibleLLM requires LLM_MODE=real.")
        if settings.provider not in {"openai", "openai-compatible"}:
            raise LLMError(
                "LLM_PROVIDER must be 'openai' or 'openai-compatible' in real mode."
            )
        if settings.provider == "openai" and not settings.api_key:
            raise LLMError("LLM_API_KEY is required when LLM_PROVIDER=openai.")
        if not settings.model:
            raise LLMError("LLM_MODEL is required when LLM_MODE=real.")
        if not settings.base_url:
            raise LLMError("LLM_BASE_URL is required when LLM_MODE=real.")
        if not settings.base_url.startswith(("https://", "http://")):
            raise LLMError("LLM_BASE_URL must be an HTTP or HTTPS URL.")
        if not math.isfinite(settings.timeout_seconds) or settings.timeout_seconds <= 0:
            raise LLMError("LLM_TIMEOUT_SECONDS must be a positive number.")

        httpx = _http_client_library()
        self._settings = settings
        self._client = client
        self._timeout = httpx.Timeout(settings.timeout_seconds)

    async def complete(self, messages: Sequence[ChatMessage]) -> str:
        """Send messages to the configured provider with a bounded timeout."""
        httpx = _http_client_library()
        request_messages = _validated_messages(messages)
        headers = {"Accept": "application/json"}
        if self._settings.api_key:
            headers["Authorization"] = f"Bearer {self._settings.api_key}"

        payload = {
            "model": self._settings.model,
            "messages": request_messages,
            "temperature": 0,
        }
        endpoint = f"{self._settings.base_url}/chat/completions"

        try:
            if self._client is None:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.post(endpoint, headers=headers, json=payload)
            else:
                response = await self._client.post(
                    endpoint,
                    headers=headers,
                    json=payload,
                    timeout=self._timeout,
                )
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise LLMError("The LLM provider request timed out.") from exc
        except httpx.HTTPStatusError as exc:
            raise LLMError(
                f"The LLM provider returned HTTP {exc.response.status_code}."
            ) from exc
        except httpx.RequestError as exc:
            raise LLMError("Could not reach the configured LLM provider.") from exc
        except httpx.HTTPError as exc:
            raise LLMError("The LLM provider request failed.") from exc

        try:
            response_data = response.json()
            content = response_data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMError("The LLM provider returned an invalid chat response.") from exc

        if not isinstance(content, str) or not content.strip():
            raise LLMError("The LLM provider returned an empty assistant response.")
        return content


def create_llm(settings: LLMSettings | None = None) -> ChatModel:
    """Build the configured model adapter; fake mode is the safe default."""
    resolved_settings = settings or LLMSettings.from_env()
    if resolved_settings.mode == "fake":
        return FakeLLM()
    return OpenAICompatibleLLM(resolved_settings)
