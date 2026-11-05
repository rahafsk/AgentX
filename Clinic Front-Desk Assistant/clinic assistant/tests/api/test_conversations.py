"""API tests for the conversation endpoints."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.conversations import router
from app.repositories.errors import ConversationNotFound
from app.repositories.models import Conversation

_CONVERSATION_ID = UUID("12345678-1234-5678-1234-567812345678")
_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _conversation() -> Conversation:
    return Conversation(
        id=_CONVERSATION_ID,
        thread_id=str(_CONVERSATION_ID),
        created_at=_NOW,
        updated_at=_NOW,
    )


class _FakeService:
    def __init__(
        self,
        conversation: Conversation | None = None,
        *,
        response: str = "We can help with that.",
        get_error: Exception | None = None,
        process_error: Exception | None = None,
    ) -> None:
        self._conversation = conversation or _conversation()
        self._response = response
        self._get_error = get_error
        self._process_error = process_error
        self.processed: list[tuple[UUID, str]] = []

    def create_conversation(self) -> Conversation:
        return self._conversation

    def get_conversation(self, conversation_id: UUID) -> Conversation:
        if self._get_error is not None:
            raise self._get_error
        return self._conversation

    async def process_message(self, conversation_id: UUID, message: str) -> str:
        self.processed.append((conversation_id, message))
        if self._process_error is not None:
            raise self._process_error
        return self._response


def _client(service: _FakeService | None) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    if service is not None:
        app.state.conversation_service = service
    return TestClient(app)


def test_create_conversation_returns_metadata() -> None:
    client = _client(_FakeService())

    response = client.post("/conversations")

    assert response.status_code == 201
    assert response.json() == {
        "id": str(_CONVERSATION_ID),
        "thread_id": str(_CONVERSATION_ID),
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
    }


def test_get_conversation_returns_metadata() -> None:
    client = _client(_FakeService())

    response = client.get(f"/conversations/{_CONVERSATION_ID}")

    assert response.status_code == 200
    assert response.json()["id"] == str(_CONVERSATION_ID)


def test_get_conversation_returns_not_found() -> None:
    client = _client(_FakeService(get_error=ConversationNotFound()))

    response = client.get(f"/conversations/{_CONVERSATION_ID}")

    assert response.status_code == 404
    assert response.json() == {"detail": "Conversation not found."}


def test_send_message_returns_workflow_response() -> None:
    service = _FakeService(response="Appointments are available.")
    client = _client(service)

    response = client.post(
        f"/conversations/{_CONVERSATION_ID}/messages",
        json={"message": "  I need an appointment.  "},
    )

    assert response.status_code == 200
    assert response.json() == {"response": "Appointments are available."}
    assert service.processed == [(_CONVERSATION_ID, "I need an appointment.")]


def test_send_message_rejects_blank_message() -> None:
    service = _FakeService()
    client = _client(service)

    response = client.post(
        f"/conversations/{_CONVERSATION_ID}/messages",
        json={"message": "   "},
    )

    assert response.status_code == 422
    assert service.processed == []


def test_send_message_returns_not_found() -> None:
    client = _client(_FakeService(process_error=ConversationNotFound()))

    response = client.post(
        f"/conversations/{_CONVERSATION_ID}/messages",
        json={"message": "Hello"},
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "Conversation not found."}


def test_send_message_returns_server_error_for_workflow_failure() -> None:
    client = _client(_FakeService(process_error=RuntimeError("workflow failed")))

    response = client.post(
        f"/conversations/{_CONVERSATION_ID}/messages",
        json={"message": "Hello"},
    )

    assert response.status_code == 500
    assert response.json() == {
        "detail": "The assistant could not process the message.",
    }


def test_conversation_endpoints_require_configured_service() -> None:
    client = _client(None)

    response = client.post("/conversations")

    assert response.status_code == 503
    assert response.json() == {"detail": "Conversation service is not configured."}
