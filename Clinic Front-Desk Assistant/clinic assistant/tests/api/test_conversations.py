"""API tests for authenticated, idempotent conversation endpoints."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.conversations import router
from app.api.dependencies import get_authenticated_owner_reference
from app.repositories.errors import ConversationNotFound
from app.repositories.models import Conversation

_CONVERSATION_ID = UUID("12345678-1234-5678-1234-567812345678")
_NOW = datetime(2026, 1, 1, tzinfo=UTC)
_OWNER = "opaque-authenticated-owner"
_CREATE_REQUEST_ID = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
_MESSAGE_REQUEST_ID = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"


def _conversation() -> Conversation:
    return Conversation(
        id=_CONVERSATION_ID,
        thread_id=str(_CONVERSATION_ID),
        created_at=_NOW,
        updated_at=_NOW,
        owner_reference=_OWNER,
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
        self.created: list[tuple[str, UUID]] = []
        self.processed: list[tuple[UUID, str, str, UUID]] = []

    async def create_conversation(
        self,
        owner_reference: str,
        request_id: UUID,
    ) -> Conversation:
        self.created.append((owner_reference, request_id))
        return self._conversation

    async def get_conversation(
        self,
        conversation_id: UUID,
        *,
        owner_reference: str,
    ) -> Conversation:
        if self._get_error is not None:
            raise self._get_error
        assert owner_reference == _OWNER
        return self._conversation

    async def process_message(
        self,
        conversation_id: UUID,
        message: str,
        *,
        owner_reference: str,
        request_id: UUID,
    ) -> str:
        self.processed.append((conversation_id, message, owner_reference, request_id))
        if self._process_error is not None:
            raise self._process_error
        return self._response


def _client(
    service: _FakeService | None,
    *,
    owner_reference: str | None = _OWNER,
) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    if service is not None:
        app.state.conversation_service = service
    if owner_reference is not None:
        app.dependency_overrides[get_authenticated_owner_reference] = lambda: owner_reference
    return TestClient(app)


def test_create_conversation_returns_metadata_and_uses_owner_and_key() -> None:
    service = _FakeService()
    response = _client(service).post(
        "/conversations",
        headers={"Idempotency-Key": _CREATE_REQUEST_ID},
    )

    assert response.status_code == 201
    assert response.json() == {
        "id": str(_CONVERSATION_ID),
        "thread_id": str(_CONVERSATION_ID),
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
    }
    assert service.created == [(_OWNER, UUID(_CREATE_REQUEST_ID))]


def test_create_conversation_requires_key_and_authenticated_owner() -> None:
    service = _FakeService()
    missing_key = _client(service).post("/conversations")
    unauthenticated = _client(service, owner_reference=None).post(
        "/conversations",
        headers={"Idempotency-Key": _CREATE_REQUEST_ID},
    )

    assert missing_key.status_code == 422
    assert unauthenticated.status_code == 401
    assert service.created == []


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


def test_send_message_returns_workflow_response_and_trusted_context() -> None:
    service = _FakeService(response="Appointments are available.")
    client = _client(service)

    response = client.post(
        f"/conversations/{_CONVERSATION_ID}/messages",
        headers={"Idempotency-Key": _MESSAGE_REQUEST_ID},
        json={"message": "  I need an appointment.  "},
    )

    assert response.status_code == 200
    assert response.json() == {"response": "Appointments are available."}
    assert service.processed == [
        (_CONVERSATION_ID, "I need an appointment.", _OWNER, UUID(_MESSAGE_REQUEST_ID))
    ]


def test_send_message_rejects_blank_message() -> None:
    service = _FakeService()
    client = _client(service)

    response = client.post(
        f"/conversations/{_CONVERSATION_ID}/messages",
        headers={"Idempotency-Key": _MESSAGE_REQUEST_ID},
        json={"message": "   "},
    )

    assert response.status_code == 422
    assert service.processed == []


def test_send_message_returns_not_found() -> None:
    client = _client(_FakeService(process_error=ConversationNotFound()))

    response = client.post(
        f"/conversations/{_CONVERSATION_ID}/messages",
        headers={"Idempotency-Key": _MESSAGE_REQUEST_ID},
        json={"message": "Hello"},
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "Conversation not found."}


def test_send_message_returns_server_error_for_workflow_failure() -> None:
    client = _client(_FakeService(process_error=RuntimeError("workflow failed")))

    response = client.post(
        f"/conversations/{_CONVERSATION_ID}/messages",
        headers={"Idempotency-Key": _MESSAGE_REQUEST_ID},
        json={"message": "Hello"},
    )

    assert response.status_code == 500
    assert response.json() == {"detail": "The assistant could not process the message."}


def test_conversation_endpoints_require_configured_service() -> None:
    client = _client(None)

    response = client.post(
        "/conversations",
        headers={"Idempotency-Key": _CREATE_REQUEST_ID},
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "Conversation service is not configured."}
