"""API tests for the conversation endpoints."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.conversations import router
from app.repositories.conversations import IdempotencyKeyConflict
from app.repositories.errors import ConversationNotFound
from app.repositories.models import Conversation

_CONVERSATION_ID = UUID("12345678-1234-5678-1234-567812345678")
_REQUEST_ID = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
_NOW = datetime(2026, 1, 1, tzinfo=UTC)
_HEADERS = {
    "X-Owner-Reference": "owner-1",
    "Idempotency-Key": str(_REQUEST_ID),
}


def _conversation() -> Conversation:
    return Conversation(
        id=_CONVERSATION_ID,
        thread_id=str(_CONVERSATION_ID),
        created_at=_NOW,
        updated_at=_NOW,
        owner_reference="owner-1",
    )


class _FakeService:
    def __init__(
        self,
        conversation: Conversation | None = None,
        *,
        response: str = "We can help with that.",
        get_error: Exception | None = None,
        process_error: Exception | None = None,
        create_error: Exception | None = None,
    ) -> None:
        self._conversation = conversation or _conversation()
        self._response = response
        self._get_error = get_error
        self._process_error = process_error
        self._create_error = create_error
        self.processed: list[tuple[UUID, str]] = []
        self.owners: list[str] = []
        self.request_ids: list[UUID] = []
        self.created: list[tuple[str, str, str]] = []
        self._created_hashes: dict[tuple[str, str], str] = {}

    async def create_conversation(
        self,
        owner_reference: str,
        *,
        idempotency_key: str,
        request_hash: str,
    ) -> Conversation:
        self.created.append((owner_reference, idempotency_key, request_hash))
        if self._create_error is not None:
            raise self._create_error

        key = (owner_reference, idempotency_key)
        existing_hash = self._created_hashes.get(key)
        if existing_hash is not None and existing_hash != request_hash:
            raise IdempotencyKeyConflict()
        self._created_hashes[key] = request_hash
        return self._conversation

    async def get_conversation(
        self,
        conversation_id: UUID,
        *,
        owner_reference: str,
    ) -> Conversation:
        self.owners.append(owner_reference)
        if self._get_error is not None:
            raise self._get_error
        return self._conversation

    async def process_message(
        self,
        conversation_id: UUID,
        message: str,
        *,
        owner_reference: str,
        request_id: UUID,
    ) -> str:
        self.processed.append((conversation_id, message))
        self.owners.append(owner_reference)
        self.request_ids.append(request_id)
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
    service = _FakeService()
    client = _client(service)

    response = client.post("/conversations", headers=_HEADERS)

    assert response.status_code == 201
    assert response.json() == {
        "id": str(_CONVERSATION_ID),
        "thread_id": str(_CONVERSATION_ID),
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
    }
    assert len(service.created) == 1
    assert service.created[0][0] == "owner-1"
    assert service.created[0][1] == str(_REQUEST_ID)
    assert service.created[0][2]


def test_create_conversation_rejects_reused_key_with_different_content() -> None:
    service = _FakeService(create_error=IdempotencyKeyConflict())
    client = _client(service)

    response = client.post("/conversations", headers=_HEADERS)

    assert response.status_code == 409
    assert "different request content" in response.json()["detail"]


def test_create_conversation_retry_returns_original_conversation() -> None:
    service = _FakeService()
    client = _client(service)

    first = client.post("/conversations", headers=_HEADERS)
    second = client.post("/conversations", headers=_HEADERS)

    assert first.status_code == 201
    assert second.status_code == 201
    assert first.json() == second.json()
    assert len(service.created) == 2


def test_create_conversation_key_reuse_with_changed_body_is_conflict() -> None:
    service = _FakeService()
    client = _client(service)

    first = client.post("/conversations", headers=_HEADERS)
    second = client.post(
        "/conversations",
        headers=_HEADERS,
        json={"changed": True},
    )

    assert first.status_code == 201
    assert second.status_code == 409


def test_get_conversation_returns_metadata() -> None:
    client = _client(_FakeService())

    response = client.get(
        f"/conversations/{_CONVERSATION_ID}",
        headers=_HEADERS,
    )

    assert response.status_code == 200
    assert response.json()["id"] == str(_CONVERSATION_ID)


def test_get_conversation_returns_not_found() -> None:
    client = _client(_FakeService(get_error=ConversationNotFound()))

    response = client.get(
        f"/conversations/{_CONVERSATION_ID}",
        headers=_HEADERS,
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "Conversation not found."}


def test_send_message_returns_workflow_response() -> None:
    service = _FakeService(response="Appointments are available.")
    client = _client(service)

    response = client.post(
        f"/conversations/{_CONVERSATION_ID}/messages",
        json={"message": "  I need an appointment.  "},
        headers=_HEADERS,
    )

    assert response.status_code == 200
    assert response.json() == {"response": "Appointments are available."}
    assert service.processed == [(_CONVERSATION_ID, "I need an appointment.")]
    assert service.owners == ["owner-1"]
    assert service.request_ids == [_REQUEST_ID]


def test_send_message_rejects_blank_message() -> None:
    service = _FakeService()
    client = _client(service)

    response = client.post(
        f"/conversations/{_CONVERSATION_ID}/messages",
        json={"message": "   "},
        headers=_HEADERS,
    )

    assert response.status_code == 422
    assert service.processed == []


def test_send_message_returns_not_found() -> None:
    client = _client(_FakeService(process_error=ConversationNotFound()))

    response = client.post(
        f"/conversations/{_CONVERSATION_ID}/messages",
        json={"message": "Hello"},
        headers=_HEADERS,
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "Conversation not found."}


def test_send_message_returns_server_error_for_workflow_failure() -> None:
    client = _client(_FakeService(process_error=RuntimeError("workflow failed")))

    response = client.post(
        f"/conversations/{_CONVERSATION_ID}/messages",
        json={"message": "Hello"},
        headers=_HEADERS,
    )

    assert response.status_code == 500
    assert response.json() == {
        "detail": "The assistant could not process the message.",
    }


def test_conversation_endpoints_require_configured_service() -> None:
    client = _client(None)

    response = client.post("/conversations", headers=_HEADERS)

    assert response.status_code == 503
    assert response.json() == {"detail": "Conversation service is not configured."}


def test_conversation_endpoints_require_owner_header() -> None:
    client = _client(_FakeService())

    response = client.post("/conversations")

    assert response.status_code == 422
