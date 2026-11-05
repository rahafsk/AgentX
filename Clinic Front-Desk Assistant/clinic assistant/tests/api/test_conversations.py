"""API tests for authenticated, durable-idempotency conversation operations."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.conversations import router
from app.api.dependencies import get_authenticated_owner_reference
from app.repositories.conversations import IdempotencyKeyConflict
from app.repositories.errors import ConversationNotFound
from app.repositories.models import Conversation

_CONVERSATION_ID = UUID("12345678-1234-5678-1234-567812345678")
_REQUEST_ID = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
_NOW = datetime(2026, 1, 1, tzinfo=UTC)
_OWNER = "opaque-authenticated-owner"


def _conversation() -> Conversation:
    return Conversation(
        id=_CONVERSATION_ID,
        thread_id=str(_CONVERSATION_ID),
        created_at=_NOW,
        updated_at=_NOW,
        owner_reference=_OWNER,
    )


class _FakeService:
    def __init__(self, *, process_error: Exception | None = None) -> None:
        self.process_error = process_error
        self.created: list[tuple[str, str, str]] = []
        self.processed: list[tuple[UUID, str, str, UUID]] = []
        self._creation_hashes: dict[tuple[str, str], str] = {}
        self._message_hashes: dict[tuple[UUID, UUID], str] = {}

    async def create_conversation(
        self,
        owner_reference: str,
        *,
        idempotency_key: str,
        request_hash: str,
    ) -> Conversation:
        self.created.append((owner_reference, idempotency_key, request_hash))
        key = (owner_reference, idempotency_key)
        previous_hash = self._creation_hashes.get(key)
        if previous_hash is not None and previous_hash != request_hash:
            raise IdempotencyKeyConflict("Request content changed.")
        self._creation_hashes[key] = request_hash
        return _conversation()

    async def get_conversation(
        self,
        conversation_id: UUID,
        *,
        owner_reference: str,
    ) -> Conversation:
        if owner_reference != _OWNER or conversation_id != _CONVERSATION_ID:
            raise ConversationNotFound("Conversation was not found.")
        return _conversation()

    async def process_message(
        self,
        conversation_id: UUID,
        message: str,
        *,
        owner_reference: str,
        request_id: UUID,
    ) -> str:
        self.processed.append((conversation_id, message, owner_reference, request_id))
        if self.process_error is not None:
            raise self.process_error
        key = (conversation_id, request_id)
        existing_message = self._message_hashes.get(key)
        if existing_message is not None and existing_message != message:
            raise IdempotencyKeyConflict("Request content changed.")
        self._message_hashes[key] = message
        return "We can help with that."


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


def test_create_conversation_uses_authenticated_owner_and_idempotency_key() -> None:
    service = _FakeService()
    client = _client(service)

    first = client.post("/conversations", headers={"Idempotency-Key": _REQUEST_ID})
    retry = client.post("/conversations", headers={"Idempotency-Key": _REQUEST_ID})

    assert first.status_code == retry.status_code == 201
    assert first.json() == retry.json()
    assert [call[0] for call in service.created] == [_OWNER, _OWNER]
    assert all(call[1] == _REQUEST_ID for call in service.created)
    assert service.created[0][2] == service.created[1][2]


def test_create_conversation_rejects_key_reuse_with_different_content() -> None:
    service = _FakeService()
    client = _client(service)

    first = client.post("/conversations", headers={"Idempotency-Key": _REQUEST_ID})
    changed = client.post(
        "/conversations",
        headers={"Idempotency-Key": _REQUEST_ID},
        json={"different": "body"},
    )

    assert first.status_code == 201
    assert changed.status_code == 409


def test_conversation_creation_requires_trusted_owner_and_key() -> None:
    service = _FakeService()
    missing_key = _client(service).post("/conversations")
    missing_owner = _client(service, owner_reference=None).post(
        "/conversations",
        headers={"Idempotency-Key": _REQUEST_ID},
    )

    assert missing_key.status_code == 422
    assert missing_owner.status_code == 401
    assert service.created == []


def test_client_owner_header_cannot_override_trusted_identity() -> None:
    service = _FakeService()
    response = _client(service).post(
        "/conversations",
        headers={
            "Idempotency-Key": _REQUEST_ID,
            "X-Owner-Reference": "forged-owner",
        },
    )

    assert response.status_code == 201
    assert service.created[0][0] == _OWNER


def test_get_conversation_is_owner_scoped() -> None:
    client = _client(_FakeService())
    response = client.get(f"/conversations/{_CONVERSATION_ID}")

    assert response.status_code == 200
    assert response.json()["id"] == str(_CONVERSATION_ID)

    foreign = _client(_FakeService(), owner_reference="another-owner").get(
        f"/conversations/{_CONVERSATION_ID}"
    )
    assert foreign.status_code == 404


def test_message_uses_authenticated_owner_and_persistent_request_key() -> None:
    service = _FakeService()
    response = _client(service).post(
        f"/conversations/{_CONVERSATION_ID}/messages",
        headers={
            "Idempotency-Key": _REQUEST_ID,
            "X-Owner-Reference": "forged-owner",
        },
        json={"message": "  I need an appointment.  "},
    )

    assert response.status_code == 200
    assert response.json() == {"response": "We can help with that."}
    assert service.processed == [
        (_CONVERSATION_ID, "I need an appointment.", _OWNER, UUID(_REQUEST_ID))
    ]


def test_message_requires_idempotency_key_and_owner_authentication() -> None:
    service = _FakeService()
    missing_key = _client(service).post(
        f"/conversations/{_CONVERSATION_ID}/messages",
        json={"message": "Hello"},
    )
    missing_owner = _client(service, owner_reference=None).post(
        f"/conversations/{_CONVERSATION_ID}/messages",
        headers={"Idempotency-Key": _REQUEST_ID},
        json={"message": "Hello"},
    )

    assert missing_key.status_code == 422
    assert missing_owner.status_code == 401
    assert service.processed == []


def test_message_key_reuse_with_changed_content_is_a_conflict() -> None:
    client = _client(_FakeService())
    first = client.post(
        f"/conversations/{_CONVERSATION_ID}/messages",
        headers={"Idempotency-Key": _REQUEST_ID},
        json={"message": "First message."},
    )
    changed = client.post(
        f"/conversations/{_CONVERSATION_ID}/messages",
        headers={"Idempotency-Key": _REQUEST_ID},
        json={"message": "Different message."},
    )

    assert first.status_code == 200
    assert changed.status_code == 409


def test_blank_message_and_missing_conversation_are_rejected() -> None:
    blank = _client(_FakeService()).post(
        f"/conversations/{_CONVERSATION_ID}/messages",
        headers={"Idempotency-Key": _REQUEST_ID},
        json={"message": "   "},
    )
    missing = _client(_FakeService(process_error=ConversationNotFound())).post(
        f"/conversations/{_CONVERSATION_ID}/messages",
        headers={"Idempotency-Key": _REQUEST_ID},
        json={"message": "Hello"},
    )

    assert blank.status_code == 422
    assert missing.status_code == 404


def test_message_body_rejects_caller_supplied_owner_identity() -> None:
    service = _FakeService()
    response = _client(service).post(
        f"/conversations/{_CONVERSATION_ID}/messages",
        headers={"Idempotency-Key": _REQUEST_ID},
        json={"message": "Hello", "owner_reference": "forged-owner"},
    )

    assert response.status_code == 422
    assert service.processed == []


def test_message_workflow_failure_returns_safe_error() -> None:
    response = _client(_FakeService(process_error=RuntimeError("workflow failed"))).post(
        f"/conversations/{_CONVERSATION_ID}/messages",
        headers={"Idempotency-Key": _REQUEST_ID},
        json={"message": "Hello"},
    )

    assert response.status_code == 500
    assert response.json() == {"detail": "The assistant could not process the message."}


def test_conversation_routes_return_503_without_a_service() -> None:
    response = _client(None).post(
        "/conversations",
        headers={"Idempotency-Key": _REQUEST_ID},
    )

    assert response.status_code == 503
