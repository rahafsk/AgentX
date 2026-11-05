"""Unit tests for the PostgreSQL conversation repository."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from app.repositories import conversations as conversation_module
from app.repositories.conversations import ConversationRepository


class _FakeCursor:
    """A fake cursor that simulates PostgreSQL responses."""

    def __init__(self, row: dict[str, object] | None = None) -> None:
        self.row = row
        self.statement = ""
        self.parameters: object | None = None

    def __enter__(self) -> _FakeCursor:
        return self

    def __exit__(self, *args: object) -> bool:
        return False

    def execute(
        self,
        statement: str,
        parameters: object | None = None,
    ) -> None:
        self.statement = statement
        self.parameters = parameters

    def fetchone(self) -> dict[str, object] | None:
        return self.row


class _FakeTransaction:
    def __enter__(self) -> _FakeTransaction:
        return self

    def __exit__(self, *args: object) -> bool:
        return False


class _FakeConnection:
    """A fake connection used by repository unit tests."""

    def __init__(self, row: dict[str, object] | None = None) -> None:
        self.cursor_instance = _FakeCursor(row)

    def cursor(self, *, row_factory: object | None = None) -> _FakeCursor:
        return self.cursor_instance

    def transaction(self) -> _FakeTransaction:
        return _FakeTransaction()


def _conversation_row(conversation_id: UUID) -> dict[str, object]:
    """Create a fake database row."""

    now = datetime.now(UTC)

    return {
        "id": conversation_id,
        "thread_id": str(conversation_id),
        "created_at": now,
        "updated_at": now,
        "owner_reference": "opaque-test-owner",
    }


def test_create_conversation(monkeypatch) -> None:
    conversation_id = UUID("12345678-1234-5678-1234-567812345678")

    monkeypatch.setattr(
        conversation_module,
        "uuid4",
        lambda: conversation_id,
    )

    fake_connection = _FakeConnection(_conversation_row(conversation_id))
    repository = ConversationRepository(fake_connection)

    conversation = repository.create("opaque-test-owner")

    assert conversation.id == conversation_id
    assert conversation.thread_id == str(conversation_id)
    assert conversation.owner_reference == "opaque-test-owner"
    assert "INSERT INTO conversations" in fake_connection.cursor_instance.statement
    assert fake_connection.cursor_instance.parameters == (
        conversation_id,
        str(conversation_id),
        "opaque-test-owner",
    )


def test_get_conversation() -> None:
    conversation_id = UUID("12345678-1234-5678-1234-567812345678")

    fake_connection = _FakeConnection(_conversation_row(conversation_id))
    repository = ConversationRepository(fake_connection)

    conversation = repository.get(conversation_id)

    assert conversation is not None
    assert conversation.id == conversation_id
    assert "WHERE id = %s" in fake_connection.cursor_instance.statement
    assert fake_connection.cursor_instance.parameters == (conversation_id,)


def test_get_by_thread_id() -> None:
    conversation_id = UUID("12345678-1234-5678-1234-567812345678")
    thread_id = str(conversation_id)

    fake_connection = _FakeConnection(_conversation_row(conversation_id))
    repository = ConversationRepository(fake_connection)

    conversation = repository.get_by_thread_id(thread_id)

    assert conversation is not None
    assert conversation.thread_id == thread_id
    assert "WHERE thread_id = %s" in fake_connection.cursor_instance.statement
    assert fake_connection.cursor_instance.parameters == (thread_id,)


def test_get_for_owner_scopes_lookup_to_owner() -> None:
    conversation_id = UUID("12345678-1234-5678-1234-567812345678")

    fake_connection = _FakeConnection(_conversation_row(conversation_id))
    repository = ConversationRepository(fake_connection)

    conversation = repository.get_for_owner(conversation_id, "opaque-test-owner")

    assert conversation is not None
    assert conversation.owner_reference == "opaque-test-owner"
    assert "WHERE id = %s AND owner_reference = %s" in fake_connection.cursor_instance.statement
    assert fake_connection.cursor_instance.parameters == (
        conversation_id,
        "opaque-test-owner",
    )


def test_touch_conversation() -> None:
    conversation_id = UUID("12345678-1234-5678-1234-567812345678")

    fake_connection = _FakeConnection(_conversation_row(conversation_id))
    repository = ConversationRepository(fake_connection)

    conversation = repository.touch(conversation_id)

    assert conversation is not None
    assert conversation.id == conversation_id
    assert "UPDATE conversations" in fake_connection.cursor_instance.statement
    assert "RETURNING" in fake_connection.cursor_instance.statement
    assert fake_connection.cursor_instance.parameters == (conversation_id,)


def test_touch_for_owner_is_scoped() -> None:
    conversation_id = UUID("12345678-1234-5678-1234-567812345678")

    fake_connection = _FakeConnection(_conversation_row(conversation_id))
    repository = ConversationRepository(fake_connection)

    conversation = repository.touch_for_owner(conversation_id, "opaque-test-owner")

    assert conversation is not None
    assert conversation.id == conversation_id
    assert "WHERE id = %s AND owner_reference = %s" in fake_connection.cursor_instance.statement
    assert fake_connection.cursor_instance.parameters == (
        conversation_id,
        "opaque-test-owner",
    )


def test_get_returns_none_when_conversation_does_not_exist() -> None:
    conversation_id = UUID("12345678-1234-5678-1234-567812345678")

    fake_connection = _FakeConnection(row=None)
    repository = ConversationRepository(fake_connection)

    conversation = repository.get(conversation_id)

    assert conversation is None


def test_touch_returns_none_when_conversation_does_not_exist() -> None:
    conversation_id = UUID("12345678-1234-5678-1234-567812345678")

    fake_connection = _FakeConnection(row=None)
    repository = ConversationRepository(fake_connection)

    conversation = repository.touch(conversation_id)

    assert conversation is None
