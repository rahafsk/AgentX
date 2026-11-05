"""PostgreSQL repository for conversation metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID, uuid4

from psycopg import Connection
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from app.repositories.errors import RepositoryError
from app.repositories.models import Conversation


class IdempotencyKeyConflict(RepositoryError):
    """The same key was reused with different request content."""


@dataclass(frozen=True, slots=True)
class ConversationRequestClaim:
    """A durable idempotency claim and its completed workflow result, if any."""

    created: bool
    result: dict[str, Any] | None


def _conversation_from_row(
    row: dict[str, Any],
) -> Conversation:
    return Conversation(
        id=row["id"],
        thread_id=row["thread_id"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        owner_reference=row["owner_reference"],
        creation_request_id=row["creation_request_id"],
    )


class ConversationRepository:
    """Persist conversation metadata; LangGraph stores the checkpoint state.

    Each method uses a short transaction scope. Callers must never hold an
    outer database transaction open while awaiting a LangGraph invocation.
    """

    def __init__(self, connection: Connection[Any]) -> None:
        self._connection = connection

    def create(self, owner_reference: str, request_id: UUID) -> Conversation:
        """Create or replay metadata for an owner's idempotent create request."""
        owner_reference = owner_reference.strip()
        if not owner_reference:
            raise ValueError("owner_reference cannot be empty.")

        conversation_id = uuid4()
        thread_id = str(conversation_id)

        with self._connection.transaction():
            with self._connection.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    """
                    INSERT INTO conversations (
                        id, thread_id, owner_reference, creation_request_id
                    )
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT DO NOTHING
                    RETURNING id, thread_id, created_at, updated_at,
                              owner_reference, creation_request_id
                    """,
                    (conversation_id, thread_id, owner_reference, request_id),
                )
                row = cursor.fetchone()
                if row is None:
                    cursor.execute(
                        """
                        SELECT id, thread_id, created_at, updated_at,
                               owner_reference, creation_request_id
                        FROM conversations
                        WHERE owner_reference = %s AND creation_request_id = %s
                        """,
                        (owner_reference, request_id),
                    )
                    row = cursor.fetchone()

        if row is None:
            raise RuntimeError("PostgreSQL did not return the inserted conversation.")

        return _conversation_from_row(row)

    def create_idempotent(
        self,
        owner_reference: str,
        idempotency_key: str,
        request_hash: str,
    ) -> Conversation:
        """Create a conversation and its idempotency record atomically.

        The advisory lock serializes concurrent retries with the same owner and
        key, while the primary key remains the final duplicate guard.
        """
        owner_reference = owner_reference.strip()
        idempotency_key = idempotency_key.strip()
        request_hash = request_hash.strip()

        if not owner_reference:
            raise ValueError("owner_reference cannot be empty.")
        if not idempotency_key:
            raise ValueError("idempotency_key cannot be empty.")
        if not request_hash:
            raise ValueError("request_hash cannot be empty.")

        conversation_id = uuid4()
        thread_id = str(conversation_id)

        with self._connection.transaction():
            with self._connection.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                    (f"{owner_reference}:{idempotency_key}",),
                )
                cursor.execute(
                    """
                    SELECT idempotency.request_hash,
                           conversation.id,
                           conversation.thread_id,
                           conversation.created_at,
                           conversation.updated_at,
                           conversation.owner_reference,
                           conversation.creation_request_id
                    FROM conversation_idempotency AS idempotency
                    JOIN conversations AS conversation
                      ON conversation.id = idempotency.conversation_id
                    WHERE idempotency.owner_reference = %s
                      AND idempotency.idempotency_key = %s
                    """,
                    (owner_reference, idempotency_key),
                )
                existing = cursor.fetchone()

                if existing is not None:
                    if existing["request_hash"] != request_hash:
                        raise IdempotencyKeyConflict(
                            "The idempotency key was reused with different request content."
                        )
                    return _conversation_from_row(existing)

                cursor.execute(
                    """
                    INSERT INTO conversations (id, thread_id, owner_reference)
                    VALUES (%s, %s, %s)
                    RETURNING id, thread_id, created_at, updated_at, owner_reference
                    """,
                    (conversation_id, thread_id, owner_reference),
                )
                row = cursor.fetchone()

                if row is None:
                    raise RuntimeError("PostgreSQL did not return the inserted conversation.")

                cursor.execute(
                    """
                    INSERT INTO conversation_idempotency (
                        owner_reference, idempotency_key, request_hash, conversation_id
                    )
                    VALUES (%s, %s, %s, %s)
                    """,
                    (
                        owner_reference,
                        idempotency_key,
                        request_hash,
                        conversation_id,
                    ),
                )

                return _conversation_from_row(row)

    def claim_request(
        self,
        conversation_id: UUID,
        request_id: UUID,
        request_hash: str,
    ) -> ConversationRequestClaim:
        """Claim one conversation turn or return its durable cached response."""
        if not request_hash.strip():
            raise ValueError("request_hash cannot be empty.")

        with self._connection.transaction():
            with self._connection.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    """
                    INSERT INTO conversation_request_results (
                        conversation_id, request_id, request_hash
                    )
                    VALUES (%s, %s, %s)
                    ON CONFLICT (conversation_id, request_id) DO NOTHING
                    RETURNING request_hash, result
                    """,
                    (conversation_id, request_id, request_hash),
                )
                row = cursor.fetchone()
                if row is not None:
                    return ConversationRequestClaim(created=True, result=None)

                cursor.execute(
                    """
                    SELECT request_hash, result
                    FROM conversation_request_results
                    WHERE conversation_id = %s AND request_id = %s
                    """,
                    (conversation_id, request_id),
                )
                row = cursor.fetchone()

        if row is None:
            raise RuntimeError("PostgreSQL did not return the idempotency record.")
        if row["request_hash"] != request_hash:
            raise IdempotencyKeyConflict("The request ID was reused with different content.")
        result = row["result"]
        if result is not None and not isinstance(result, dict):
            raise RuntimeError("PostgreSQL returned an invalid idempotency result.")
        return ConversationRequestClaim(created=False, result=result)

    def complete_request(
        self,
        conversation_id: UUID,
        request_id: UUID,
        result: dict[str, Any],
    ) -> None:
        """Persist a sanitized result independently of bounded message history."""
        with self._connection.transaction():
            with self._connection.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE conversation_request_results
                    SET result = %s, completed_at = statement_timestamp()
                    WHERE conversation_id = %s
                      AND request_id = %s
                    """,
                    (Jsonb(result), conversation_id, request_id),
                )
                if cursor.rowcount == 1:
                    return

                cursor.execute(
                    """
                    SELECT result
                    FROM conversation_request_results
                    WHERE conversation_id = %s AND request_id = %s
                    """,
                    (conversation_id, request_id),
                )
                row = cursor.fetchone()

        if row is None or row[0] is None:
            raise RuntimeError("The conversation request claim disappeared before completion.")

    def get(self, conversation_id: UUID) -> Conversation | None:
        with self._connection.transaction():
            with self._connection.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    """
                    SELECT id, thread_id, created_at, updated_at,
                           owner_reference, creation_request_id
                    FROM conversations
                    WHERE id = %s
                    """,
                    (conversation_id,),
                )
                row = cursor.fetchone()

        return _conversation_from_row(row) if row is not None else None

    def get_for_owner(
        self,
        conversation_id: UUID,
        owner_reference: str,
    ) -> Conversation | None:
        """Return a conversation only when it belongs to the authenticated owner."""
        with self._connection.transaction():
            with self._connection.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    """
                    SELECT id, thread_id, created_at, updated_at,
                           owner_reference, creation_request_id
                    FROM conversations
                    WHERE id = %s AND owner_reference = %s
                    """,
                    (conversation_id, owner_reference),
                )
                row = cursor.fetchone()

        return _conversation_from_row(row) if row is not None else None

    def get_by_thread_id(self, thread_id: str) -> Conversation | None:
        with self._connection.transaction():
            with self._connection.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    """
                    SELECT id, thread_id, created_at, updated_at,
                           owner_reference, creation_request_id
                    FROM conversations
                    WHERE thread_id = %s
                    """,
                    (thread_id,),
                )
                row = cursor.fetchone()

        return _conversation_from_row(row) if row is not None else None

    def touch(self, conversation_id: UUID) -> Conversation | None:
        """Refresh updated_at using the database trigger."""
        with self._connection.transaction():
            with self._connection.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    """
                    UPDATE conversations
                    SET updated_at = updated_at
                    WHERE id = %s
                    RETURNING id, thread_id, created_at, updated_at,
                              owner_reference, creation_request_id
                    """,
                    (conversation_id,),
                )
                row = cursor.fetchone()

        return _conversation_from_row(row) if row is not None else None

    def touch_for_owner(
        self,
        conversation_id: UUID,
        owner_reference: str,
    ) -> Conversation | None:
        """Touch activity only when the conversation still belongs to the owner."""
        with self._connection.transaction():
            with self._connection.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    """
                    UPDATE conversations
                    SET updated_at = updated_at
                    WHERE id = %s AND owner_reference = %s
                    RETURNING id, thread_id, created_at, updated_at,
                              owner_reference, creation_request_id
                    """,
                    (conversation_id, owner_reference),
                )
                row = cursor.fetchone()

        return _conversation_from_row(row) if row is not None else None
