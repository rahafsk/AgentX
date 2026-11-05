"""PostgreSQL repository for conversation metadata."""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

from psycopg import Connection
from psycopg.rows import dict_row

from app.repositories.models import Conversation


def _conversation_from_row(
    row: dict[str, Any],
) -> Conversation:
    return Conversation(
        id=row["id"],
        thread_id=row["thread_id"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        owner_reference=row["owner_reference"],
    )


class ConversationRepository:
    """Persist conversation metadata; LangGraph stores the checkpoint state.

    Each method uses a short transaction scope. Callers must never hold an
    outer database transaction open while awaiting a LangGraph invocation.
    """

    def __init__(self, connection: Connection[Any]) -> None:
        self._connection = connection

    def create(self, owner_reference: str) -> Conversation:
        """Create a conversation whose UUID string is also its LangGraph thread ID."""
        owner_reference = owner_reference.strip()
        if not owner_reference:
            raise ValueError("owner_reference cannot be empty.")

        conversation_id = uuid4()
        thread_id = str(conversation_id)

        with self._connection.transaction():
            with self._connection.cursor(row_factory=dict_row) as cursor:
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

        return _conversation_from_row(row)

    def get(self, conversation_id: UUID) -> Conversation | None:
        with self._connection.transaction():
            with self._connection.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    """
                    SELECT id, thread_id, created_at, updated_at, owner_reference
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
                    SELECT id, thread_id, created_at, updated_at, owner_reference
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
                    SELECT id, thread_id, created_at, updated_at, owner_reference
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
                    RETURNING id, thread_id, created_at, updated_at, owner_reference
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
                    RETURNING id, thread_id, created_at, updated_at, owner_reference
                    """,
                    (conversation_id, owner_reference),
                )
                row = cursor.fetchone()

        return _conversation_from_row(row) if row is not None else None
