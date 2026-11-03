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
    )


class ConversationRepository:
    """Persist conversation metadata; LangGraph stores the checkpoint state."""

    def __init__(self, connection: Connection[Any]) -> None:
        self._connection = connection

    def create(self) -> Conversation:
        """Create a conversation whose UUID string is also its LangGraph thread ID."""
        conversation_id = uuid4()
        thread_id = str(conversation_id)

        with self._connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """
                INSERT INTO conversations (id, thread_id)
                VALUES (%s, %s)
                RETURNING id, thread_id, created_at, updated_at
                """,
                (conversation_id, thread_id),
            )
            row = cursor.fetchone()

        if row is None:
            raise RuntimeError(
                "PostgreSQL did not return the inserted conversation."
            )

        return _conversation_from_row(row)

    def get(self, conversation_id: UUID) -> Conversation | None:
        with self._connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """
                SELECT id, thread_id, created_at, updated_at
                FROM conversations
                WHERE id = %s
                """,
                (conversation_id,),
            )
            row = cursor.fetchone()

        return _conversation_from_row(row) if row is not None else None

    def get_by_thread_id(self, thread_id: str) -> Conversation | None:
        with self._connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """
                SELECT id, thread_id, created_at, updated_at
                FROM conversations
                WHERE thread_id = %s
                """,
                (thread_id,),
            )
            row = cursor.fetchone()

        return _conversation_from_row(row) if row is not None else None

    def touch(self, conversation_id: UUID) -> Conversation | None:
        """Refresh updated_at when conversation activity resumes."""
        with self._connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """
                UPDATE conversations
                SET updated_at = statement_timestamp()
                WHERE id = %s
                RETURNING id, thread_id, created_at, updated_at
                """,
                (conversation_id,),
            )
            row = cursor.fetchone()

        return _conversation_from_row(row) if row is not None else None
