from typing import Any
from uuid import UUID

import psycopg
from psycopg.rows import dict_row

from app.config import get_settings


class ApprovalNotFoundError(Exception):
    """The approval does not exist."""


class ApprovalConflictError(Exception):
    """The approval already has a decision."""


def get_database_url() -> str:
    """Read the existing database connection setting."""

    settings = get_settings()
    return settings.database_url.get_secret_value()


def list_pending() -> list[dict[str, Any]]:
    """Return pending approvals with their conversation thread IDs."""

    with psycopg.connect(
        get_database_url(),
        row_factory=dict_row,
        connect_timeout=5,
        options="-c statement_timeout=5000",
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT approvals.*, conversations.thread_id
                FROM approvals
                JOIN conversations
                    ON conversations.id = approvals.conversation_id
                WHERE approvals.status = %s
                ORDER BY approvals.requested_at, approvals.id
                """,
                ("pending",),
            )

            return cursor.fetchall()


def get_approval(approval_id: UUID) -> dict[str, Any]:
    """Return one approval or raise an error if it does not exist."""

    with psycopg.connect(
        get_database_url(),
        row_factory=dict_row,
        connect_timeout=5,
        options="-c statement_timeout=5000",
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT approvals.*, conversations.thread_id
                FROM approvals
                JOIN conversations
                    ON conversations.id = approvals.conversation_id
                WHERE approvals.id = %s
                """,
                (approval_id,),
            )

            approval = cursor.fetchone()

            if approval is None:
                raise ApprovalNotFoundError("Approval not found.")

            return approval


def record_decision(
    approval_id: UUID,
    status: str,
    decided_by: str,
    decision_note: str,
) -> dict[str, Any]:
    """Record a decision only if the approval is still pending."""

    if status not in ["approved", "rejected"]:
        raise ValueError("Status must be approved or rejected.")

    decided_by = decided_by.strip()
    decision_note = decision_note.strip()

    if not decided_by:
        raise ValueError("The staff identifier is required.")

    if not decision_note:
        raise ValueError("A decision reason is required.")

    with psycopg.connect(
        get_database_url(),
        row_factory=dict_row,
        connect_timeout=5,
        options="-c statement_timeout=5000",
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE approvals
                SET status = %s,
                    decided_at = CURRENT_TIMESTAMP,
                    decided_by = %s,
                    decision_note = %s
                FROM conversations
                WHERE approvals.id = %s
                AND approvals.status = 'pending'
                AND conversations.id = approvals.conversation_id
                RETURNING approvals.*, conversations.thread_id
                """,
                (
                    status,
                    decided_by,
                    decision_note,
                    approval_id,
                ),
            )

            approval = cursor.fetchone()

            if approval is not None:
                return approval

            cursor.execute(
                "SELECT id FROM approvals WHERE id = %s",
                (approval_id,),
            )

            if cursor.fetchone() is None:
                raise ApprovalNotFoundError("Approval not found.")

            raise ApprovalConflictError(
                "This approval already has a decision."
            )