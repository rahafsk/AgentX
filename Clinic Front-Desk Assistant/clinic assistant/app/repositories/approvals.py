"""Parameterized PostgreSQL operations for staff approval workflows."""

from __future__ import annotations

from typing import Any, Literal
from uuid import UUID, uuid4

import psycopg
from psycopg import Connection, Cursor
from psycopg.rows import dict_row

from app.repositories._locks import lock_slot
from app.repositories.errors import (
    ApprovalAlreadyDecided,
    ApprovalAlreadyPending,
    ApprovalContextMismatch,
    ApprovalNotFound,
    ApprovalStateConflict,
)
from app.repositories.models import Approval, ApprovalAction, BookingStatus

ApprovalDecision = Literal["approved", "rejected"]


def _approval_from_row(row: dict[str, Any]) -> Approval:
    return Approval(
        id=row["id"],
        booking_id=row["booking_id"],
        conversation_id=row["conversation_id"],
        action=row["action"],
        status=row["status"],
        requested_at=row["requested_at"],
        decided_at=row["decided_at"],
        decided_by=row["decided_by"],
        decision_note=row["decision_note"],
    )


def _lock_booking(
    cursor: Cursor[dict[str, Any]],
    booking_id: UUID,
) -> dict[str, Any]:
    """Lock doctor, slot, then booking."""
    cursor.execute(
        "SELECT slot_id FROM bookings WHERE id = %s",
        (booking_id,),
    )
    reference = cursor.fetchone()

    if reference is None:
        raise ApprovalContextMismatch("The booking does not exist.")

    slot_id = reference["slot_id"]

    # This locks the doctor first, then the slot.
    lock_slot(cursor, slot_id)

    cursor.execute(
        """
        SELECT id, slot_id, conversation_id, status
        FROM bookings
        WHERE id = %s
        FOR UPDATE
        """,
        (booking_id,),
    )
    booking = cursor.fetchone()

    if booking is None or booking["slot_id"] != slot_id:
        raise ApprovalContextMismatch("The booking changed. Retry the request.")

    return booking


class ApprovalRepository:
    """Create, list, and decide durable staff approvals."""

    def __init__(self, connection: Connection[Any]) -> None:
        self._connection = connection

    def create(
        self,
        booking_id: UUID,
        conversation_id: UUID,
        action: ApprovalAction,
        *,
        request_id: UUID | None = None,
    ) -> Approval:
        """Create a pending approval using the shared locking order."""
        if action not in {"same_day_booking", "cancel_booking"}:
            raise ValueError("action must be 'same_day_booking' or 'cancel_booking'.")

        approval_id = request_id if request_id is not None else uuid4()

        if request_id is not None:
            existing = self.get(approval_id)
            if existing is not None:
                return self._reuse_idempotent_approval(
                    existing,
                    booking_id=booking_id,
                    conversation_id=conversation_id,
                    action=action,
                )

        try:
            with self._connection.transaction():
                with self._connection.cursor(row_factory=dict_row) as cursor:
                    booking_row = _lock_booking(cursor, booking_id)

                    if booking_row["conversation_id"] != conversation_id:
                        raise ApprovalContextMismatch(
                            "The approval must reference the booking's conversation."
                        )

                    expected_status = (
                        "pending_approval" if action == "same_day_booking" else "confirmed"
                    )

                    if booking_row["status"] != expected_status:
                        raise ApprovalStateConflict(
                            f"A {action} approval requires a booking in {expected_status!r} state."
                        )

                    cursor.execute(
                        """
                        INSERT INTO approvals (
                            id, booking_id, conversation_id, action
                        )
                        VALUES (%s, %s, %s, %s)
                        RETURNING id, booking_id, conversation_id, action, status,
                                  requested_at, decided_at, decided_by, decision_note
                        """,
                        (
                            approval_id,
                            booking_id,
                            conversation_id,
                            action,
                        ),
                    )
                    row = cursor.fetchone()

                    if row is None:
                        raise RuntimeError("PostgreSQL did not return the inserted approval.")

                    return _approval_from_row(row)
        except psycopg.errors.ForeignKeyViolation as exc:
            raise ApprovalContextMismatch(
                "The approval must reference the booking's conversation."
            ) from exc
        except psycopg.errors.UniqueViolation as exc:
            if request_id is not None:
                existing = self.get(approval_id)
                if existing is not None:
                    return self._reuse_idempotent_approval(
                        existing,
                        booking_id=booking_id,
                        conversation_id=conversation_id,
                        action=action,
                    )
            raise ApprovalAlreadyPending(
                "An approval for this booking action is already pending."
            ) from exc

    @staticmethod
    def _reuse_idempotent_approval(
        approval: Approval,
        *,
        booking_id: UUID,
        conversation_id: UUID,
        action: ApprovalAction,
    ) -> Approval:
        """Return the previous result only when its key matches the same action."""
        if (
            approval.booking_id != booking_id
            or approval.conversation_id != conversation_id
            or approval.action != action
        ):
            raise ApprovalContextMismatch("The request ID was already used for another approval.")
        return approval

    def get(self, approval_id: UUID) -> Approval | None:
        with self._connection.transaction():
            with self._connection.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    """
                    SELECT id, booking_id, conversation_id, action, status,
                           requested_at, decided_at, decided_by, decision_note
                    FROM approvals
                    WHERE id = %s
                    """,
                    (approval_id,),
                )
                row = cursor.fetchone()

                return _approval_from_row(row) if row is not None else None

    def list_pending(self, *, limit: int = 50) -> list[Approval]:
        if limit < 1:
            raise ValueError("limit must be greater than zero.")

        with self._connection.transaction():
            with self._connection.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    """
                    SELECT id, booking_id, conversation_id, action, status,
                           requested_at, decided_at, decided_by, decision_note
                    FROM approvals
                    WHERE status = 'pending'
                    ORDER BY requested_at, id
                    LIMIT %s
                    """,
                    (limit,),
                )

                return [_approval_from_row(row) for row in cursor.fetchall()]

    def decide(
        self,
        approval_id: UUID,
        decision: ApprovalDecision,
        *,
        decided_by: str,
        decision_note: str | None = None,
    ) -> Approval:
        """Lock doctor, slot, booking, then approval."""
        if decision not in {"approved", "rejected"}:
            raise ValueError("decision must be 'approved' or 'rejected'.")

        decided_by = decided_by.strip()
        if not decided_by:
            raise ValueError("decided_by cannot be empty.")

        if not decision_note or not decision_note.strip():
            raise ValueError("A decision reason is required.")

        decision_note = decision_note.strip()

        with self._connection.transaction():
            with self._connection.cursor(row_factory=dict_row) as cursor:
                # Read the booking ID without locking the approval.
                cursor.execute(
                    """
                    SELECT booking_id
                    FROM approvals
                    WHERE id = %s
                    """,
                    (approval_id,),
                )
                approval_reference = cursor.fetchone()

                if approval_reference is None:
                    raise ApprovalNotFound("The approval does not exist.")

                booking_id = approval_reference["booking_id"]

                _lock_booking(cursor, booking_id)

                # Lock and read the approval again after obtaining the booking lock.
                cursor.execute(
                    """
                    SELECT id, booking_id, conversation_id, action, status,
                           requested_at, decided_at, decided_by, decision_note
                    FROM approvals
                    WHERE id = %s
                    FOR UPDATE
                    """,
                    (approval_id,),
                )
                approval_row = cursor.fetchone()

                if approval_row is None:
                    raise ApprovalNotFound("The approval does not exist.")

                if approval_row["booking_id"] != booking_id:
                    raise ApprovalContextMismatch(
                        "The approval's booking changed. Retry the request."
                    )

                if approval_row["status"] == decision:
                    # A staff client may retry after the database commit but
                    # before the checkpointed conversation resume completed.
                    return _approval_from_row(approval_row)

                if approval_row["status"] != "pending":
                    raise ApprovalAlreadyDecided("The approval has already been decided.")

                action: ApprovalAction = approval_row["action"]
                booking_status: BookingStatus
                expected_booking_status: BookingStatus

                if action == "same_day_booking":
                    expected_booking_status = "pending_approval"
                    booking_status = "confirmed" if decision == "approved" else "rejected"
                elif action == "cancel_booking":
                    expected_booking_status = "confirmed"
                    booking_status = "cancelled" if decision == "approved" else "confirmed"
                else:
                    raise ApprovalStateConflict("The approval action is not supported.")

                cursor.execute(
                    """
                    UPDATE bookings
                    SET status = %s
                    WHERE id = %s AND status = %s
                    RETURNING id
                    """,
                    (
                        booking_status,
                        booking_id,
                        expected_booking_status,
                    ),
                )

                if cursor.fetchone() is None:
                    raise ApprovalStateConflict(
                        "The booking is not in the state required by this approval."
                    )

                cursor.execute(
                    """
                    UPDATE approvals
                    SET status = %s,
                        decided_at = statement_timestamp(),
                        decided_by = %s,
                        decision_note = %s
                    WHERE id = %s
                    RETURNING id, booking_id, conversation_id, action, status,
                              requested_at, decided_at, decided_by, decision_note
                    """,
                    (decision, decided_by, decision_note, approval_id),
                )
                row = cursor.fetchone()

                if row is None:
                    raise RuntimeError("PostgreSQL did not return the updated approval.")

                return _approval_from_row(row)
