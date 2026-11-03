"""Parameterized PostgreSQL operations for appointment bookings."""

from __future__ import annotations

from typing import Any
from uuid import UUID

import psycopg
from psycopg import Connection
from psycopg.rows import dict_row

from app.repositories.errors import BookingConflict, SlotUnavailable
from app.repositories.models import Booking, BookingStatus


def _booking_from_row(row: dict[str, Any]) -> Booking:
    return Booking(
        id=row["id"],
        slot_id=row["slot_id"],
        conversation_id=row["conversation_id"],
        patient_reference=row["patient_reference"],
        status=row["status"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


class BookingRepository:
    """Persist bookings using database locks and constraints for safety."""

    def __init__(self, connection: Connection[Any]) -> None:
        self._connection = connection

    def create(
        self,
        slot_id: UUID,
        patient_reference: str,
        *,
        conversation_id: UUID | None = None,
        requires_approval: bool = False,
    ) -> Booking:
        """Create a confirmed booking or one requiring approval."""
        if not patient_reference.strip():
            raise ValueError("patient_reference cannot be empty.")

        if requires_approval and conversation_id is None:
            raise ValueError("A conversation_id is required for approval-based bookings.")

        status: BookingStatus = "pending_approval" if requires_approval else "confirmed"

        try:
            with self._connection.transaction():
                with self._connection.cursor(row_factory=dict_row) as cursor:
                    cursor.execute(
                        """
                        INSERT INTO bookings (
                            slot_id, conversation_id, patient_reference, status
                        )
                        VALUES (%s, %s, %s, %s)
                        RETURNING id, slot_id, conversation_id, patient_reference,
                                  status, created_at, updated_at
                        """,
                        (slot_id, conversation_id, patient_reference, status),
                    )
                    row = cursor.fetchone()

        except psycopg.errors.UniqueViolation as exc:
            raise BookingConflict("The appointment slot already has an active booking.") from exc

        except psycopg.errors.CheckViolation as exc:
            raise SlotUnavailable(
                "Bookings require an open slot belonging to an active doctor."
            ) from exc

        if row is None:
            raise RuntimeError("PostgreSQL did not return the inserted booking.")

        return _booking_from_row(row)

    def get(self, booking_id: UUID) -> Booking | None:
        with self._connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """
                SELECT id, slot_id, conversation_id, patient_reference,
                       status, created_at, updated_at
                FROM bookings
                WHERE id = %s
                """,
                (booking_id,),
            )
            row = cursor.fetchone()

        return _booking_from_row(row) if row is not None else None

    def list_for_conversation(self, conversation_id: UUID) -> list[Booking]:
        with self._connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """
                SELECT id, slot_id, conversation_id, patient_reference,
                       status, created_at, updated_at
                FROM bookings
                WHERE conversation_id = %s
                ORDER BY created_at, id
                """,
                (conversation_id,),
            )
            return [_booking_from_row(row) for row in cursor.fetchall()]
