"""Parameterized PostgreSQL operations for appointment bookings."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID
from zoneinfo import ZoneInfo

import psycopg
from psycopg import Connection
from psycopg.rows import dict_row

from app.repositories._locks import lock_slot
from app.repositories.approvals import ApprovalRepository
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
    """Create bookings and their required approvals in one transaction."""

    def __init__(
        self,
        connection: Connection[Any],
        *,
        clinic_timezone: str = "Asia/Muscat",
    ) -> None:
        self._connection = connection
        self._timezone = ZoneInfo(clinic_timezone)

    def create(
        self,
        slot_id: UUID,
        patient_reference: str,
        *,
        conversation_id: UUID | None = None,
        requires_approval: bool = False,
    ) -> Booking:
        """Same-day bookings always require staff approval."""
        patient_reference = patient_reference.strip()

        if not patient_reference:
            raise ValueError("patient_reference cannot be empty.")

        try:
            with self._connection.transaction():
                with self._connection.cursor(row_factory=dict_row) as cursor:
                    slot = lock_slot(cursor, slot_id)

                    if slot["status"] != "open" or not slot["doctor_is_active"]:
                        raise SlotUnavailable("The slot is blocked or its doctor is inactive.")

                    # Read the current time after obtaining the locks.
                    cursor.execute("SELECT clock_timestamp() AS now")
                    clock = cursor.fetchone()

                    if clock is None:
                        raise RuntimeError("PostgreSQL did not return the current time.")

                    starts_at: datetime = slot["starts_at"]
                    now: datetime = clock["now"]

                    same_day = (
                        starts_at.astimezone(self._timezone).date()
                        == now.astimezone(self._timezone).date()
                    )

                    needs_approval = same_day or requires_approval

                    if needs_approval and conversation_id is None:
                        raise ValueError(
                            "A conversation_id is required for approval-based bookings."
                        )

                    status: BookingStatus = "pending_approval" if needs_approval else "confirmed"

                    cursor.execute(
                        """
                        INSERT INTO bookings (
                            slot_id,
                            conversation_id,
                            patient_reference,
                            status
                        )
                        VALUES (%s, %s, %s, %s)
                        RETURNING id, slot_id, conversation_id,
                                  patient_reference, status,
                                  created_at, updated_at
                        """,
                        (
                            slot_id,
                            conversation_id,
                            patient_reference,
                            status,
                        ),
                    )
                    row = cursor.fetchone()

                    if row is None:
                        raise RuntimeError("PostgreSQL did not return the inserted booking.")

                    booking = _booking_from_row(row)

                if needs_approval:
                    # Use the same connection and outer transaction.
                    assert conversation_id is not None

                    ApprovalRepository(self._connection).create(
                        booking.id,
                        conversation_id,
                        "same_day_booking",
                        request_id=booking.id,
                    )

                return booking

        except psycopg.errors.UniqueViolation as exc:
            raise BookingConflict("The appointment slot already has an active booking.") from exc

        except psycopg.errors.CheckViolation as exc:
            raise SlotUnavailable("Bookings require an open slot and an active doctor.") from exc

    def get(self, booking_id: UUID) -> Booking | None:
        with self._connection.transaction():
            with self._connection.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    "SELECT * FROM bookings WHERE id = %s",
                    (booking_id,),
                )
                row = cursor.fetchone()

                return _booking_from_row(row) if row is not None else None

    def list_for_conversation(self, conversation_id: UUID) -> list[Booking]:
        with self._connection.transaction():
            with self._connection.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    """
                    SELECT *
                    
                    FROM bookings
                    WHERE conversation_id = %s
                    ORDER BY created_at, id
                    """,
                    (conversation_id,),
                )

                return [_booking_from_row(row) for row in cursor.fetchall()]
