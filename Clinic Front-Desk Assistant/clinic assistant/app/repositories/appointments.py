"""Parameterized PostgreSQL operations for appointment slots."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

import psycopg
from psycopg import Connection
from psycopg.rows import dict_row

from app.repositories.errors import (
    AppointmentSlotConflict,
    DoctorNotFound,
    InvalidAppointmentSlot,
)
from app.repositories.models import (
    AppointmentSlot,
    AvailableAppointmentSlot,
    SlotStatus,
)

MAX_SEARCH_LIMIT = 100


def _slot_from_row(row: dict[str, Any]) -> AppointmentSlot:
    """Convert a database row into an AppointmentSlot model."""
    return AppointmentSlot(
        id=row["id"],
        doctor_id=row["doctor_id"],
        starts_at=row["starts_at"],
        ends_at=row["ends_at"],
        status=row["status"],
        created_at=row["created_at"],
    )


class AppointmentRepository:
    """Search and maintain appointment slots."""

    def __init__(self, connection: Connection[Any]) -> None:
        self._connection = connection

    def search_open_slots(
        self,
        starts_at: datetime,
        ends_at: datetime,
        *,
        doctor_id: UUID | None = None,
        limit: int = 25,
    ) -> list[AvailableAppointmentSlot]:
        """List open, unreserved slots for active doctors.

        The maximum number of returned slots is 100.
        """
        _validate_aware_time(starts_at, "starts_at")
        _validate_aware_time(ends_at, "ends_at")

        if ends_at <= starts_at:
            raise ValueError("ends_at must be later than starts_at.")

        if not 1 <= limit <= MAX_SEARCH_LIMIT:
            raise ValueError(f"limit must be between 1 and {MAX_SEARCH_LIMIT}.")

        query = """
            SELECT slot.id, slot.doctor_id,
                   doctor.full_name AS doctor_name,
                   doctor.specialty AS doctor_specialty,
                   slot.starts_at, slot.ends_at
            FROM appointment_slots AS slot
            JOIN doctors AS doctor ON doctor.id = slot.doctor_id
            WHERE slot.status = 'open'
              AND doctor.is_active = TRUE
              AND slot.starts_at >= %s
              AND slot.starts_at < %s
              AND NOT EXISTS (
                  SELECT 1
                  FROM bookings AS booking
                  WHERE booking.slot_id = slot.id
                    AND booking.status IN ('pending_approval', 'confirmed')
              )
        """

        parameters: list[Any] = [starts_at, ends_at]

        if doctor_id is not None:
            query += " AND slot.doctor_id = %s"
            parameters.append(doctor_id)

        query += " ORDER BY slot.starts_at, doctor.full_name LIMIT %s"
        parameters.append(limit)

        with self._connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(query, parameters)
            return [AvailableAppointmentSlot(**row) for row in cursor.fetchall()]

    def get_slot(self, slot_id: UUID) -> AppointmentSlot | None:
        """Retrieve an appointment slot by ID."""
        with self._connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """
                SELECT id, doctor_id, starts_at, ends_at, status, created_at
                FROM appointment_slots
                WHERE id = %s
                """,
                (slot_id,),
            )
            row = cursor.fetchone()

        return _slot_from_row(row) if row is not None else None

    def create_slot(
        self,
        doctor_id: UUID,
        starts_at: datetime,
        ends_at: datetime,
        *,
        status: SlotStatus = "open",
    ) -> AppointmentSlot:
        """Create a slot; PostgreSQL rejects invalid or overlapping times."""
        _validate_aware_time(starts_at, "starts_at")
        _validate_aware_time(ends_at, "ends_at")

        if ends_at <= starts_at:
            raise ValueError("ends_at must be later than starts_at.")

        if status not in {"open", "blocked"}:
            raise ValueError("status must be 'open' or 'blocked'.")

        try:
            with self._connection.transaction():
                with self._connection.cursor(row_factory=dict_row) as cursor:
                    cursor.execute(
                        """
                        INSERT INTO appointment_slots (
                            doctor_id, starts_at, ends_at, status
                        )
                        VALUES (%s, %s, %s, %s)
                        RETURNING id, doctor_id, starts_at, ends_at,
                                  status, created_at
                        """,
                        (doctor_id, starts_at, ends_at, status),
                    )
                    row = cursor.fetchone()

        except psycopg.errors.ExclusionViolation as exc:
            raise AppointmentSlotConflict(
                "The doctor already has a slot overlapping this time."
            ) from exc

        except psycopg.errors.ForeignKeyViolation as exc:
            raise DoctorNotFound("The selected doctor does not exist.") from exc

        except psycopg.errors.CheckViolation as exc:
            raise InvalidAppointmentSlot(
                "The appointment slot violates a database constraint."
            ) from exc

        if row is None:
            raise RuntimeError("PostgreSQL did not return the inserted appointment slot.")

        return _slot_from_row(row)


def _validate_aware_time(value: datetime, field_name: str) -> None:
    """Ensure datetime values include timezone information."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must include a timezone.")
