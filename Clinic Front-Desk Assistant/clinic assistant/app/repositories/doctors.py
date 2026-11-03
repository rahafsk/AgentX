
"""Parameterized PostgreSQL reads for doctors."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from psycopg import Connection
from psycopg.rows import dict_row

from app.repositories.models import Doctor


class DoctorRepository:
    """Read doctor records using a caller-owned psycopg connection."""

    def __init__(self, connection: Connection[Any]) -> None:
        self._connection = connection

    def get(self, doctor_id: UUID) -> Doctor | None:
        with self._connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """
                SELECT id, full_name, specialty, is_active, created_at
                FROM doctors
                WHERE id = %s
                """,
                (doctor_id,),
            )
            row = cursor.fetchone()

        return Doctor(**row) if row is not None else None

    def list_active(self) -> list[Doctor]:
        with self._connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """
                SELECT id, full_name, specialty, is_active, created_at
                FROM doctors
                WHERE is_active = TRUE
                ORDER BY full_name ASC, id ASC
                """
            )
            rows = cursor.fetchall()

        return [Doctor(**row) for row in rows]
