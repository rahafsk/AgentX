"""PostgreSQL tests for booking approval behavior."""

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo

import pytest
from psycopg import Connection

from app.repositories import (
    AppointmentRepository,
    ApprovalRepository,
    BookingRepository,
)


def test_same_day_booking_automatically_requires_approval(
    database_connection: Connection[Any],
) -> None:
    doctor_id = uuid4()
    conversation_id = uuid4()
    timezone = ZoneInfo("Asia/Muscat")

    now = datetime.now(UTC)
    start = now + timedelta(minutes=5)

    if start.astimezone(timezone).date() != now.astimezone(timezone).date():
        pytest.skip("The test slot would fall on the next clinic day.")

    database_connection.execute(
        """
        INSERT INTO doctors (id, full_name, specialty)
        VALUES (%s, %s, %s)
        """,
        (doctor_id, "Fictional Booking Test Doctor", "General"),
    )

    database_connection.execute(
        """
        INSERT INTO conversations (id, thread_id)
        VALUES (%s, %s)
        """,
        (conversation_id, str(conversation_id)),
    )

    slot = AppointmentRepository(database_connection).create_slot(
        doctor_id,
        start,
        start + timedelta(minutes=30),
    )

    # Omit requires_approval to check automatic enforcement.
    booking = BookingRepository(
        database_connection,
        clinic_timezone="Asia/Muscat",
    ).create(
        slot.id,
        "fictional-test-patient",
        conversation_id=conversation_id,
    )

    assert booking.status == "pending_approval"

    approval = ApprovalRepository(database_connection).get(booking.id)
    assert approval is not None
    assert approval.booking_id == booking.id
    assert approval.conversation_id == conversation_id
    assert approval.action == "same_day_booking"
    assert approval.status == "pending"
