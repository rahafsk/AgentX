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
    ApprovalWorkflowRequired,
    BookingRepository,
    SlotUnavailable,
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
    repository = BookingRepository(database_connection, clinic_timezone="Asia/Muscat")
    request_id = uuid4()
    booking = repository.create(
        slot.id,
        "fictional-test-patient",
        conversation_id=conversation_id,
        approval_workflow=True,
        request_id=request_id,
    )
    replay = repository.create(
        slot.id,
        "fictional-test-patient",
        conversation_id=conversation_id,
        approval_workflow=True,
        request_id=request_id,
    )

    assert booking.status == "pending_approval"
    assert replay == booking

    approval = ApprovalRepository(database_connection).get(booking.id)
    assert approval is not None
    assert approval.booking_id == booking.id
    assert approval.conversation_id == conversation_id
    assert approval.action == "same_day_booking"
    assert approval.status == "pending"


def test_same_day_booking_without_workflow_is_rejected(
    database_connection: Connection[Any],
) -> None:
    doctor_id = uuid4()
    conversation_id = uuid4()
    start = datetime.now(UTC) + timedelta(minutes=5)

    database_connection.execute(
        "INSERT INTO doctors (id, full_name, specialty) VALUES (%s, %s, %s)",
        (doctor_id, "Fictional No-Workflow Doctor", "General"),
    )
    database_connection.execute(
        "INSERT INTO conversations (id, thread_id) VALUES (%s, %s)",
        (conversation_id, str(conversation_id)),
    )
    slot = AppointmentRepository(database_connection).create_slot(
        doctor_id,
        start,
        start + timedelta(minutes=30),
    )

    with pytest.raises(ApprovalWorkflowRequired):
        BookingRepository(database_connection).create(
            slot.id,
            "fictional-test-patient",
            conversation_id=conversation_id,
        )


def test_elapsed_slot_cannot_be_booked(database_connection: Connection[Any]) -> None:
    doctor_id = uuid4()
    start = datetime.now(UTC) - timedelta(days=1)
    database_connection.execute(
        "INSERT INTO doctors (id, full_name, specialty) VALUES (%s, %s, %s)",
        (doctor_id, "Fictional Past-Slot Doctor", "General"),
    )
    slot = AppointmentRepository(database_connection).create_slot(
        doctor_id,
        start,
        start + timedelta(minutes=30),
    )

    with pytest.raises(SlotUnavailable, match="already started"):
        BookingRepository(database_connection).create(slot.id, "fictional-test-patient")


def test_booking_request_id_replays_the_original_booking(
    database_connection: Connection[Any],
) -> None:
    doctor_id = uuid4()
    start = datetime.now(UTC) + timedelta(days=2)
    database_connection.execute(
        "INSERT INTO doctors (id, full_name, specialty) VALUES (%s, %s, %s)",
        (doctor_id, "Fictional Idempotency Doctor", "General"),
    )
    slot = AppointmentRepository(database_connection).create_slot(
        doctor_id,
        start,
        start + timedelta(minutes=30),
    )
    request_id = uuid4()
    repository = BookingRepository(database_connection)

    booking = repository.create(
        slot.id,
        "fictional-test-patient",
        request_id=request_id,
    )
    replay = repository.create(
        slot.id,
        "fictional-test-patient",
        request_id=request_id,
    )

    assert booking.id == request_id
    assert replay == booking
