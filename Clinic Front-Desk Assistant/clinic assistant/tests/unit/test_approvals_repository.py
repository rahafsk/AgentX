"""PostgreSQL tests for the approval repository."""

from typing import Any
from uuid import UUID, uuid4

import pytest
from psycopg import Connection

from app.repositories.approvals import ApprovalRepository
from app.repositories.errors import (
    ApprovalAlreadyDecided,
    ApprovalAlreadyPending,
    ApprovalNotFound,
    ApprovalStateConflict,
)
from app.repositories.models import ApprovalAction, BookingStatus


@pytest.fixture
def booking_data(
    database_connection: Connection[Any],
) -> tuple[UUID, UUID]:
    """Create a fictional doctor, slot, conversation, and booking."""
    doctor_id = uuid4()
    slot_id = uuid4()
    conversation_id = uuid4()
    booking_id = uuid4()

    database_connection.execute(
        """
        INSERT INTO doctors (id, full_name, specialty)
        VALUES (%s, %s, %s)
        """,
        (doctor_id, "Test Doctor", "General"),
    )

    database_connection.execute(
        """
        INSERT INTO appointment_slots (
            id, doctor_id, starts_at, ends_at
        )
        VALUES (
            %s, %s,
            CURRENT_TIMESTAMP + INTERVAL '1 day',
            CURRENT_TIMESTAMP + INTERVAL '1 day 30 minutes'
        )
        """,
        (slot_id, doctor_id),
    )

    database_connection.execute(
        """
        INSERT INTO conversations (id, thread_id)
        VALUES (%s, %s)
        """,
        (conversation_id, f"test-{conversation_id}"),
    )

    database_connection.execute(
        """
        INSERT INTO bookings (
            id, slot_id, conversation_id, patient_reference, status
        )
        VALUES (%s, %s, %s, %s, %s)
        """,
        (
            booking_id,
            slot_id,
            conversation_id,
            "test-patient",
            "pending_approval",
        ),
    )

    return booking_id, conversation_id


def test_create_get_and_list_pending(
    database_connection: Connection[Any],
    booking_data: tuple[UUID, UUID],
) -> None:
    repository = ApprovalRepository(database_connection)
    booking_id, conversation_id = booking_data

    approval = repository.create(
        booking_id,
        conversation_id,
        "same_day_booking",
    )

    assert approval.status == "pending"
    assert approval.booking_id == booking_id
    assert approval.conversation_id == conversation_id

    saved = repository.get(approval.id)
    assert saved is not None
    assert saved.id == approval.id

    pending = repository.list_pending()
    assert approval.id in [item.id for item in pending]


@pytest.mark.parametrize(
    ("action", "decision", "expected_status"),
    [
        ("same_day_booking", "approved", "confirmed"),
        ("same_day_booking", "rejected", "rejected"),
        ("cancel_booking", "approved", "cancelled"),
        ("cancel_booking", "rejected", "confirmed"),
    ],
)
def test_decision_updates_approval_and_booking(
    database_connection: Connection[Any],
    booking_data: tuple[UUID, UUID],
    action: ApprovalAction,
    decision: str,
    expected_status: BookingStatus,
) -> None:
    repository = ApprovalRepository(database_connection)
    booking_id, conversation_id = booking_data

    if action == "cancel_booking":
        database_connection.execute(
            "UPDATE bookings SET status = %s WHERE id = %s",
            ("confirmed", booking_id),
        )

    approval = repository.create(booking_id, conversation_id, action)

    if decision == "approved":
        result = repository.decide(
            approval.id,
            "approved",
            decided_by="test-staff",
            decision_note="Test approval.",
        )
    else:
        result = repository.decide(
            approval.id,
            "rejected",
            decided_by="test-staff",
            decision_note="Test rejection.",
        )

    assert result.status == decision
    assert result.decided_by == "test-staff"
    assert result.decided_at is not None

    booking_row = database_connection.execute(
        "SELECT status FROM bookings WHERE id = %s",
        (booking_id,),
    ).fetchone()

    assert booking_row is not None
    assert booking_row[0] == expected_status

    pending = repository.list_pending()
    assert approval.id not in [item.id for item in pending]


def test_duplicate_pending_request(
    database_connection: Connection[Any],
    booking_data: tuple[UUID, UUID],
) -> None:
    repository = ApprovalRepository(database_connection)
    booking_id, conversation_id = booking_data

    repository.create(booking_id, conversation_id, "same_day_booking")

    with pytest.raises(ApprovalAlreadyPending):
        repository.create(booking_id, conversation_id, "same_day_booking")


def test_duplicate_decision(
    database_connection: Connection[Any],
    booking_data: tuple[UUID, UUID],
) -> None:
    repository = ApprovalRepository(database_connection)
    booking_id, conversation_id = booking_data

    approval = repository.create(
        booking_id,
        conversation_id,
        "same_day_booking",
    )

    repository.decide(
        approval.id,
        "approved",
        decided_by="test-staff",
        decision_note="First decision.",
    )

    with pytest.raises(ApprovalAlreadyDecided):
        repository.decide(
            approval.id,
            "rejected",
            decided_by="test-staff",
            decision_note="Repeated decision.",
        )


def test_unknown_approval(
    database_connection: Connection[Any],
) -> None:
    repository = ApprovalRepository(database_connection)

    assert repository.get(uuid4()) is None

    with pytest.raises(ApprovalNotFound):
        repository.decide(
            uuid4(),
            "approved",
            decided_by="test-staff",
            decision_note="Unknown approval.",
        )


def test_invalid_booking_state(
    database_connection: Connection[Any],
    booking_data: tuple[UUID, UUID],
) -> None:
    repository = ApprovalRepository(database_connection)
    booking_id, conversation_id = booking_data

    # Cancellation requires a confirmed booking.
    with pytest.raises(ApprovalStateConflict):
        repository.create(
            booking_id,
            conversation_id,
            "cancel_booking",
        )
