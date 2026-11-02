"""PostgreSQL integration tests for database constraints and triggers."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from threading import Barrier
from typing import Any
from uuid import UUID, uuid4

import psycopg
import pytest
from psycopg import Connection
from psycopg.errors import CheckViolation, ExclusionViolation, ForeignKeyViolation, UniqueViolation


def _insert_doctor(connection: Connection[Any], *, active: bool = True) -> UUID:
    doctor_id = uuid4()
    connection.execute(
        """
        INSERT INTO doctors (id, full_name, specialty, is_active)
        VALUES (%s, %s, %s, %s)
        """,
        (doctor_id, f"Test provider {doctor_id}", "Test specialty", active),
    )
    return doctor_id


def _insert_slot(
    connection: Connection[Any],
    doctor_id: UUID,
    starts_at: datetime,
    *,
    duration: timedelta = timedelta(minutes=30),
    status: str = "open",
) -> UUID:
    slot_id = uuid4()
    connection.execute(
        """
        INSERT INTO appointment_slots (id, doctor_id, starts_at, ends_at, status)
        VALUES (%s, %s, %s, %s, %s)
        """,
        (slot_id, doctor_id, starts_at, starts_at + duration, status),
    )
    return slot_id


def _insert_conversation(connection: Connection[Any]) -> UUID:
    conversation_id = uuid4()
    connection.execute(
        "INSERT INTO conversations (id, thread_id) VALUES (%s, %s)",
        (conversation_id, str(conversation_id)),
    )
    return conversation_id


def _insert_booking(
    connection: Connection[Any],
    slot_id: UUID,
    *,
    conversation_id: UUID | None = None,
    status: str = "confirmed",
) -> UUID:
    booking_id = uuid4()
    connection.execute(
        """
        INSERT INTO bookings (id, slot_id, conversation_id, patient_reference, status)
        VALUES (%s, %s, %s, %s, %s)
        """,
        (booking_id, slot_id, conversation_id, f"test-patient-{uuid4()}", status),
    )
    return booking_id


def _future_start() -> datetime:
    return datetime.now(UTC) + timedelta(days=3)


def test_adjacent_slots_for_a_doctor_are_allowed(
    database_connection: Connection[Any],
) -> None:
    doctor_id = _insert_doctor(database_connection)
    starts_at = _future_start()

    _insert_slot(database_connection, doctor_id, starts_at)
    _insert_slot(
        database_connection,
        doctor_id,
        starts_at + timedelta(minutes=30),
    )


def test_overlapping_slots_for_a_doctor_are_rejected(
    database_connection: Connection[Any],
) -> None:
    doctor_id = _insert_doctor(database_connection)
    starts_at = _future_start()
    _insert_slot(database_connection, doctor_id, starts_at)

    with pytest.raises(ExclusionViolation):
        _insert_slot(
            database_connection,
            doctor_id,
            starts_at + timedelta(minutes=15),
        )


def test_a_slot_cannot_have_two_active_bookings(
    database_connection: Connection[Any],
) -> None:
    doctor_id = _insert_doctor(database_connection)
    slot_id = _insert_slot(database_connection, doctor_id, _future_start())
    _insert_booking(database_connection, slot_id, status="confirmed")

    with pytest.raises(UniqueViolation):
        _insert_booking(database_connection, slot_id, status="pending_approval")


def test_concurrent_bookings_cannot_reserve_the_same_slot(database_url: str) -> None:
    with psycopg.connect(database_url) as setup_connection:
        doctor_id = _insert_doctor(setup_connection)
        slot_id = _insert_slot(setup_connection, doctor_id, _future_start())

    barrier = Barrier(2)

    def attempt_booking() -> str:
        try:
            with psycopg.connect(database_url) as connection:
                barrier.wait(timeout=10)
                _insert_booking(connection, slot_id)
        except UniqueViolation:
            return "conflict"
        return "booked"

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(lambda _: attempt_booking(), range(2)))
        assert results.count("booked") == 1
        assert results.count("conflict") == 1
    finally:
        with psycopg.connect(database_url) as cleanup_connection:
            cleanup_connection.execute("DELETE FROM bookings WHERE slot_id = %s", (slot_id,))
            cleanup_connection.execute("DELETE FROM appointment_slots WHERE id = %s", (slot_id,))
            cleanup_connection.execute("DELETE FROM doctors WHERE id = %s", (doctor_id,))


def test_blocked_slot_cannot_be_booked(database_connection: Connection[Any]) -> None:
    doctor_id = _insert_doctor(database_connection)
    slot_id = _insert_slot(database_connection, doctor_id, _future_start(), status="blocked")

    with pytest.raises(CheckViolation):
        _insert_booking(database_connection, slot_id)


def test_slot_with_active_booking_cannot_be_blocked(
    database_connection: Connection[Any],
) -> None:
    doctor_id = _insert_doctor(database_connection)
    slot_id = _insert_slot(database_connection, doctor_id, _future_start())
    _insert_booking(database_connection, slot_id)

    with pytest.raises(CheckViolation):
        database_connection.execute(
            "UPDATE appointment_slots SET status = 'blocked' WHERE id = %s",
            (slot_id,),
        )


def test_inactive_doctor_slot_cannot_be_booked(database_connection: Connection[Any]) -> None:
    doctor_id = _insert_doctor(database_connection, active=False)
    slot_id = _insert_slot(database_connection, doctor_id, _future_start())

    with pytest.raises(CheckViolation):
        _insert_booking(database_connection, slot_id)


def test_doctor_with_active_booking_cannot_be_deactivated(
    database_connection: Connection[Any],
) -> None:
    doctor_id = _insert_doctor(database_connection)
    slot_id = _insert_slot(database_connection, doctor_id, _future_start())
    _insert_booking(database_connection, slot_id)

    with pytest.raises(CheckViolation):
        database_connection.execute(
            "UPDATE doctors SET is_active = FALSE WHERE id = %s",
            (doctor_id,),
        )


def test_approval_must_match_the_booking_conversation(
    database_connection: Connection[Any],
) -> None:
    doctor_id = _insert_doctor(database_connection)
    slot_id = _insert_slot(database_connection, doctor_id, _future_start())
    booking_conversation_id = _insert_conversation(database_connection)
    other_conversation_id = _insert_conversation(database_connection)
    booking_id = _insert_booking(
        database_connection,
        slot_id,
        conversation_id=booking_conversation_id,
    )

    with pytest.raises(ForeignKeyViolation):
        database_connection.execute(
            """
            INSERT INTO approvals (booking_id, conversation_id, action)
            VALUES (%s, %s, 'cancel_booking')
            """,
            (booking_id, other_conversation_id),
        )


def test_updated_at_triggers_refresh_conversation_and_booking_timestamps(
    database_connection: Connection[Any],
) -> None:
    doctor_id = _insert_doctor(database_connection)
    slot_id = _insert_slot(database_connection, doctor_id, _future_start())
    conversation_id = _insert_conversation(database_connection)
    booking_id = _insert_booking(
        database_connection,
        slot_id,
        conversation_id=conversation_id,
    )

    original_conversation_time = database_connection.execute(
        "SELECT updated_at FROM conversations WHERE id = %s",
        (conversation_id,),
    ).fetchone()[0]
    original_booking_time = database_connection.execute(
        "SELECT updated_at FROM bookings WHERE id = %s",
        (booking_id,),
    ).fetchone()[0]

    database_connection.execute("SELECT pg_sleep(0.02)")
    database_connection.execute(
        "UPDATE conversations SET thread_id = %s WHERE id = %s",
        (f"updated-{uuid4()}", conversation_id),
    )
    database_connection.execute(
        "UPDATE bookings SET patient_reference = %s WHERE id = %s",
        (f"updated-patient-{uuid4()}", booking_id),
    )

    updated_conversation_time = database_connection.execute(
        "SELECT updated_at FROM conversations WHERE id = %s",
        (conversation_id,),
    ).fetchone()[0]
    updated_booking_time = database_connection.execute(
        "SELECT updated_at FROM bookings WHERE id = %s",
        (booking_id,),
    ).fetchone()[0]

    assert updated_conversation_time > original_conversation_time
    assert updated_booking_time > original_booking_time
