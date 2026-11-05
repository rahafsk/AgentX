"""PostgreSQL integration tests for repository behavior."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

import pytest
from psycopg import Connection

from app.repositories import (
    AppointmentRepository,
    AppointmentSlotConflict,
    ApprovalAlreadyDecided,
    ApprovalAlreadyPending,
    ApprovalContextMismatch,
    ApprovalNotFound,
    ApprovalRepository,
    ApprovalStateConflict,
    BookingConflict,
    BookingRepository,
    ConversationRepository,
    DoctorRepository,
    SlotUnavailable,
)


def _insert_doctor(connection: Connection[Any], *, active: bool = True) -> UUID:
    doctor_id = uuid4()
    connection.execute(
        """
        INSERT INTO doctors (id, full_name, specialty, is_active)
        VALUES (%s, %s, %s, %s)
        """,
        (doctor_id, f"Repository test doctor {doctor_id}", "Test specialty", active),
    )
    return doctor_id


def _insert_conversation(connection: Connection[Any]) -> UUID:
    conversation_id = uuid4()
    connection.execute(
        "INSERT INTO conversations (id, thread_id) VALUES (%s, %s)",
        (conversation_id, str(conversation_id)),
    )
    return conversation_id


def _future_start(hours_from_now: int = 72) -> datetime:
    return datetime.now(UTC) + timedelta(hours=hours_from_now)


def test_doctor_repository_reads_active_doctors(database_connection: Connection[Any]) -> None:
    active_id = _insert_doctor(database_connection)
    inactive_id = _insert_doctor(database_connection, active=False)
    repository = DoctorRepository(database_connection)

    active_doctors = repository.list_active()

    assert active_id in {doctor.id for doctor in active_doctors}
    assert inactive_id not in {doctor.id for doctor in active_doctors}
    assert repository.get(active_id).full_name == f"Repository test doctor {active_id}"
    assert repository.get(uuid4()) is None


def test_appointment_repository_filters_unavailable_slots(
    database_connection: Connection[Any],
) -> None:
    appointments = AppointmentRepository(database_connection)
    bookings = BookingRepository(database_connection)
    active_doctor = _insert_doctor(database_connection)
    inactive_doctor = _insert_doctor(database_connection, active=False)
    start = _future_start()

    available = appointments.create_slot(active_doctor, start, start + timedelta(minutes=30))
    appointments.create_slot(
        active_doctor,
        start + timedelta(hours=1),
        start + timedelta(hours=1, minutes=30),
        status="blocked",
    )
    reserved = appointments.create_slot(
        active_doctor,
        start + timedelta(hours=2),
        start + timedelta(hours=2, minutes=30),
    )
    appointments.create_slot(
        inactive_doctor,
        start + timedelta(hours=3),
        start + timedelta(hours=3, minutes=30),
    )
    bookings.create(reserved.id, "opaque-test-patient")

    results = appointments.search_open_slots(
        start - timedelta(minutes=1),
        start + timedelta(hours=4),
    )

    assert [slot.id for slot in results] == [available.id]
    assert results[0].doctor_name == f"Repository test doctor {active_doctor}"


def test_appointment_repository_maps_overlapping_slot_conflict(
    database_connection: Connection[Any],
) -> None:
    appointments = AppointmentRepository(database_connection)
    doctor_id = _insert_doctor(database_connection)
    start = _future_start()
    appointments.create_slot(doctor_id, start, start + timedelta(minutes=30))

    with pytest.raises(AppointmentSlotConflict):
        appointments.create_slot(
            doctor_id,
            start + timedelta(minutes=10),
            start + timedelta(minutes=40),
        )


def test_booking_repository_maps_conflict_and_unavailable_slot(
    database_connection: Connection[Any],
) -> None:
    appointments = AppointmentRepository(database_connection)
    bookings = BookingRepository(database_connection)
    doctor_id = _insert_doctor(database_connection)
    start = _future_start()
    slot = appointments.create_slot(doctor_id, start, start + timedelta(minutes=30))
    conversation = ConversationRepository(database_connection).create()

    booking = bookings.create(slot.id, "opaque-confirmed-patient")
    assert booking.status == "confirmed"
    with pytest.raises(BookingConflict):
        bookings.create(
            slot.id,
            "opaque-pending-patient",
            conversation_id=conversation.id,
            requires_approval=True,
        )

    blocked = appointments.create_slot(
        doctor_id,
        start + timedelta(hours=1),
        start + timedelta(hours=1, minutes=30),
        status="blocked",
    )
    with pytest.raises(SlotUnavailable):
        bookings.create(blocked.id, "opaque-blocked-slot-patient")

    inactive_doctor = _insert_doctor(database_connection, active=False)
    inactive_slot = appointments.create_slot(
        inactive_doctor,
        start + timedelta(hours=2),
        start + timedelta(hours=2, minutes=30),
    )
    with pytest.raises(SlotUnavailable):
        bookings.create(inactive_slot.id, "opaque-inactive-doctor-patient")


def test_booking_repository_requires_conversation_for_approval(
    database_connection: Connection[Any],
) -> None:
    doctor_id = _insert_doctor(database_connection)
    slot = AppointmentRepository(database_connection).create_slot(
        doctor_id, _future_start(), _future_start() + timedelta(minutes=30)
    )

    with pytest.raises(ValueError, match="conversation_id is required"):
        BookingRepository(database_connection).create(
            slot.id,
            "opaque-same-day-patient",
            requires_approval=True,
        )


def test_conversation_repository_keeps_id_and_thread_id_consistent(
    database_connection: Connection[Any],
) -> None:
    repository = ConversationRepository(database_connection)

    conversation = repository.create()

    assert conversation.thread_id == str(conversation.id)
    assert repository.get(conversation.id) == conversation
    assert repository.get_by_thread_id(conversation.thread_id) == conversation
    assert repository.get(uuid4()) is None


def test_approval_repository_enforces_context_and_updates_booking_atomically(
    database_connection: Connection[Any],
) -> None:
    conversations = ConversationRepository(database_connection)
    appointments = AppointmentRepository(database_connection)
    bookings = BookingRepository(database_connection)
    approvals = ApprovalRepository(database_connection)
    doctor_id = _insert_doctor(database_connection)
    start = _future_start()
    conversation = conversations.create()
    other_conversation = conversations.create()
    slot = appointments.create_slot(doctor_id, start, start + timedelta(minutes=30))
    booking = bookings.create(
        slot.id,
        "opaque-approval-patient",
        conversation_id=conversation.id,
        requires_approval=True,
    )

    with pytest.raises(ApprovalContextMismatch):
        approvals.create(booking.id, other_conversation.id, "same_day_booking")

    approval = next(item for item in approvals.list_pending() if item.booking_id == booking.id)

    assert approval.id in {item.id for item in approvals.list_pending()}

    with pytest.raises(ApprovalAlreadyPending):
        approvals.create(booking.id, conversation.id, "same_day_booking")

    decided = approvals.decide(
        approval.id,
        "approved",
        decided_by="test-front-desk",
        decision_note="Approved in a repository integration test.",
    )

    assert decided.status == "approved"
    assert bookings.get(booking.id).status == "confirmed"
    assert approval.id not in {item.id for item in approvals.list_pending()}

    with pytest.raises(ApprovalAlreadyDecided):
        approvals.decide(
            approval.id,
            "rejected",
            decided_by="test-front-desk",
            decision_note="Checking a repeated decision.",
        )

    with pytest.raises(ApprovalNotFound):
        approvals.decide(
            uuid4(),
            "approved",
            decided_by="test-front-desk",
            decision_note="Checking an unknown approval.",
        )


def test_approval_repository_rejects_wrong_booking_state(
    database_connection: Connection[Any],
) -> None:
    doctor_id = _insert_doctor(database_connection)
    conversation = ConversationRepository(database_connection).create()
    start = _future_start()
    slot = AppointmentRepository(database_connection).create_slot(
        doctor_id, start, start + timedelta(minutes=30)
    )
    booking = BookingRepository(database_connection).create(
        slot.id,
        "opaque-confirmed-patient",
        conversation_id=conversation.id,
    )

    with pytest.raises(ApprovalStateConflict):
        ApprovalRepository(database_connection).create(
            booking.id,
            conversation.id,
            "same_day_booking",
        )
