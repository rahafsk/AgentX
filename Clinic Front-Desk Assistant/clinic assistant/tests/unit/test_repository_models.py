"""Unit tests for typed repository records."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from app.repositories.models import (
    AppointmentSlot,
    Approval,
    AvailableAppointmentSlot,
    Booking,
    Conversation,
    Doctor,
)


def test_repository_models_represent_database_records() -> None:
    now = datetime.now(UTC)
    doctor_id = uuid4()
    slot_id = uuid4()
    conversation_id = uuid4()
    booking_id = uuid4()
    approval_id = uuid4()

    doctor = Doctor(doctor_id, "Demo Provider", "DEMO ONLY", True, now)
    slot = AppointmentSlot(
        slot_id,
        doctor_id,
        now + timedelta(days=1),
        now + timedelta(days=1, minutes=30),
        "open",
        now,
    )
    available_slot = AvailableAppointmentSlot(
        slot.id,
        doctor.id,
        doctor.full_name,
        doctor.specialty,
        slot.starts_at,
        slot.ends_at,
    )
    conversation = Conversation(conversation_id, str(conversation_id), now, now)
    booking = Booking(
        booking_id,
        slot_id,
        conversation_id,
        "opaque-test-reference",
        "pending_approval",
        now,
        now,
    )
    approval = Approval(
        approval_id,
        booking_id,
        conversation_id,
        "same_day_booking",
        "pending",
        now,
        None,
        None,
        None,
    )

    assert available_slot.doctor_id == doctor.id
    assert conversation.thread_id == str(conversation.id)
    assert booking.status == "pending_approval"
    assert approval.booking_id == booking.id
    with pytest.raises(FrozenInstanceError):
        doctor.is_active = False  # type: ignore[misc]
