"""Tests for the appointment-service workflow adapter."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from uuid import UUID

from app.graphs.states import ConversationState
from app.graphs.workflows import make_appointment_handler

_CONVERSATION_ID = UUID("11111111-1111-1111-1111-111111111111")
_BOOKING_ID = UUID("22222222-2222-2222-2222-222222222222")
_SLOT_ID = UUID("33333333-3333-3333-3333-333333333333")


class _FakeAppointments:
    def __init__(self, *, booking_status: str = "confirmed") -> None:
        self.booking_status = booking_status
        self.cancellations: list[tuple[UUID, UUID]] = []
        self.bookings: list[tuple[UUID, str, UUID]] = []

    def request_cancellation(self, booking_id: UUID, conversation_id: UUID) -> Any:
        self.cancellations.append((booking_id, conversation_id))
        return SimpleNamespace(status="pending")

    def book_appointment(
        self,
        slot_id: UUID,
        patient_reference: str,
        conversation_id: UUID,
    ) -> Any:
        self.bookings.append((slot_id, patient_reference, conversation_id))
        return SimpleNamespace(status=self.booking_status)


class _FakeBookings:
    def __init__(self, bookings: list[Any] | None = None) -> None:
        self._bookings = bookings or []

    def list_for_conversation(self, conversation_id: UUID) -> list[Any]:
        return self._bookings


def _state(
    intent: str,
    *,
    requested_slot: str | None = None,
    patient_id: str | None = None,
) -> ConversationState:
    return cast(
        ConversationState,
        {
            "conversation_id": str(_CONVERSATION_ID),
            "intent": intent,
            "requested_slot": requested_slot,
            "patient_id": patient_id,
            "messages": [],
        },
    )


async def test_cancellation_creates_pending_approval() -> None:
    appointments: Any = _FakeAppointments()
    bookings: Any = _FakeBookings([SimpleNamespace(id=_BOOKING_ID, status="confirmed")])
    handler = make_appointment_handler(
        appointments,
        bookings,
    )

    result = await handler(_state("cancel_appointment"))

    assert result["approval_status"] == "pending"
    assert appointments.cancellations == [(_BOOKING_ID, _CONVERSATION_ID)]


async def test_cancellation_without_active_booking_is_safe() -> None:
    appointments: Any = _FakeAppointments()
    handler = make_appointment_handler(
        appointments,
        cast(Any, _FakeBookings()),
    )

    result = await handler(_state("cancel_appointment"))

    assert result.get("approval_status") is None
    assert appointments.cancellations == []


async def test_same_day_booking_creates_pending_approval() -> None:
    appointments: Any = _FakeAppointments(booking_status="pending_approval")
    handler = make_appointment_handler(
        appointments,
        cast(Any, _FakeBookings()),
    )

    result = await handler(
        _state(
            "book_appointment",
            requested_slot=str(_SLOT_ID),
            patient_id="patient-1",
        )
    )

    assert result["approval_status"] == "pending"
    assert appointments.bookings == [(_SLOT_ID, "patient-1", _CONVERSATION_ID)]


async def test_normal_booking_is_confirmed() -> None:
    appointments: Any = _FakeAppointments(booking_status="confirmed")
    handler = make_appointment_handler(
        appointments,
        cast(Any, _FakeBookings()),
    )

    result = await handler(_state("book_appointment", requested_slot=str(_SLOT_ID)))

    assert result["approval_status"] == "not_required"


async def test_missing_or_invalid_slot_is_safe() -> None:
    appointments: Any = _FakeAppointments()
    handler = make_appointment_handler(
        appointments,
        cast(Any, _FakeBookings()),
    )

    missing = await handler(_state("book_appointment"))
    invalid = await handler(_state("book_appointment", requested_slot="not-a-uuid"))

    assert "slot" in str(missing["response"]).lower()
    assert "valid" in str(invalid["response"]).lower()
    assert appointments.bookings == []
