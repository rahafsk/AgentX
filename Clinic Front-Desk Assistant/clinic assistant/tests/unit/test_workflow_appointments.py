"""Tests for appointment search, explicit selection, and workflow failure handling."""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime, time, timedelta
from types import SimpleNamespace
from typing import Any, cast
from uuid import UUID

import pytest
from langchain_core.messages import HumanMessage

from app.graphs.states import ConversationState
from app.graphs.workflows import (
    _appointment_failure,
    _filter_daypart,
    _requested_time,
    _search_window,
    _selected_id,
    make_appointment_handler,
)
from app.repositories.errors import (
    ApprovalAlreadyPending,
    ApprovalStateConflict,
    ApprovalWorkflowRequired,
    BookingConflict,
    ConversationNotFound,
    SlotUnavailable,
)
from app.repositories.models import AppointmentSlot, AvailableAppointmentSlot, Booking
from app.services.appointments import BookingConversationMismatch, BookingNotFound

_CONVERSATION_ID = UUID("11111111-1111-4111-8111-111111111111")
_OWNER = "opaque-test-owner"
_BOOKING_ID = UUID("22222222-2222-4222-8222-222222222222")
_SECOND_BOOKING_ID = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
_SLOT_ID = UUID("33333333-3333-4333-8333-333333333333")
_SECOND_SLOT_ID = UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")
_DOCTOR_ID = UUID("44444444-4444-4444-8444-444444444444")


def _slot(
    slot_id: UUID = _SLOT_ID, *, starts_at: datetime | None = None
) -> AvailableAppointmentSlot:
    start = starts_at or datetime.now(UTC) + timedelta(days=1)
    return AvailableAppointmentSlot(
        id=slot_id,
        doctor_id=_DOCTOR_ID,
        doctor_name="Dr. Fictional",
        doctor_specialty="General",
        starts_at=start,
        ends_at=start + timedelta(minutes=30),
    )


class _FakeAppointments:
    def __init__(
        self,
        *,
        booking_status: str = "confirmed",
        slots: list[AvailableAppointmentSlot] | None = None,
        slot_records: list[AppointmentSlot] | None = None,
        search_error: Exception | None = None,
        booking_error: Exception | None = None,
        cancellation_error: Exception | None = None,
    ) -> None:
        self.booking_status = booking_status
        self.slots = list(slots) if slots is not None else [_slot()]
        self.slot_records = {slot.id: slot for slot in slot_records or []}
        self.search_error = search_error
        self.booking_error = booking_error
        self.cancellation_error = cancellation_error
        self.cancellations: list[tuple[UUID, UUID, str]] = []
        self.bookings: list[tuple[UUID, str, UUID, UUID | None]] = []
        self.search_calls: list[tuple[datetime, datetime, int]] = []

    def search_available_slots(
        self,
        starts_at: datetime,
        ends_at: datetime,
        *,
        doctor_id: UUID | None = None,
        limit: int = 5,
    ) -> list[AvailableAppointmentSlot]:
        self.search_calls.append((starts_at, ends_at, limit))
        if self.search_error is not None:
            raise self.search_error
        return [slot for slot in self.slots if starts_at <= slot.starts_at < ends_at][:limit]

    def get_slot(self, slot_id: UUID) -> AppointmentSlot | None:
        return self.slot_records.get(slot_id)

    def book_appointment(
        self,
        slot_id: UUID,
        owner_reference: str,
        conversation_id: UUID,
        *,
        approval_workflow: bool,
        request_id: UUID | None,
    ) -> Any:
        self.bookings.append((slot_id, owner_reference, conversation_id, request_id))
        if self.booking_error is not None:
            raise self.booking_error
        return SimpleNamespace(
            id=request_id or _BOOKING_ID,
            slot_id=slot_id,
            status=self.booking_status,
        )

    def request_cancellation(
        self,
        booking_id: UUID,
        conversation_id: UUID,
        owner_reference: str,
        *,
        request_id: UUID | None,
    ) -> Any:
        self.cancellations.append((booking_id, conversation_id, owner_reference))
        if self.cancellation_error is not None:
            raise self.cancellation_error
        return SimpleNamespace(
            id=request_id or UUID("55555555-5555-4555-8555-555555555555"),
            booking_id=booking_id,
            status="pending",
        )


class _FakeBookings:
    def __init__(self, bookings: list[Booking] | None = None) -> None:
        self._bookings = bookings or []

    def get(self, booking_id: UUID) -> Booking | None:
        return next((booking for booking in self._bookings if booking.id == booking_id), None)

    def list_for_conversation(self, conversation_id: UUID) -> list[Booking]:
        return [booking for booking in self._bookings if booking.conversation_id == conversation_id]


def _state(
    intent: str,
    *,
    text: str = "",
    requested_slot: str | None = None,
    requested_booking_id: str | None = None,
    available_slots: list[dict[str, str]] | None = None,
    available_bookings: list[dict[str, str]] | None = None,
    selected_slot_id: str | None = None,
) -> ConversationState:
    return cast(
        ConversationState,
        {
            "conversation_id": str(_CONVERSATION_ID),
            "intent": intent,
            "requested_slot": requested_slot,
            "requested_booking_id": requested_booking_id,
            "available_slots": available_slots or [],
            "available_bookings": available_bookings or [],
            "selected_slot_id": selected_slot_id,
            "patient_id": _OWNER,
            "request_id": "66666666-6666-4666-8666-666666666666",
            "messages": [HumanMessage(content=text)] if text else [],
        },
    )


async def test_search_intent_returns_available_selectable_slot_ids() -> None:
    slot = _slot()
    appointments = _FakeAppointments(slots=[slot])
    handler = make_appointment_handler(appointments, _FakeBookings())

    result = await handler(_state("find_slots", text="What appointments are available?"))

    assert result["available_slots"][0]["id"] == str(slot.id)
    assert str(slot.id) in result["response"]
    assert appointments.search_calls
    assert appointments.bookings == []


async def test_search_intent_returns_an_honest_no_slots_fallback() -> None:
    handler = make_appointment_handler(_FakeAppointments(slots=[]), _FakeBookings())

    result = await handler(_state("find_slots", text="Find available appointments tomorrow"))

    assert result["available_slots"] == []
    assert "couldn't find an available slot" in result["response"].lower()


async def test_slot_selection_requires_an_explicit_booking_request() -> None:
    slot = _slot()
    option = {
        "id": str(slot.id),
        "starts_at": slot.starts_at.isoformat(),
        "doctor_name": slot.doctor_name,
        "doctor_specialty": slot.doctor_specialty,
    }
    appointments = _FakeAppointments(slots=[slot])
    handler = make_appointment_handler(appointments, _FakeBookings())

    result = await handler(_state("find_slots", text="option 1", available_slots=[option]))

    assert result["selected_slot_id"] == str(slot.id)
    assert "book this slot" in result["response"]
    assert appointments.bookings == []


async def test_booking_request_uses_selected_slot_and_records_booking() -> None:
    slot = _slot()
    option = {
        "id": str(slot.id),
        "starts_at": slot.starts_at.isoformat(),
        "doctor_name": slot.doctor_name,
        "doctor_specialty": slot.doctor_specialty,
    }
    appointments = _FakeAppointments(slots=[slot])
    handler = make_appointment_handler(appointments, _FakeBookings())

    result = await handler(
        _state(
            "book_appointment",
            text="Book this slot",
            available_slots=[option],
            selected_slot_id=str(slot.id),
        )
    )

    assert result["booking_status"] == "confirmed"
    assert appointments.bookings == [
        (slot.id, _OWNER, _CONVERSATION_ID, UUID("66666666-6666-4666-8666-666666666666"))
    ]


async def test_same_day_booking_returns_pending_approval() -> None:
    appointments = _FakeAppointments(booking_status="pending_approval")
    handler = make_appointment_handler(appointments, _FakeBookings())

    result = await handler(_state("book_appointment", requested_slot=str(_SLOT_ID)))

    assert result["approval_status"] == "pending"
    assert result["booking_id"] == "66666666-6666-4666-8666-666666666666"


async def test_cancellation_lists_multiple_bookings_instead_of_choosing_oldest() -> None:
    future = datetime.now(UTC) + timedelta(days=2)
    first = Booking(
        id=_BOOKING_ID,
        slot_id=_SLOT_ID,
        conversation_id=_CONVERSATION_ID,
        patient_reference=_OWNER,
        status="confirmed",
        created_at=future,
        updated_at=future,
    )
    second = Booking(
        id=_SECOND_BOOKING_ID,
        slot_id=_SECOND_SLOT_ID,
        conversation_id=_CONVERSATION_ID,
        patient_reference=_OWNER,
        status="confirmed",
        created_at=future,
        updated_at=future,
    )
    slot_one = AppointmentSlot(
        id=_SLOT_ID,
        doctor_id=_DOCTOR_ID,
        starts_at=future,
        ends_at=future + timedelta(minutes=30),
        status="open",
        created_at=future,
    )
    slot_two = AppointmentSlot(
        id=_SECOND_SLOT_ID,
        doctor_id=_DOCTOR_ID,
        starts_at=future + timedelta(days=1),
        ends_at=future + timedelta(days=1, minutes=30),
        status="open",
        created_at=future,
    )
    appointments = _FakeAppointments(slot_records=[slot_one, slot_two])
    handler = make_appointment_handler(appointments, _FakeBookings([first, second]))

    result = await handler(_state("cancel_appointment", text="Cancel my appointment"))

    assert len(result["available_bookings"]) == 2
    assert "Which appointment" in result["response"]
    assert appointments.cancellations == []


async def test_cancellation_uses_the_selected_booking() -> None:
    future = datetime.now(UTC) + timedelta(days=1)
    booking = Booking(
        id=_SECOND_BOOKING_ID,
        slot_id=_SECOND_SLOT_ID,
        conversation_id=_CONVERSATION_ID,
        patient_reference=_OWNER,
        status="confirmed",
        created_at=future,
        updated_at=future,
    )
    slot = AppointmentSlot(
        id=_SECOND_SLOT_ID,
        doctor_id=_DOCTOR_ID,
        starts_at=future,
        ends_at=future + timedelta(minutes=30),
        status="open",
        created_at=future,
    )
    appointments = _FakeAppointments(slot_records=[slot])
    handler = make_appointment_handler(appointments, _FakeBookings([booking]))

    result = await handler(
        _state(
            "cancel_appointment",
            text="Cancel option 1",
            available_bookings=[
                {
                    "booking_id": str(booking.id),
                    "slot_id": str(slot.id),
                    "starts_at": slot.starts_at.isoformat(),
                }
            ],
        )
    )

    assert result["approval_status"] == "pending"
    assert appointments.cancellations == [(_SECOND_BOOKING_ID, _CONVERSATION_ID, _OWNER)]


async def test_explicit_single_cancellation_requests_staff_approval() -> None:
    future = datetime.now(UTC) + timedelta(days=1)
    booking = Booking(
        id=_BOOKING_ID,
        slot_id=_SLOT_ID,
        conversation_id=_CONVERSATION_ID,
        patient_reference=_OWNER,
        status="confirmed",
        created_at=future,
        updated_at=future,
    )
    slot = AppointmentSlot(
        id=_SLOT_ID,
        doctor_id=_DOCTOR_ID,
        starts_at=future,
        ends_at=future + timedelta(minutes=30),
        status="open",
        created_at=future,
    )
    appointments = _FakeAppointments(slot_records=[slot])
    handler = make_appointment_handler(appointments, _FakeBookings([booking]))

    result = await handler(_state("cancel_appointment", text="Cancel my appointment"))

    assert result["approval_status"] == "pending"
    assert appointments.cancellations == [(_BOOKING_ID, _CONVERSATION_ID, _OWNER)]


async def test_cancel_menu_ignores_past_and_nonconfirmed_bookings() -> None:
    now = datetime.now(UTC)
    old_booking = Booking(
        id=_BOOKING_ID,
        slot_id=_SLOT_ID,
        conversation_id=_CONVERSATION_ID,
        patient_reference=_OWNER,
        status="confirmed",
        created_at=now - timedelta(days=2),
        updated_at=now - timedelta(days=2),
    )
    rejected_booking = Booking(
        id=_SECOND_BOOKING_ID,
        slot_id=_SECOND_SLOT_ID,
        conversation_id=_CONVERSATION_ID,
        patient_reference=_OWNER,
        status="rejected",
        created_at=now,
        updated_at=now,
    )
    old_slot = AppointmentSlot(
        id=_SLOT_ID,
        doctor_id=_DOCTOR_ID,
        starts_at=now - timedelta(days=2),
        ends_at=now - timedelta(days=2) + timedelta(minutes=30),
        status="open",
        created_at=now - timedelta(days=2),
    )
    appointments = _FakeAppointments(slot_records=[old_slot])
    handler = make_appointment_handler(
        appointments,
        _FakeBookings([old_booking, rejected_booking]),
    )

    result = await handler(_state("cancel_appointment", text="Cancel my appointment"))

    assert result["available_bookings"] == []
    assert appointments.cancellations == []


async def test_reschedule_request_gets_actionable_help() -> None:
    handler = make_appointment_handler(_FakeAppointments(), _FakeBookings())

    result = await handler(_state("reschedule_appointment", text="Move my appointment"))

    assert "replacement time" in result["response"]


async def test_appointment_service_failure_is_not_reported_as_success() -> None:
    appointments = _FakeAppointments(search_error=RuntimeError("database unavailable"))
    handler = make_appointment_handler(appointments, _FakeBookings())

    result = await handler(_state("find_slots", text="Find available appointments"))

    assert result["workflow_error"] == "appointment_handler_failed"
    assert "couldn't complete" in result["response"]


def test_invalid_conversation_state_fails_closed() -> None:
    handler = make_appointment_handler(_FakeAppointments(), _FakeBookings())

    invalid_owner = cast(ConversationState, {"conversation_id": str(_CONVERSATION_ID)})
    invalid_conversation = _state("find_slots")
    invalid_conversation["conversation_id"] = "not-a-uuid"

    owner_result = asyncio.run(handler(invalid_owner))
    conversation_result = asyncio.run(handler(invalid_conversation))

    assert owner_result["workflow_error"] == "conversation_not_found"
    assert conversation_result["workflow_error"] == "conversation_not_found"


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (ConversationNotFound(), "conversation_not_found"),
        (BookingNotFound(), "booking_not_found"),
        (BookingConversationMismatch(), "booking_conversation_mismatch"),
        (BookingConflict(), "booking_conflict"),
        (SlotUnavailable(), "slot_unavailable"),
        (ApprovalAlreadyPending(), "approval_already_pending"),
        (ApprovalStateConflict(), "approval_state_conflict"),
        (ApprovalWorkflowRequired(), "approval_workflow_required"),
        (RuntimeError("database unavailable"), "appointment_handler_failed"),
    ],
)
def test_domain_failures_have_explicit_safe_outcomes(error: Exception, expected: str) -> None:
    result = _appointment_failure(error)

    assert result["workflow_error"] == expected
    assert isinstance(result["response"], str)
    assert "database unavailable" not in result["response"]


def test_date_daypart_and_time_preferences_are_parsed() -> None:
    start, end, daypart = _search_window("Find a slot tomorrow morning")
    assert end - start <= timedelta(days=1)
    assert daypart == "morning"

    specific_start, specific_end, _ = _search_window("Appointment on 2030-04-05 at 3:30pm")
    assert specific_start.date() == date(2030, 4, 5)
    assert specific_end - specific_start == timedelta(days=1)
    assert _requested_time("Any time near 3:30pm") == time(15, 30)


def test_weekday_preferences_and_daypart_filtering() -> None:
    start, end, daypart = _search_window("Next Tuesday afternoon")
    assert start.weekday() == 1
    assert end - start <= timedelta(days=1)
    assert daypart == "afternoon"

    morning = _slot(starts_at=datetime(2030, 4, 5, 5, tzinfo=UTC))
    evening = _slot(
        UUID("cccccccc-cccc-4ccc-8ccc-cccccccccccc"),
        starts_at=datetime(2030, 4, 5, 19, tzinfo=UTC),
    )
    assert _filter_daypart([morning, evening], "morning") == [morning]


def test_selection_parser_accepts_only_offered_ids_and_explicit_options() -> None:
    options = [
        {"id": str(_SLOT_ID)},
        {"id": "55555555-5555-4555-8555-555555555555"},
    ]

    assert _selected_id("option 2", options, id_field="id") == options[1]["id"]
    assert _selected_id("the first one", options, id_field="id") == options[0]["id"]
    assert _selected_id("book first available appointment", options, id_field="id") is None
    assert _selected_id(str(_SLOT_ID), options, id_field="id") == options[0]["id"]
