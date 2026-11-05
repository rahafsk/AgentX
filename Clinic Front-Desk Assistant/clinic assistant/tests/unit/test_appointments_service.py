"""Unit tests for the appointment service."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

import pytest

from app.repositories.errors import (
    ApprovalAlreadyPending,
    ApprovalStateConflict,
    BookingConflict,
    ConversationNotFound,
    SlotUnavailable,
)
from app.repositories.models import (
    Approval,
    AvailableAppointmentSlot,
    Booking,
    BookingStatus,
    Conversation,
)
from app.services.appointments import (
    DEFAULT_SEARCH_LIMIT,
    AppointmentService,
    BookingConversationMismatch,
    BookingNotFound,
)

_NOW = datetime(2026, 1, 1, 9, 0, tzinfo=UTC)
_OWNER_REFERENCE = "fictional-patient"


def _slot(
    *,
    slot_id: UUID | None = None,
    doctor_id: UUID | None = None,
    starts_at: datetime = _NOW,
) -> AvailableAppointmentSlot:
    return AvailableAppointmentSlot(
        id=slot_id or uuid4(),
        doctor_id=doctor_id or uuid4(),
        doctor_name="Dr. Fictional",
        doctor_specialty="General",
        starts_at=starts_at,
        ends_at=starts_at + timedelta(minutes=30),
    )


def _booking(
    *,
    booking_id: UUID | None = None,
    conversation_id: UUID | None = None,
    status: BookingStatus = "confirmed",
) -> Booking:
    return Booking(
        id=booking_id or uuid4(),
        slot_id=uuid4(),
        conversation_id=conversation_id or uuid4(),
        patient_reference="fictional-patient",
        status=status,
        created_at=_NOW,
        updated_at=_NOW,
    )


def _approval(
    *,
    booking_id: UUID,
    conversation_id: UUID,
    action: str = "cancel_booking",
) -> Approval:
    return Approval(
        id=uuid4(),
        booking_id=booking_id,
        conversation_id=conversation_id,
        action=action,  # type: ignore[arg-type]
        status="pending",
        requested_at=_NOW,
        decided_at=None,
        decided_by=None,
        decision_note=None,
    )


class _FakeAppointmentRepository:
    def __init__(
        self,
        slots: list[AvailableAppointmentSlot] | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self.slots = list(slots or [])
        self.error = error
        self.calls: list[tuple[datetime, datetime, UUID | None, int]] = []

    def search_open_slots(
        self,
        starts_at: datetime,
        ends_at: datetime,
        *,
        doctor_id: UUID | None = None,
        limit: int = DEFAULT_SEARCH_LIMIT,
    ) -> list[AvailableAppointmentSlot]:
        self.calls.append((starts_at, ends_at, doctor_id, limit))
        if self.error is not None:
            raise self.error
        return list(self.slots)


class _FakeBookingRepository:
    def __init__(
        self,
        bookings: list[Booking] | None = None,
        *,
        create_result: Booking | None = None,
        create_error: Exception | None = None,
    ) -> None:
        self._bookings = {booking.id: booking for booking in bookings or []}
        self.create_result = create_result
        self.create_error = create_error
        self.create_calls: list[dict[str, Any]] = []
        self.get_calls: list[UUID] = []

    def create(
        self,
        slot_id: UUID,
        patient_reference: str,
        *,
        conversation_id: UUID | None = None,
        approval_workflow: bool = False,
        request_id: UUID | None = None,
    ) -> Booking:
        self.create_calls.append(
            {
                "slot_id": slot_id,
                "patient_reference": patient_reference,
                "conversation_id": conversation_id,
                "approval_workflow": approval_workflow,
                "request_id": request_id,
            }
        )
        if self.create_error is not None:
            raise self.create_error
        if self.create_result is None:
            raise AssertionError("The fake booking repository has no result configured.")
        return self.create_result

    def get(self, booking_id: UUID) -> Booking | None:
        self.get_calls.append(booking_id)
        return self._bookings.get(booking_id)


class _FakeApprovalRepository:
    def __init__(
        self,
        approval: Approval | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self.approval = approval
        self.error = error
        self.create_calls: list[dict[str, Any]] = []

    def create(
        self,
        booking_id: UUID,
        conversation_id: UUID,
        action: str,
        *,
        request_id: UUID | None = None,
    ) -> Approval:
        self.create_calls.append(
            {
                "booking_id": booking_id,
                "conversation_id": conversation_id,
                "action": action,
                "request_id": request_id,
            }
        )
        if self.error is not None:
            raise self.error
        if self.approval is None:
            raise AssertionError("The fake approval repository has no result configured.")
        return self.approval


class _FakeConversationRepository:
    def __init__(
        self,
        conversations: list[Conversation] | None = None,
        *,
        allow_any: bool = False,
    ) -> None:
        self._conversations = {
            conversation.id: conversation for conversation in conversations or []
        }
        self._allow_any = allow_any
        self.calls: list[UUID] = []

    def get_for_owner(
        self,
        conversation_id: UUID,
        owner_reference: str,
    ) -> Conversation | None:
        self.calls.append(conversation_id)
        conversation = self._conversations.get(conversation_id)
        if conversation is not None and conversation.owner_reference == owner_reference:
            return conversation
        if self._allow_any:
            return _conversation(conversation_id, owner_reference)
        return None


def _conversation(
    conversation_id: UUID,
    owner_reference: str = "fictional-patient",
) -> Conversation:
    return Conversation(
        id=conversation_id,
        thread_id=str(conversation_id),
        created_at=_NOW,
        updated_at=_NOW,
        owner_reference=owner_reference,
    )


def _service(
    *,
    appointments: _FakeAppointmentRepository | None = None,
    bookings: _FakeBookingRepository | None = None,
    approvals: _FakeApprovalRepository | None = None,
    conversations: _FakeConversationRepository | None = None,
) -> AppointmentService:
    return AppointmentService(
        appointments or _FakeAppointmentRepository(),  # type: ignore[arg-type]
        bookings or _FakeBookingRepository(),  # type: ignore[arg-type]
        approvals or _FakeApprovalRepository(),  # type: ignore[arg-type]
        conversations or _FakeConversationRepository(allow_any=True),  # type: ignore[arg-type]
    )


def test_from_connection_builds_a_service() -> None:
    service = AppointmentService.from_connection(object())  # type: ignore[arg-type]

    assert isinstance(service, AppointmentService)


def test_search_available_slots_returns_repository_results() -> None:
    slots = [_slot(), _slot(starts_at=_NOW + timedelta(hours=1))]
    appointments = _FakeAppointmentRepository(slots)
    service = _service(appointments=appointments)
    ends_at = _NOW + timedelta(hours=2)

    result = service.search_available_slots(_NOW, ends_at)

    assert result == slots
    assert appointments.calls == [(_NOW, ends_at, None, DEFAULT_SEARCH_LIMIT)]


def test_search_available_slots_filters_by_doctor_and_limit() -> None:
    doctor_id = uuid4()
    appointments = _FakeAppointmentRepository([_slot()])
    service = _service(appointments=appointments)
    ends_at = _NOW + timedelta(hours=2)

    service.search_available_slots(
        _NOW,
        ends_at,
        doctor_id=doctor_id,
        limit=10,
    )

    assert appointments.calls == [(_NOW, ends_at, doctor_id, 10)]


def test_search_available_slots_propagates_repository_errors() -> None:
    error = RuntimeError("database unavailable")
    appointments = _FakeAppointmentRepository(error=error)
    service = _service(appointments=appointments)

    with pytest.raises(RuntimeError, match="database unavailable"):
        service.search_available_slots(_NOW, _NOW + timedelta(hours=1))


@pytest.mark.parametrize(
    ("starts_at", "ends_at"),
    [
        (datetime(2026, 1, 1, 9, 0), _NOW + timedelta(hours=1)),
        (_NOW, datetime(2026, 1, 1, 10, 0)),
        (_NOW, _NOW),
        (_NOW, _NOW - timedelta(minutes=1)),
    ],
)
def test_search_available_slots_rejects_invalid_time_ranges(
    starts_at: datetime,
    ends_at: datetime,
) -> None:
    appointments = _FakeAppointmentRepository()
    service = _service(appointments=appointments)

    with pytest.raises(ValueError):
        service.search_available_slots(starts_at, ends_at)

    assert appointments.calls == []


@pytest.mark.parametrize("limit", [0, -1, 101])
def test_search_available_slots_rejects_out_of_range_limits(limit: int) -> None:
    appointments = _FakeAppointmentRepository()
    service = _service(appointments=appointments)

    with pytest.raises(ValueError, match="limit"):
        service.search_available_slots(_NOW, _NOW + timedelta(hours=1), limit=limit)

    assert appointments.calls == []


@pytest.mark.parametrize("limit", [True, "10", 1.5])
def test_search_available_slots_rejects_non_integer_limits(limit: object) -> None:
    appointments = _FakeAppointmentRepository()
    service = _service(appointments=appointments)

    with pytest.raises(TypeError, match="limit"):
        service.search_available_slots(
            _NOW,
            _NOW + timedelta(hours=1),
            limit=limit,  # type: ignore[arg-type]
        )

    assert appointments.calls == []


def test_search_available_slots_rejects_invalid_doctor_id() -> None:
    appointments = _FakeAppointmentRepository()
    service = _service(appointments=appointments)

    with pytest.raises(TypeError, match="doctor_id"):
        service.search_available_slots(
            _NOW,
            _NOW + timedelta(hours=1),
            doctor_id="not-a-uuid",  # type: ignore[arg-type]
        )

    assert appointments.calls == []


def test_search_available_slots_rejects_non_datetime_input() -> None:
    appointments = _FakeAppointmentRepository()
    service = _service(appointments=appointments)

    with pytest.raises(TypeError, match="starts_at"):
        service.search_available_slots(
            "tomorrow",  # type: ignore[arg-type]
            _NOW + timedelta(hours=1),
        )

    assert appointments.calls == []


def test_book_appointment_delegates_to_repository() -> None:
    booking = _booking(status="confirmed")
    bookings = _FakeBookingRepository(create_result=booking)
    approvals = _FakeApprovalRepository()
    service = _service(bookings=bookings, approvals=approvals)
    slot_id = uuid4()
    conversation_id = booking.conversation_id
    assert conversation_id is not None
    request_id = uuid4()

    result = service.book_appointment(
        slot_id,
        "  fictional-patient  ",
        conversation_id,
        request_id=request_id,
    )

    assert result is booking
    assert bookings.create_calls == [
        {
            "slot_id": slot_id,
            "patient_reference": "fictional-patient",
            "conversation_id": conversation_id,
            "approval_workflow": False,
            "request_id": request_id,
        }
    ]
    assert approvals.create_calls == []


def test_book_appointment_returns_pending_booking_for_same_day_approval() -> None:
    booking = _booking(status="pending_approval")
    bookings = _FakeBookingRepository(create_result=booking)
    approvals = _FakeApprovalRepository()
    service = _service(bookings=bookings, approvals=approvals)
    conversation_id = booking.conversation_id
    assert conversation_id is not None

    result = service.book_appointment(
        uuid4(),
        "fictional-patient",
        conversation_id,
        approval_workflow=True,
    )

    assert result.status == "pending_approval"
    assert bookings.create_calls[0]["approval_workflow"] is True
    # BookingRepository creates the approval; the service must not duplicate it.
    assert approvals.create_calls == []


def test_book_appointment_passes_explicit_approval_requirement() -> None:
    booking = _booking(status="pending_approval")
    bookings = _FakeBookingRepository(create_result=booking)
    service = _service(bookings=bookings)
    conversation_id = booking.conversation_id
    assert conversation_id is not None

    service.book_appointment(
        uuid4(),
        "fictional-patient",
        conversation_id,
        approval_workflow=True,
    )

    assert bookings.create_calls[0]["approval_workflow"] is True


def test_book_appointment_propagates_booking_conflict() -> None:
    error = BookingConflict("The appointment slot already has an active booking.")
    bookings = _FakeBookingRepository(create_error=error)
    service = _service(bookings=bookings)

    with pytest.raises(BookingConflict):
        service.book_appointment(uuid4(), "fictional-patient", uuid4())


def test_book_appointment_propagates_unavailable_slot() -> None:
    error = SlotUnavailable("The slot is blocked or its doctor is inactive.")
    bookings = _FakeBookingRepository(create_error=error)
    service = _service(bookings=bookings)

    with pytest.raises(SlotUnavailable):
        service.book_appointment(uuid4(), "fictional-patient", uuid4())


def test_book_appointment_validates_existing_conversation() -> None:
    conversation_id = uuid4()
    conversations = _FakeConversationRepository([_conversation(conversation_id)])
    booking = _booking(conversation_id=conversation_id)
    bookings = _FakeBookingRepository(create_result=booking)
    service = _service(bookings=bookings, conversations=conversations)

    result = service.book_appointment(uuid4(), "fictional-patient", conversation_id)

    assert result is booking
    assert conversations.calls == [conversation_id]
    assert len(bookings.create_calls) == 1


def test_book_appointment_rejects_missing_conversation() -> None:
    conversation_id = uuid4()
    conversations = _FakeConversationRepository()
    bookings = _FakeBookingRepository(create_result=_booking())
    service = _service(bookings=bookings, conversations=conversations)

    with pytest.raises(ConversationNotFound):
        service.book_appointment(uuid4(), "fictional-patient", conversation_id)

    assert conversations.calls == [conversation_id]
    assert bookings.create_calls == []


def test_book_appointment_requires_owner_bound_conversation() -> None:
    booking = _booking()
    bookings = _FakeBookingRepository(create_result=booking)
    conversation_id = booking.conversation_id
    assert conversation_id is not None
    service = _service(
        bookings=bookings,
        conversations=_FakeConversationRepository(),
    )

    with pytest.raises(ConversationNotFound):
        service.book_appointment(uuid4(), "fictional-patient", conversation_id)

    assert bookings.create_calls == []


@pytest.mark.parametrize(
    ("slot_id", "patient_reference", "conversation_id", "error_type"),
    [
        ("not-a-uuid", "fictional-patient", uuid4(), TypeError),
        (uuid4(), 123, uuid4(), TypeError),
        (uuid4(), "   ", uuid4(), ValueError),
        (uuid4(), "fictional-patient", "not-a-uuid", TypeError),
    ],
)
def test_book_appointment_rejects_invalid_inputs(
    slot_id: object,
    patient_reference: object,
    conversation_id: object,
    error_type: type[Exception],
) -> None:
    bookings = _FakeBookingRepository(create_result=_booking())
    service = _service(bookings=bookings)

    with pytest.raises(error_type):
        service.book_appointment(
            slot_id,  # type: ignore[arg-type]
            patient_reference,  # type: ignore[arg-type]
            conversation_id,  # type: ignore[arg-type]
        )

    assert bookings.create_calls == []


def test_book_appointment_rejects_invalid_approval_flag() -> None:
    bookings = _FakeBookingRepository(create_result=_booking())
    service = _service(bookings=bookings)

    with pytest.raises(TypeError, match="approval_workflow"):
        service.book_appointment(
            uuid4(),
            "fictional-patient",
            uuid4(),
            approval_workflow="yes",  # type: ignore[arg-type]
        )

    assert bookings.create_calls == []


def test_request_cancellation_creates_pending_approval() -> None:
    conversation_id = uuid4()
    booking = _booking(conversation_id=conversation_id)
    approval = _approval(
        booking_id=booking.id,
        conversation_id=conversation_id,
    )
    bookings = _FakeBookingRepository([booking])
    approvals = _FakeApprovalRepository(approval)
    service = _service(bookings=bookings, approvals=approvals)
    request_id = uuid4()

    result = service.request_cancellation(
        booking.id,
        conversation_id,
        _OWNER_REFERENCE,
        request_id=request_id,
    )

    assert result is approval
    assert bookings.get_calls == [booking.id]
    assert approvals.create_calls == [
        {
            "booking_id": booking.id,
            "conversation_id": conversation_id,
            "action": "cancel_booking",
            "request_id": request_id,
        }
    ]


def test_request_cancellation_rejects_missing_booking() -> None:
    bookings = _FakeBookingRepository()
    approvals = _FakeApprovalRepository()
    service = _service(bookings=bookings, approvals=approvals)
    booking_id = uuid4()

    with pytest.raises(BookingNotFound):
        service.request_cancellation(booking_id, uuid4(), _OWNER_REFERENCE)

    assert bookings.get_calls == [booking_id]
    assert approvals.create_calls == []


def test_request_cancellation_rejects_another_conversation() -> None:
    booking = _booking(conversation_id=uuid4())
    bookings = _FakeBookingRepository([booking])
    approvals = _FakeApprovalRepository()
    service = _service(bookings=bookings, approvals=approvals)

    with pytest.raises(BookingConversationMismatch):
        service.request_cancellation(booking.id, uuid4(), _OWNER_REFERENCE)

    assert bookings.get_calls == [booking.id]
    assert approvals.create_calls == []


def test_request_cancellation_propagates_already_pending_approval() -> None:
    conversation_id = uuid4()
    booking = _booking(conversation_id=conversation_id)
    error = ApprovalAlreadyPending("An approval for this booking action is already pending.")
    bookings = _FakeBookingRepository([booking])
    approvals = _FakeApprovalRepository(error=error)
    service = _service(bookings=bookings, approvals=approvals)

    with pytest.raises(ApprovalAlreadyPending):
        service.request_cancellation(booking.id, conversation_id, _OWNER_REFERENCE)


def test_request_cancellation_propagates_booking_state_conflict() -> None:
    conversation_id = uuid4()
    booking = _booking(conversation_id=conversation_id, status="pending_approval")
    error = ApprovalStateConflict(
        "A cancel_booking approval requires a booking in 'confirmed' state."
    )
    bookings = _FakeBookingRepository([booking])
    approvals = _FakeApprovalRepository(error=error)
    service = _service(bookings=bookings, approvals=approvals)

    with pytest.raises(ApprovalStateConflict):
        service.request_cancellation(booking.id, conversation_id, _OWNER_REFERENCE)


@pytest.mark.parametrize(
    ("booking_id", "conversation_id"),
    [
        ("not-a-uuid", uuid4()),
        (uuid4(), "not-a-uuid"),
    ],
)
def test_request_cancellation_rejects_invalid_ids(
    booking_id: object,
    conversation_id: object,
) -> None:
    bookings = _FakeBookingRepository()
    approvals = _FakeApprovalRepository()
    service = _service(bookings=bookings, approvals=approvals)

    with pytest.raises(TypeError):
        service.request_cancellation(
            booking_id,  # type: ignore[arg-type]
            conversation_id,  # type: ignore[arg-type]
            _OWNER_REFERENCE,
        )

    assert bookings.get_calls == []
    assert approvals.create_calls == []
