"""API tests for the appointment endpoints."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.appointments import router
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
)
from app.services.appointments import (
    BookingConversationMismatch,
    BookingNotFound,
)

_SLOT_ID = UUID("11111111-1111-4111-8111-111111111111")
_DOCTOR_ID = UUID("22222222-2222-4222-8222-222222222222")
_CONVERSATION_ID = UUID("33333333-3333-4333-8333-333333333333")
_BOOKING_ID = UUID("44444444-4444-4444-8444-444444444444")
_APPROVAL_ID = UUID("55555555-5555-4555-8555-555555555555")
_STARTS_AT = datetime(2026, 12, 1, 9, 0, tzinfo=UTC)
_ENDS_AT = datetime(2026, 12, 1, 11, 0, tzinfo=UTC)


def _slot() -> AvailableAppointmentSlot:
    """Return a fictional open slot for API responses."""
    return AvailableAppointmentSlot(
        id=_SLOT_ID,
        doctor_id=_DOCTOR_ID,
        doctor_name="Dr. Fictional",
        doctor_specialty="General Medicine",
        starts_at=_STARTS_AT,
        ends_at=_STARTS_AT + timedelta(minutes=30),
    )


def _booking(status: BookingStatus = "confirmed") -> Booking:
    """Return a fictional booking in the requested status."""
    return Booking(
        id=_BOOKING_ID,
        slot_id=_SLOT_ID,
        conversation_id=_CONVERSATION_ID,
        patient_reference="fictional-patient",
        status=status,
        created_at=_STARTS_AT,
        updated_at=_STARTS_AT,
    )


def _approval() -> Approval:
    """Return a fictional pending cancellation approval."""
    return Approval(
        id=_APPROVAL_ID,
        booking_id=_BOOKING_ID,
        conversation_id=_CONVERSATION_ID,
        action="cancel_booking",
        status="pending",
        requested_at=_STARTS_AT,
        decided_at=None,
        decided_by=None,
        decision_note=None,
    )


class _FakeService:
    """Configurable appointment service stand-in for HTTP tests."""

    def __init__(
        self,
        *,
        slots: list[AvailableAppointmentSlot] | None = None,
        booking: Booking | None = None,
        approval: Approval | None = None,
        search_error: Exception | None = None,
        booking_error: Exception | None = None,
        cancellation_error: Exception | None = None,
    ) -> None:
        self._slots = list(slots) if slots is not None else [_slot()]
        self._booking = booking or _booking()
        self._approval = approval or _approval()
        self._search_error = search_error
        self._booking_error = booking_error
        self._cancellation_error = cancellation_error
        self.search_calls: list[dict[str, object]] = []
        self.booking_calls: list[dict[str, object]] = []
        self.cancellation_calls: list[dict[str, object]] = []

    def search_available_slots(
        self,
        starts_at: datetime,
        ends_at: datetime,
        *,
        doctor_id: UUID | None = None,
        limit: int = 25,
    ) -> list[AvailableAppointmentSlot]:
        self.search_calls.append(
            {
                "starts_at": starts_at,
                "ends_at": ends_at,
                "doctor_id": doctor_id,
                "limit": limit,
            }
        )
        if self._search_error is not None:
            raise self._search_error
        return list(self._slots)

    def book_appointment(
        self,
        slot_id: UUID,
        patient_reference: str,
        conversation_id: UUID,
        *,
        requires_approval: bool = False,
    ) -> Booking:
        self.booking_calls.append(
            {
                "slot_id": slot_id,
                "patient_reference": patient_reference,
                "conversation_id": conversation_id,
                "requires_approval": requires_approval,
            }
        )
        if self._booking_error is not None:
            raise self._booking_error
        return self._booking

    def request_cancellation(
        self,
        booking_id: UUID,
        conversation_id: UUID,
    ) -> Approval:
        self.cancellation_calls.append(
            {
                "booking_id": booking_id,
                "conversation_id": conversation_id,
            }
        )
        if self._cancellation_error is not None:
            raise self._cancellation_error
        return self._approval


def _client(service: _FakeService | None) -> TestClient:
    """Build an isolated app with the appointments router."""
    app = FastAPI()
    app.include_router(router)
    if service is not None:
        app.state.appointment_service = service
    return TestClient(app)


def _search_params() -> dict[str, str]:
    """Return a valid slot-search query string."""
    return {
        "starts_at": _STARTS_AT.isoformat(),
        "ends_at": _ENDS_AT.isoformat(),
    }


def _booking_payload(**overrides: object) -> dict[str, object]:
    """Return a valid booking request body."""
    payload: dict[str, object] = {
        "slot_id": str(_SLOT_ID),
        "patient_reference": "fictional-patient",
        "conversation_id": str(_CONVERSATION_ID),
        "requires_approval": False,
    }
    payload.update(overrides)
    return payload


def _cancellation_payload(**overrides: object) -> dict[str, object]:
    """Return a valid cancellation request body."""
    payload: dict[str, object] = {
        "booking_id": str(_BOOKING_ID),
        "conversation_id": str(_CONVERSATION_ID),
    }
    payload.update(overrides)
    return payload


def test_search_slots_returns_available_slots() -> None:
    client = _client(_FakeService())

    response = client.get("/appointments/slots", params=_search_params())

    assert response.status_code == 200
    assert response.json() == [
        {
            "id": str(_SLOT_ID),
            "doctor_id": str(_DOCTOR_ID),
            "doctor_name": "Dr. Fictional",
            "doctor_specialty": "General Medicine",
            "starts_at": "2026-12-01T09:00:00Z",
            "ends_at": "2026-12-01T09:30:00Z",
        }
    ]


def test_search_slots_applies_optional_doctor_and_limit() -> None:
    service = _FakeService()
    client = _client(service)
    params = {
        **_search_params(),
        "doctor_id": str(_DOCTOR_ID),
        "limit": "10",
    }

    response = client.get("/appointments/slots", params=params)

    assert response.status_code == 200
    assert service.search_calls == [
        {
            "starts_at": _STARTS_AT,
            "ends_at": _ENDS_AT,
            "doctor_id": _DOCTOR_ID,
            "limit": 10,
        }
    ]


@pytest.mark.parametrize(
    "params",
    [
        {
            "starts_at": "2026-12-01T09:00:00",
            "ends_at": _ENDS_AT.isoformat(),
        },
        {
            "starts_at": _ENDS_AT.isoformat(),
            "ends_at": _STARTS_AT.isoformat(),
        },
        {
            **_search_params(),
            "limit": "0",
        },
        {
            **_search_params(),
            "limit": "101",
        },
        {
            **_search_params(),
            "doctor_id": "not-a-uuid",
        },
        {
            "starts_at": "not-a-datetime",
            "ends_at": _ENDS_AT.isoformat(),
        },
    ],
)
def test_search_slots_rejects_invalid_input(params: dict[str, str]) -> None:
    service = _FakeService()
    client = _client(service)

    response = client.get("/appointments/slots", params=params)

    assert response.status_code == 422
    assert service.search_calls == []


def test_create_booking_returns_confirmed_booking() -> None:
    service = _FakeService(booking=_booking(status="confirmed"))
    client = _client(service)

    response = client.post(
        "/appointments/bookings",
        json=_booking_payload(patient_reference="  fictional-patient  "),
    )

    assert response.status_code == 201
    assert response.json() == {
        "id": str(_BOOKING_ID),
        "slot_id": str(_SLOT_ID),
        "conversation_id": str(_CONVERSATION_ID),
        "patient_reference": "fictional-patient",
        "status": "confirmed",
        "created_at": "2026-12-01T09:00:00Z",
        "updated_at": "2026-12-01T09:00:00Z",
    }
    assert service.booking_calls == [
        {
            "slot_id": _SLOT_ID,
            "patient_reference": "fictional-patient",
            "conversation_id": _CONVERSATION_ID,
            "requires_approval": False,
        }
    ]


def test_create_booking_preserves_same_day_pending_approval() -> None:
    service = _FakeService(booking=_booking(status="pending_approval"))
    client = _client(service)

    response = client.post(
        "/appointments/bookings",
        json=_booking_payload(requires_approval=True),
    )

    assert response.status_code == 201
    assert response.json()["status"] == "pending_approval"
    assert service.booking_calls == [
        {
            "slot_id": _SLOT_ID,
            "patient_reference": "fictional-patient",
            "conversation_id": _CONVERSATION_ID,
            "requires_approval": True,
        }
    ]


def test_create_booking_rejects_blank_patient_reference() -> None:
    service = _FakeService()
    client = _client(service)

    response = client.post(
        "/appointments/bookings",
        json=_booking_payload(patient_reference="   "),
    )

    assert response.status_code == 422
    assert service.booking_calls == []


def test_create_booking_returns_conflict_for_duplicate_booking() -> None:
    service = _FakeService(
        booking_error=BookingConflict("The appointment slot already has an active booking.")
    )
    client = _client(service)

    response = client.post("/appointments/bookings", json=_booking_payload())

    assert response.status_code == 409
    assert response.json() == {
        "detail": "The appointment slot already has an active booking.",
    }


def test_create_booking_returns_conflict_for_unavailable_slot() -> None:
    service = _FakeService(
        booking_error=SlotUnavailable("The slot is blocked or its doctor is inactive.")
    )
    client = _client(service)

    response = client.post("/appointments/bookings", json=_booking_payload())

    assert response.status_code == 409
    assert response.json() == {
        "detail": "The selected appointment slot is unavailable.",
    }


def test_create_booking_returns_not_found_for_missing_conversation() -> None:
    service = _FakeService(booking_error=ConversationNotFound("Conversation was not found."))
    client = _client(service)

    response = client.post("/appointments/bookings", json=_booking_payload())

    assert response.status_code == 404
    assert response.json() == {"detail": "Conversation not found."}


def test_request_cancellation_returns_pending_approval() -> None:
    service = _FakeService()
    client = _client(service)

    response = client.post(
        "/appointments/cancellations",
        json=_cancellation_payload(),
    )

    assert response.status_code == 201
    assert response.json() == {
        "id": str(_APPROVAL_ID),
        "booking_id": str(_BOOKING_ID),
        "conversation_id": str(_CONVERSATION_ID),
        "action": "cancel_booking",
        "status": "pending",
        "requested_at": "2026-12-01T09:00:00Z",
        "decided_at": None,
        "decided_by": None,
        "decision_note": None,
    }
    assert service.cancellation_calls == [
        {
            "booking_id": _BOOKING_ID,
            "conversation_id": _CONVERSATION_ID,
        }
    ]


def test_request_cancellation_returns_not_found_for_missing_booking() -> None:
    service = _FakeService(cancellation_error=BookingNotFound("The booking does not exist."))
    client = _client(service)

    response = client.post(
        "/appointments/cancellations",
        json=_cancellation_payload(),
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "Booking not found."}


def test_request_cancellation_returns_conflict_for_another_conversation() -> None:
    service = _FakeService(
        cancellation_error=BookingConversationMismatch(
            "The booking belongs to a different conversation."
        )
    )
    client = _client(service)

    response = client.post(
        "/appointments/cancellations",
        json=_cancellation_payload(),
    )

    assert response.status_code == 409
    assert response.json() == {
        "detail": "The booking belongs to a different conversation.",
    }


def test_request_cancellation_returns_conflict_for_invalid_booking_state() -> None:
    service = _FakeService(
        cancellation_error=ApprovalStateConflict(
            "A cancel_booking approval requires a booking in 'confirmed' state."
        )
    )
    client = _client(service)

    response = client.post(
        "/appointments/cancellations",
        json=_cancellation_payload(),
    )

    assert response.status_code == 409
    assert response.json() == {
        "detail": "Only confirmed bookings can be cancelled.",
    }


def test_request_cancellation_returns_conflict_for_pending_approval() -> None:
    service = _FakeService(
        cancellation_error=ApprovalAlreadyPending(
            "An approval for this booking action is already pending."
        )
    )
    client = _client(service)

    response = client.post(
        "/appointments/cancellations",
        json=_cancellation_payload(),
    )

    assert response.status_code == 409
    assert response.json() == {
        "detail": "A cancellation request for this booking is already pending.",
    }


@pytest.mark.parametrize(
    "payload",
    [
        {"conversation_id": str(_CONVERSATION_ID)},
        {"booking_id": str(_BOOKING_ID)},
        {
            "booking_id": "not-a-uuid",
            "conversation_id": str(_CONVERSATION_ID),
        },
        {
            "booking_id": str(_BOOKING_ID),
            "conversation_id": "not-a-uuid",
        },
    ],
)
def test_request_cancellation_rejects_invalid_input(payload: dict[str, object]) -> None:
    service = _FakeService()
    client = _client(service)

    response = client.post("/appointments/cancellations", json=payload)

    assert response.status_code == 422
    assert service.cancellation_calls == []


def test_appointment_endpoints_require_configured_service() -> None:
    client = _client(None)

    search_response = client.get("/appointments/slots", params=_search_params())
    booking_response = client.post("/appointments/bookings", json=_booking_payload())
    cancellation_response = client.post(
        "/appointments/cancellations",
        json=_cancellation_payload(),
    )

    assert search_response.status_code == 503
    assert search_response.json() == {
        "detail": "Appointment service is not configured.",
    }
    assert booking_response.status_code == 503
    assert booking_response.json() == {
        "detail": "Appointment service is not configured.",
    }
    assert cancellation_response.status_code == 503
    assert cancellation_response.json() == {
        "detail": "Appointment service is not configured.",
    }
