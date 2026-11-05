"""API tests for slot search and checkpointed appointment actions."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.repositories.errors import ConversationNotFound
from app.repositories.models import AvailableAppointmentSlot

_SLOT_ID = UUID("11111111-1111-4111-8111-111111111111")
_DOCTOR_ID = UUID("22222222-2222-4222-8222-222222222222")
_CONVERSATION_ID = UUID("33333333-3333-4333-8333-333333333333")
_OTHER_CONVERSATION_ID = UUID("66666666-6666-4666-8666-666666666666")
_BOOKING_ID = UUID("44444444-4444-4444-8444-444444444444")
_APPROVAL_ID = UUID("55555555-5555-4555-8555-555555555555")
_OWNER_REFERENCE = "opaque-authenticated-owner"
_BOOKING_REQUEST_ID = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
_CANCEL_REQUEST_ID = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
_STARTS_AT = datetime(2026, 12, 1, 9, 0, tzinfo=UTC)
_ENDS_AT = datetime(2026, 12, 1, 11, 0, tzinfo=UTC)


def _slot() -> AvailableAppointmentSlot:
    return AvailableAppointmentSlot(
        id=_SLOT_ID,
        doctor_id=_DOCTOR_ID,
        doctor_name="Dr. Fictional",
        doctor_specialty="General Medicine",
        starts_at=_STARTS_AT,
        ends_at=_STARTS_AT + timedelta(minutes=30),
    )


class _FakeAppointmentService:
    def __init__(
        self,
        slots: list[AvailableAppointmentSlot] | None = None,
        *,
        search_error: Exception | None = None,
    ) -> None:
        self.slots = list(slots) if slots is not None else [_slot()]
        self.search_error = search_error
        self.search_calls: list[dict[str, Any]] = []

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
        if self.search_error is not None:
            raise self.search_error
        return list(self.slots)


class _FakeConversationService:
    def __init__(
        self,
        booking_result: Mapping[str, object] | None = None,
        cancellation_result: Mapping[str, object] | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self.booking_result = dict(
            booking_result
            or {
                "booking_id": str(_BOOKING_ID),
                "booking_status": "confirmed",
                "approval_status": "not_required",
                "response": "Your appointment is booked.",
            }
        )
        self.cancellation_result = dict(
            cancellation_result
            or {
                "approval_id": str(_APPROVAL_ID),
                "booking_id": str(_BOOKING_ID),
                "approval_status": "pending",
                "response": "Your cancellation request is waiting for staff approval.",
            }
        )
        self.error = error
        self.actions: list[dict[str, object]] = []

    async def process_appointment_action(
        self,
        conversation_id: UUID,
        *,
        owner_reference: str,
        request_id: UUID,
        action: str,
        slot_id: UUID | None = None,
        booking_id: UUID | None = None,
    ) -> Mapping[str, object]:
        self.actions.append(
            {
                "conversation_id": conversation_id,
                "owner_reference": owner_reference,
                "request_id": request_id,
                "action": action,
                "slot_id": slot_id,
                "booking_id": booking_id,
            }
        )
        if owner_reference != _OWNER_REFERENCE or conversation_id != _CONVERSATION_ID:
            raise ConversationNotFound("Conversation was not found.")
        if self.error is not None:
            raise self.error
        return self.booking_result if action == "book" else self.cancellation_result


def _client(
    appointment_service: _FakeAppointmentService | None = None,
    conversation_service: _FakeConversationService | None = None,
    *,
    owner_reference: str | None = _OWNER_REFERENCE,
) -> TestClient:
    resolver = (lambda request: owner_reference) if owner_reference is not None else None
    app = create_app(
        appointment_service=appointment_service,
        conversation_service=conversation_service,
        owner_reference_resolver=resolver,
        configure_services=False,
    )
    return TestClient(app)


def _search_params() -> dict[str, str]:
    return {"starts_at": _STARTS_AT.isoformat(), "ends_at": _ENDS_AT.isoformat()}


def _booking_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "slot_id": str(_SLOT_ID),
        "conversation_id": str(_CONVERSATION_ID),
    }
    payload.update(overrides)
    return payload


def _cancellation_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "booking_id": str(_BOOKING_ID),
        "conversation_id": str(_CONVERSATION_ID),
    }
    payload.update(overrides)
    return payload


def test_search_slots_returns_available_slots() -> None:
    service = _FakeAppointmentService()
    client = _client(service)

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
    service = _FakeAppointmentService()
    client = _client(service)

    response = client.get(
        "/appointments/slots",
        params={**_search_params(), "doctor_id": str(_DOCTOR_ID), "limit": "10"},
    )

    assert response.status_code == 200
    assert service.search_calls == [
        {
            "starts_at": _STARTS_AT,
            "ends_at": _ENDS_AT,
            "doctor_id": _DOCTOR_ID,
            "limit": 10,
        }
    ]


def test_search_slots_rejects_invalid_range() -> None:
    service = _FakeAppointmentService()
    client = _client(service)

    response = client.get(
        "/appointments/slots",
        params={"starts_at": "2026-12-01T11:00:00Z", "ends_at": "2026-12-01T10:00:00Z"},
    )

    assert response.status_code == 422
    assert service.search_calls == []


def test_search_slots_rejects_naive_datetime_and_invalid_limit() -> None:
    client = _client(_FakeAppointmentService())

    naive = client.get(
        "/appointments/slots",
        params={"starts_at": "2026-12-01T09:00:00", "ends_at": _ENDS_AT.isoformat()},
    )
    invalid_limit = client.get(
        "/appointments/slots",
        params={**_search_params(), "limit": "101"},
    )

    assert naive.status_code == 422
    assert invalid_limit.status_code == 422


def test_create_booking_routes_through_workflow_with_authenticated_owner_and_key() -> None:
    workflow = _FakeConversationService()
    client = _client(conversation_service=workflow)

    response = client.post(
        "/appointments/bookings",
        headers={"Idempotency-Key": _BOOKING_REQUEST_ID},
        json=_booking_payload(),
    )

    assert response.status_code == 201
    assert response.json() == {
        "id": str(_BOOKING_ID),
        "slot_id": str(_SLOT_ID),
        "conversation_id": str(_CONVERSATION_ID),
        "status": "confirmed",
        "response": "Your appointment is booked.",
    }
    assert workflow.actions == [
        {
            "conversation_id": _CONVERSATION_ID,
            "owner_reference": _OWNER_REFERENCE,
            "request_id": UUID(_BOOKING_REQUEST_ID),
            "action": "book",
            "slot_id": _SLOT_ID,
            "booking_id": None,
        }
    ]


def test_same_day_booking_returns_accepted_while_waiting_for_staff() -> None:
    workflow = _FakeConversationService(
        booking_result={
            "booking_id": str(_BOOKING_ID),
            "booking_status": "pending_approval",
            "approval_status": "pending",
            "response": "Your booking request is waiting for staff approval.",
        }
    )
    client = _client(conversation_service=workflow)

    response = client.post(
        "/appointments/bookings",
        headers={"Idempotency-Key": _BOOKING_REQUEST_ID},
        json=_booking_payload(),
    )

    assert response.status_code == 202
    assert response.json()["status"] == "pending_approval"


def test_booking_requires_idempotency_key_and_authenticated_owner() -> None:
    missing_key = _client(conversation_service=_FakeConversationService()).post(
        "/appointments/bookings",
        json=_booking_payload(),
    )
    unauthenticated = _client(
        conversation_service=_FakeConversationService(),
        owner_reference=None,
    ).post(
        "/appointments/bookings",
        headers={"Idempotency-Key": _BOOKING_REQUEST_ID},
        json=_booking_payload(),
    )

    assert missing_key.status_code == 422
    assert unauthenticated.status_code == 401


def test_booking_rejects_identity_fields_in_request_body() -> None:
    workflow = _FakeConversationService()
    response = _client(conversation_service=workflow).post(
        "/appointments/bookings",
        headers={"Idempotency-Key": _BOOKING_REQUEST_ID},
        json=_booking_payload(patient_reference="forged-owner"),
    )

    assert response.status_code == 422
    assert workflow.actions == []


def test_cancellation_routes_through_checkpointed_workflow() -> None:
    workflow = _FakeConversationService()
    client = _client(conversation_service=workflow)

    response = client.post(
        "/appointments/cancellations",
        headers={"Idempotency-Key": _CANCEL_REQUEST_ID},
        json=_cancellation_payload(),
    )

    assert response.status_code == 202
    assert response.json() == {
        "id": str(_APPROVAL_ID),
        "booking_id": str(_BOOKING_ID),
        "conversation_id": str(_CONVERSATION_ID),
        "status": "pending",
        "response": "Your cancellation request is waiting for staff approval.",
    }
    assert workflow.actions[0]["action"] == "cancel"
    assert workflow.actions[0]["booking_id"] == _BOOKING_ID


def test_cancellation_endpoint_is_exposed_and_requires_idempotency_key() -> None:
    client = _client(conversation_service=_FakeConversationService())

    missing_key = client.post(
        "/appointments/cancellations",
        json=_cancellation_payload(),
    )
    valid = client.post(
        "/appointments/cancellations",
        headers={"Idempotency-Key": _CANCEL_REQUEST_ID},
        json=_cancellation_payload(),
    )

    assert missing_key.status_code == 422
    assert valid.status_code == 202


def test_cancellation_requires_authenticated_owner() -> None:
    response = _client(
        conversation_service=_FakeConversationService(),
        owner_reference=None,
    ).post(
        "/appointments/cancellations",
        headers={"Idempotency-Key": _CANCEL_REQUEST_ID},
        json=_cancellation_payload(),
    )

    assert response.status_code == 401


@pytest.mark.parametrize(
    ("path", "payload", "request_id"),
    [
        (
            "/appointments/bookings",
            {"slot_id": str(_SLOT_ID), "conversation_id": str(_OTHER_CONVERSATION_ID)},
            _BOOKING_REQUEST_ID,
        ),
        (
            "/appointments/cancellations",
            {"booking_id": str(_BOOKING_ID), "conversation_id": str(_OTHER_CONVERSATION_ID)},
            _CANCEL_REQUEST_ID,
        ),
    ],
)
def test_appointment_actions_hide_another_owners_conversation(
    path: str,
    payload: dict[str, str],
    request_id: str,
) -> None:
    workflow = _FakeConversationService()
    response = _client(conversation_service=workflow).post(
        path,
        headers={"Idempotency-Key": request_id},
        json=payload,
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "Conversation not found."}
    assert workflow.actions[0]["owner_reference"] == _OWNER_REFERENCE


def test_approval_workflow_domain_error_maps_to_conflict() -> None:
    workflow = _FakeConversationService(
        booking_result={
            "workflow_error": "approval_workflow_required",
            "response": "Requires approval.",
        }
    )
    client = _client(conversation_service=workflow)

    response = client.post(
        "/appointments/bookings",
        headers={"Idempotency-Key": _BOOKING_REQUEST_ID},
        json=_booking_payload(),
    )

    assert response.status_code == 503
    assert (
        response.json()["detail"] == "This request requires a configured staff-approval workflow."
    )


@pytest.mark.parametrize(
    ("workflow_error", "expected_status"),
    [
        ("booking_conflict", 409),
        ("slot_unavailable", 409),
        ("conversation_not_found", 404),
        ("booking_not_found", 404),
        ("booking_conversation_mismatch", 409),
        ("approval_already_pending", 409),
        ("approval_state_conflict", 409),
        ("unexpected_workflow_error", 500),
    ],
)
def test_appointment_workflow_errors_map_to_http_status(
    workflow_error: str,
    expected_status: int,
) -> None:
    workflow = _FakeConversationService(
        booking_result={"workflow_error": workflow_error, "response": "Safe error."}
    )
    response = _client(conversation_service=workflow).post(
        "/appointments/bookings",
        headers={"Idempotency-Key": _BOOKING_REQUEST_ID},
        json=_booking_payload(),
    )

    assert response.status_code == expected_status


def test_missing_services_fail_with_service_unavailable() -> None:
    client = _client()

    response = client.post(
        "/appointments/bookings",
        headers={"Idempotency-Key": _BOOKING_REQUEST_ID},
        json=_booking_payload(),
    )

    assert response.status_code == 503
