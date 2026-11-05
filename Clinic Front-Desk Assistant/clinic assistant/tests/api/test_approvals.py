"""HTTP tests for staff-authenticated approval actions."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal
from uuid import UUID

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.approvals import router
from app.api.dependencies import get_authenticated_staff_reference
from app.repositories.errors import ApprovalNotFound
from app.repositories.models import Approval
from app.services.approvals import ApprovalDecisionConflict, ApprovalResumeTimeout

_APPROVAL_ID = UUID("11111111-1111-4111-8111-111111111111")
_BOOKING_ID = UUID("22222222-2222-4222-8222-222222222222")
_CONVERSATION_ID = UUID("33333333-3333-4333-8333-333333333333")
_NOW = datetime(2026, 1, 1, tzinfo=UTC)
_STAFF = "trusted-staff-context"


def _approval(status: Literal["pending", "approved", "rejected"] = "pending") -> Approval:
    decided = status != "pending"
    return Approval(
        id=_APPROVAL_ID,
        booking_id=_BOOKING_ID,
        conversation_id=_CONVERSATION_ID,
        action="cancel_booking",
        status=status,
        requested_at=_NOW,
        decided_at=_NOW if decided else None,
        decided_by=_STAFF if decided else None,
        decision_note="Reviewed." if decided else None,
    )


class _FakeApprovalService:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.decisions: list[tuple[UUID, str, str, str]] = []

    async def list_pending(self, *, limit: int = 50) -> list[Approval]:
        return [_approval()][:limit]

    async def decide(
        self,
        approval_id: UUID,
        decision: Literal["approved", "rejected"],
        *,
        decided_by: str,
        decision_note: str,
    ) -> tuple[Approval, str]:
        self.decisions.append((approval_id, decision, decided_by, decision_note))
        if self.error is not None:
            raise self.error
        return _approval(decision), "The approval decision was recorded."


def _client(
    service: _FakeApprovalService | None,
    *,
    staff_reference: str | None = _STAFF,
) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    if service is not None:
        app.state.approval_service = service
    if staff_reference is not None:
        app.dependency_overrides[get_authenticated_staff_reference] = lambda: staff_reference
    return TestClient(app)


def test_list_pending_approvals_requires_authenticated_staff() -> None:
    response = _client(_FakeApprovalService()).get("/approvals")
    assert response.status_code == 200
    assert response.json()[0]["id"] == str(_APPROVAL_ID)
    assert response.json()[0]["status"] == "pending"

    unauthorized = _client(_FakeApprovalService(), staff_reference=None).get("/approvals/pending")
    assert unauthorized.status_code == 401


def test_staff_decision_uses_trusted_identity_and_records_audit_note() -> None:
    service = _FakeApprovalService()
    response = _client(service).post(
        f"/approvals/{_APPROVAL_ID}/decision",
        json={"decision": "approved", "decision_note": "Reviewed by clinic staff."},
    )

    assert response.status_code == 200
    assert response.json()["approval"]["status"] == "approved"
    assert response.json()["approval"]["decided_by"] == _STAFF
    assert response.json()["response"] == "The approval decision was recorded."
    assert service.decisions == [(_APPROVAL_ID, "approved", _STAFF, "Reviewed by clinic staff.")]


def test_decision_identity_cannot_be_supplied_in_request_body() -> None:
    service = _FakeApprovalService()
    response = _client(service).post(
        f"/approvals/{_APPROVAL_ID}/decision",
        json={
            "decision": "approved",
            "decided_by": "forged-staff",
            "decision_note": "Reviewed.",
        },
    )

    assert response.status_code == 422
    assert service.decisions == []


def test_decision_rejects_blank_note_and_invalid_decision() -> None:
    client = _client(_FakeApprovalService())
    blank_note = client.post(
        f"/approvals/{_APPROVAL_ID}/decision",
        json={"decision": "rejected", "decision_note": "   "},
    )
    invalid_decision = client.post(
        f"/approvals/{_APPROVAL_ID}/decision",
        json={"decision": "maybe", "decision_note": "Reviewed."},
    )

    assert blank_note.status_code == 422
    assert invalid_decision.status_code == 422


@pytest.mark.parametrize(
    ("error", "status_code"),
    [
        (ApprovalNotFound(), 404),
        (ApprovalDecisionConflict(), 409),
        (ApprovalResumeTimeout(), 504),
    ],
)
def test_approval_failures_map_to_safe_http_responses(
    error: Exception,
    status_code: int,
) -> None:
    response = _client(_FakeApprovalService(error=error)).post(
        f"/approvals/{_APPROVAL_ID}/decision",
        json={"decision": "approved", "decision_note": "Reviewed."},
    )

    assert response.status_code == status_code


def test_missing_approval_service_returns_service_unavailable() -> None:
    response = _client(None).get("/approvals/pending")

    assert response.status_code == 503
