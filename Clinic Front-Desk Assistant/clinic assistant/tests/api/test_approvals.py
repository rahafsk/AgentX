"""HTTP behavior for staff approval decisions and checkpoint resumes."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal
from uuid import UUID

from fastapi.testclient import TestClient

from app.main import create_app
from app.repositories.models import Approval

_APPROVAL_ID = UUID("11111111-1111-4111-8111-111111111111")
_BOOKING_ID = UUID("22222222-2222-4222-8222-222222222222")
_CONVERSATION_ID = UUID("33333333-3333-4333-8333-333333333333")
_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _approval(status: Literal["pending", "approved", "rejected"] = "pending") -> Approval:
    return Approval(
        id=_APPROVAL_ID,
        booking_id=_BOOKING_ID,
        conversation_id=_CONVERSATION_ID,
        action="same_day_booking",
        status=status,
        requested_at=_NOW,
        decided_at=_NOW if status != "pending" else None,
        decided_by="test-staff" if status != "pending" else None,
        decision_note="Approved for test." if status != "pending" else None,
    )


class _FakeApprovalService:
    def __init__(self) -> None:
        self.decisions: list[tuple[UUID, str, str, str]] = []

    async def list_pending(self, *, limit: int = 50) -> list[Approval]:
        assert limit == 50
        return [_approval()]

    async def decide(
        self,
        approval_id: UUID,
        decision: Literal["approved", "rejected"],
        *,
        decided_by: str,
        decision_note: str,
    ) -> tuple[Approval, dict[str, Any]]:
        self.decisions.append((approval_id, decision, decided_by, decision_note))
        return _approval(decision), {"response": "Your appointment is approved."}


def _client(
    service: _FakeApprovalService,
    *,
    staff_reference: str | None = "trusted-clinic-staff",
) -> TestClient:
    resolver = (lambda request: staff_reference) if staff_reference is not None else None
    application = create_app(
        approval_service=service,
        staff_reference_resolver=resolver,
        configure_services=False,
    )
    return TestClient(application)


def test_pending_approval_list_requires_authenticated_staff() -> None:
    response = _client(_FakeApprovalService()).get("/approvals/pending")

    assert response.status_code == 200
    assert response.json()[0]["id"] == str(_APPROVAL_ID)
    assert response.json()[0]["status"] == "pending"

    unauthorized = _client(
        _FakeApprovalService(),
        staff_reference=None,
    ).get("/approvals/pending")
    assert unauthorized.status_code == 401


def test_staff_decision_persists_audit_identity_and_returns_resumed_response() -> None:
    service = _FakeApprovalService()
    response = _client(service).post(
        f"/approvals/{_APPROVAL_ID}/decision",
        json={"decision": "approved", "decision_note": "Reviewed with clinic staff."},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "approved"
    assert response.json()["response"] == "Your appointment is approved."
    assert service.decisions == [
        (_APPROVAL_ID, "approved", "trusted-clinic-staff", "Reviewed with clinic staff.")
    ]


def test_staff_decision_rejects_blank_audit_note() -> None:
    response = _client(_FakeApprovalService()).post(
        f"/approvals/{_APPROVAL_ID}/decision",
        json={"decision": "rejected", "decision_note": "   "},
    )

    assert response.status_code == 422
