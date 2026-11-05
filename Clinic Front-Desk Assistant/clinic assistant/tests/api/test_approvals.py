"""API tests for staff approval decisions."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.approvals import router
from app.repositories.errors import ApprovalNotFound
from app.repositories.models import Approval
from app.services.approvals import (
    ApprovalDecisionConflict,
    ApprovalResumeTimeout,
)

_APPROVAL_ID = UUID("11111111-1111-1111-1111-111111111111")
_BOOKING_ID = UUID("22222222-2222-2222-2222-222222222222")
_CONVERSATION_ID = UUID("33333333-3333-3333-3333-333333333333")
_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _approval(*, status: str = "pending") -> Approval:
    return Approval(
        id=_APPROVAL_ID,
        booking_id=_BOOKING_ID,
        conversation_id=_CONVERSATION_ID,
        action="cancel_booking",
        status=status,  # type: ignore[arg-type]
        requested_at=_NOW,
        decided_at=None,
        decided_by=None,
        decision_note=None,
    )


class _FakeApprovalService:
    def __init__(
        self,
        *,
        pending: list[Approval] | None = None,
        result: tuple[Approval, str] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.pending = pending or []
        self.result = result or (_approval(status="approved"), "The decision was applied.")
        self.error = error
        self.decisions: list[tuple[UUID, str, str, str]] = []

    async def list_pending(self, *, limit: int) -> list[Approval]:
        return self.pending[:limit]

    async def decide(
        self,
        approval_id: UUID,
        decision: str,
        *,
        decided_by: str,
        decision_note: str,
    ) -> tuple[Approval, str]:
        self.decisions.append((approval_id, decision, decided_by, decision_note))
        if self.error is not None:
            raise self.error
        return self.result


def _client(service: _FakeApprovalService | None) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    if service is not None:
        app.state.approval_service = service
    return TestClient(app)


def test_list_pending_approvals() -> None:
    client = _client(_FakeApprovalService(pending=[_approval()]))

    response = client.get("/approvals")

    assert response.status_code == 200
    assert response.json()[0]["id"] == str(_APPROVAL_ID)
    assert response.json()[0]["status"] == "pending"


@pytest.mark.parametrize("decision", ["approved", "rejected"])
def test_decision_returns_resumed_response(decision: str) -> None:
    service = _FakeApprovalService(result=(_approval(status=decision), "The decision was applied."))
    client = _client(service)

    response = client.post(
        f"/approvals/{_APPROVAL_ID}/decision",
        json={
            "decision": decision,
            "decided_by": "staff-1",
            "decision_note": "Reviewed",
        },
    )

    assert response.status_code == 200
    assert response.json()["response"] == "The decision was applied."
    assert service.decisions == [(_APPROVAL_ID, decision, "staff-1", "Reviewed")]


@pytest.mark.parametrize(
    ("error", "status_code"),
    [
        (ApprovalNotFound(), 404),
        (ApprovalDecisionConflict(), 409),
        (ApprovalResumeTimeout(), 504),
    ],
)
def test_service_errors_are_mapped_safely(
    error: Exception,
    status_code: int,
) -> None:
    client = _client(_FakeApprovalService(error=error))

    response = client.post(
        f"/approvals/{_APPROVAL_ID}/decision",
        json={
            "decision": "approved",
            "decided_by": "staff-1",
            "decision_note": "Reviewed",
        },
    )

    assert response.status_code == status_code


def test_resume_timeout_explains_decision_was_recorded() -> None:
    client = _client(_FakeApprovalService(error=ApprovalResumeTimeout()))

    response = client.post(
        f"/approvals/{_APPROVAL_ID}/decision",
        json={
            "decision": "approved",
            "decided_by": "staff-1",
            "decision_note": "Reviewed",
        },
    )

    assert response.status_code == 504
    assert "decision was recorded" in response.json()["detail"]
    assert "rolled back" not in response.json()["detail"]


def test_workflow_failure_returns_500() -> None:
    client = _client(_FakeApprovalService(error=RuntimeError("workflow failed")))

    response = client.post(
        f"/approvals/{_APPROVAL_ID}/decision",
        json={
            "decision": "approved",
            "decided_by": "staff-1",
            "decision_note": "Reviewed",
        },
    )

    assert response.status_code == 500


def test_invalid_decision_and_blank_note_are_rejected() -> None:
    client = _client(_FakeApprovalService())

    invalid_decision = client.post(
        f"/approvals/{_APPROVAL_ID}/decision",
        json={
            "decision": "maybe",
            "decided_by": "staff-1",
            "decision_note": "Reviewed",
        },
    )
    blank_note = client.post(
        f"/approvals/{_APPROVAL_ID}/decision",
        json={
            "decision": "approved",
            "decided_by": "staff-1",
            "decision_note": "   ",
        },
    )

    assert invalid_decision.status_code == 422
    assert blank_note.status_code == 422


def test_missing_service_state_returns_503() -> None:
    response = _client(None).get("/approvals")

    assert response.status_code == 503
