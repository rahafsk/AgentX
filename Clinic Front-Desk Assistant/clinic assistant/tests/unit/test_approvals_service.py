"""Unit tests for durable approval decision coordination."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import UUID

from app.repositories.models import Approval
from app.services.approvals import ApprovalService

_APPROVAL_ID = UUID("11111111-1111-4111-8111-111111111111")
_BOOKING_ID = UUID("22222222-2222-4222-8222-222222222222")
_CONVERSATION_ID = UUID("33333333-3333-4333-8333-333333333333")


def _approval(status: Literal["pending", "approved", "rejected"]) -> Approval:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    return Approval(
        id=_APPROVAL_ID,
        booking_id=_BOOKING_ID,
        conversation_id=_CONVERSATION_ID,
        action="same_day_booking",
        status=status,
        requested_at=now,
        decided_at=now if status != "pending" else None,
        decided_by="opaque-staff" if status != "pending" else None,
        decision_note="Fictional review." if status != "pending" else None,
    )


class _Store:
    def __init__(self) -> None:
        self.calls: list[tuple[UUID, str, str, str]] = []

    def list_pending(self, *, limit: int = 50) -> list[Approval]:
        return [_approval("pending")][:limit]

    def decide(
        self,
        approval_id: UUID,
        decision: Literal["approved", "rejected"],
        *,
        decided_by: str,
        decision_note: str | None = None,
    ) -> Approval:
        self.calls.append((approval_id, decision, decided_by, str(decision_note)))
        return _approval(decision)


class _Conversations:
    def __init__(self) -> None:
        self.calls: list[tuple[UUID, UUID, str]] = []

    async def resume_approval(
        self,
        conversation_id: UUID,
        approval_id: UUID,
        decision: Literal["approved", "rejected"],
    ) -> Mapping[str, Any]:
        self.calls.append((conversation_id, approval_id, decision))
        return {"response": f"Workflow {decision}."}


def test_decision_persists_then_resumes_matching_conversation() -> None:
    store = _Store()
    conversations = _Conversations()
    service = ApprovalService(store, conversations)

    approval, state = asyncio.run(
        service.decide(
            _APPROVAL_ID,
            "approved",
            decided_by=" opaque-staff ",
            decision_note=" Fictional review. ",
        )
    )

    assert approval.status == "approved"
    assert store.calls == [(_APPROVAL_ID, "approved", "opaque-staff", "Fictional review.")]
    assert conversations.calls == [(_CONVERSATION_ID, _APPROVAL_ID, "approved")]
    assert state == {"response": "Workflow approved."}


def test_pending_approvals_are_read_off_the_event_loop() -> None:
    service = ApprovalService(_Store(), _Conversations())

    approvals = asyncio.run(service.list_pending(limit=1))

    assert [approval.status for approval in approvals] == ["pending"]
