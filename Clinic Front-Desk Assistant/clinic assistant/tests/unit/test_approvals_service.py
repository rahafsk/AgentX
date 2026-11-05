"""Unit tests for durable approval decision coordination."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import UUID

import pytest

from app.repositories.errors import ApprovalNotFound
from app.repositories.models import Approval, Booking, Conversation
from app.services.approvals import ApprovalResumeTimeout, ApprovalService

_APPROVAL_ID = UUID("11111111-1111-4111-8111-111111111111")
_BOOKING_ID = UUID("22222222-2222-4222-8222-222222222222")
_CONVERSATION_ID = UUID("33333333-3333-4333-8333-333333333333")
_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _approval(status: Literal["pending", "approved", "rejected"]) -> Approval:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    return Approval(
        id=_APPROVAL_ID,
        booking_id=_BOOKING_ID,
        conversation_id=_CONVERSATION_ID,
        action="cancel_booking",
        status=status,
        requested_at=now,
        decided_at=now if status != "pending" else None,
        decided_by="opaque-staff" if status != "pending" else None,
        decision_note="Fictional review." if status != "pending" else None,
    )


class _Store:
    def __init__(self) -> None:
        self.calls: list[tuple[UUID, str, str, str]] = []
        self.approval = _approval("pending")

    def get(self, approval_id: UUID) -> Approval | None:
        return self.approval if approval_id == self.approval.id else None

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
        self.approval = _approval(decision)
        return self.approval


class _Bookings:
    def get(self, booking_id: UUID) -> Booking | None:
        return None


class _ConversationStore:
    def get(self, conversation_id: UUID) -> Conversation | None:
        return Conversation(
            id=conversation_id,
            thread_id=str(conversation_id),
            created_at=_NOW,
            updated_at=_NOW,
            owner_reference="opaque-owner",
        )


class _Conversations:
    def __init__(self, *, resume_error_once: Exception | None = None) -> None:
        self.calls: list[tuple[UUID, UUID, str]] = []
        self.resume_error_once = resume_error_once
        self.replayed_actions: list[tuple[UUID, str, UUID, UUID | None]] = []

    async def resume_approval(
        self,
        conversation_id: UUID,
        approval_id: UUID,
        decision: Literal["approved", "rejected"],
    ) -> Mapping[str, Any]:
        self.calls.append((conversation_id, approval_id, decision))
        if self.resume_error_once is not None:
            error = self.resume_error_once
            self.resume_error_once = None
            raise error
        return {"response": f"Workflow {decision}."}

    async def process_appointment_action(
        self,
        conversation_id: UUID,
        *,
        owner_reference: str,
        request_id: UUID,
        action: Literal["book", "cancel"],
        slot_id: UUID | None = None,
        booking_id: UUID | None = None,
    ) -> Mapping[str, Any]:
        self.replayed_actions.append((conversation_id, action, request_id, booking_id))
        return {
            "approval_id": str(request_id),
            "approval_status": "approved",
            "booking_status": "cancelled",
            "response": "The cancellation was approved.",
        }


def test_decision_persists_then_resumes_matching_conversation() -> None:
    store = _Store()
    conversations = _Conversations()
    service = ApprovalService(store, _Bookings(), _ConversationStore(), conversations)

    approval, response = asyncio.run(
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
    assert response == "Workflow approved."


def test_pending_approvals_are_read_off_the_event_loop() -> None:
    service = ApprovalService(_Store(), _Bookings(), _ConversationStore(), _Conversations())

    approvals = asyncio.run(service.list_pending(limit=1))

    assert [approval.status for approval in approvals] == ["pending"]


def test_saved_decision_reconstructs_a_missing_workflow_pause() -> None:
    store = _Store()
    conversations = _Conversations(resume_error_once=ApprovalNotFound())
    service = ApprovalService(store, _Bookings(), _ConversationStore(), conversations)

    approval, response = asyncio.run(
        service.decide(
            _APPROVAL_ID,
            "approved",
            decided_by="staff",
            decision_note="Reviewed.",
        )
    )

    assert approval.status == "approved"
    assert response == "The cancellation was approved."
    assert conversations.replayed_actions == [
        (_CONVERSATION_ID, "cancel", _APPROVAL_ID, _BOOKING_ID)
    ]


def test_resume_timeout_keeps_the_recorded_decision_retryable() -> None:
    class _SlowWorkflow(_Conversations):
        async def resume_approval(
            self,
            conversation_id: UUID,
            approval_id: UUID,
            decision: Literal["approved", "rejected"],
        ) -> Mapping[str, Any]:
            await asyncio.sleep(0.05)
            return {"response": "Late response."}

    store = _Store()
    service = ApprovalService(
        store,
        _Bookings(),
        _ConversationStore(),
        _SlowWorkflow(),
        resume_timeout_seconds=0.001,
    )

    with pytest.raises(ApprovalResumeTimeout):
        asyncio.run(
            service.decide(
                _APPROVAL_ID,
                "approved",
                decided_by="staff",
                decision_note="Reviewed.",
            )
        )

    assert store.approval.status == "approved"
