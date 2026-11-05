"""Coordinate durable approval decisions with their checkpointed conversations."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Mapping
from typing import Any, Literal, Protocol
from uuid import UUID

from app.repositories.errors import (
    ApprovalAlreadyDecided,
    ApprovalContextMismatch,
    ApprovalNotFound,
    ApprovalStateConflict,
    ConversationNotFound,
)
from app.repositories.models import Approval, ApprovalDecision, Booking, Conversation
from app.services.conversations import ConversationProcessingError

DEFAULT_RESUME_TIMEOUT_SECONDS = 30.0


class ApprovalStore(Protocol):
    """Short-lived PostgreSQL operations used by staff approval requests."""

    def get(self, approval_id: UUID) -> Approval | None: ...

    def decide(
        self,
        approval_id: UUID,
        decision: ApprovalDecision,
        *,
        decided_by: str,
        decision_note: str | None = None,
    ) -> Approval: ...

    def list_pending(self, *, limit: int = 50) -> list[Approval]: ...


class BookingStore(Protocol):
    """Read-only lookup needed to replay an interrupted same-day booking."""

    def get(self, booking_id: UUID) -> Booking | None: ...


class ConversationStore(Protocol):
    """Lookup of the thread and trusted owner needed for recovery."""

    def get(self, conversation_id: UUID) -> Conversation | None: ...


class ConversationWorkflow(Protocol):
    """Shared conversation service; it serializes patient and staff thread runs."""

    async def resume_approval(
        self,
        conversation_id: UUID,
        approval_id: UUID,
        decision: ApprovalDecision,
    ) -> Mapping[str, Any]: ...

    async def process_appointment_action(
        self,
        conversation_id: UUID,
        *,
        owner_reference: str,
        request_id: UUID,
        action: Literal["book", "cancel"],
        slot_id: UUID | None = None,
        booking_id: UUID | None = None,
    ) -> Mapping[str, Any]: ...


class ApprovalDecisionConflict(RuntimeError):
    """The approval already has a different decision."""


class ApprovalResumeTimeout(RuntimeError):
    """The decision is durable but the workflow resume timed out."""


class ApprovalService:
    """Persist decisions and recover or resume the associated workflow turn."""

    def __init__(
        self,
        approvals: ApprovalStore,
        bookings: BookingStore,
        conversations: ConversationStore,
        workflow: ConversationWorkflow,
        *,
        resume_timeout_seconds: float = DEFAULT_RESUME_TIMEOUT_SECONDS,
    ) -> None:
        if not math.isfinite(resume_timeout_seconds) or resume_timeout_seconds <= 0:
            raise ValueError("resume_timeout_seconds must be finite and greater than zero.")
        self._approvals = approvals
        self._bookings = bookings
        self._conversations = conversations
        self._workflow = workflow
        self._resume_timeout_seconds = resume_timeout_seconds

    async def list_pending(self, *, limit: int = 50) -> list[Approval]:
        return await asyncio.to_thread(self._approvals.list_pending, limit=limit)

    async def decide(
        self,
        approval_id: UUID,
        decision: ApprovalDecision,
        *,
        decided_by: str,
        decision_note: str,
    ) -> tuple[Approval, str]:
        """Commit a staff decision and make workflow resume retryable.

        The repository updates the approval and booking atomically. The workflow
        resume uses the same per-conversation serializer as patient turns. If a
        process stopped before LangGraph checkpointed the interrupt, replay the
        idempotent appointment action from its durable approval/booking records.
        """
        decided_by = decided_by.strip()
        decision_note = decision_note.strip()
        if not decided_by:
            raise ValueError("decided_by cannot be empty.")
        if not decision_note:
            raise ValueError("A decision reason is required.")
        if decision not in {"approved", "rejected"}:
            raise ValueError("decision must be 'approved' or 'rejected'.")

        approval = await self._get_approval(approval_id)
        if approval.status not in {"pending", decision}:
            raise ApprovalDecisionConflict("The approval was already decided differently.")

        if approval.status == "pending":
            try:
                approval = await asyncio.to_thread(
                    self._approvals.decide,
                    approval_id,
                    decision,
                    decided_by=decided_by,
                    decision_note=decision_note,
                )
            except ApprovalAlreadyDecided:
                approval = await self._get_approval(approval_id)
                if approval.status != decision:
                    raise ApprovalDecisionConflict(
                        "The approval was already decided differently."
                    ) from None

        try:
            state = await asyncio.wait_for(
                self._workflow.resume_approval(
                    approval.conversation_id,
                    approval.id,
                    decision,
                ),
                timeout=self._resume_timeout_seconds,
            )
        except TimeoutError as exc:
            raise ApprovalResumeTimeout(
                "The decision was recorded, but its workflow could not be resumed."
            ) from exc
        except (ApprovalNotFound, ApprovalStateConflict):
            try:
                state = await self._recover_paused_action(approval, decision)
            except (TimeoutError, ConversationProcessingError) as exc:
                raise ApprovalResumeTimeout(
                    "The decision was recorded, but its workflow could not be resumed."
                ) from exc

        response = state.get("response")
        if not isinstance(response, str) or not response.strip():
            response = "The staff decision was recorded."
        return approval, response

    async def _recover_paused_action(
        self,
        approval: Approval,
        decision: ApprovalDecision,
    ) -> Mapping[str, Any]:
        """Recreate a missing checkpoint using the idempotent database action."""
        conversation = await asyncio.to_thread(
            self._conversations.get,
            approval.conversation_id,
        )
        if conversation is None:
            raise ConversationNotFound("Conversation was not found.")
        owner_reference = conversation.owner_reference
        if not isinstance(owner_reference, str) or not owner_reference.strip():
            raise ConversationNotFound("Conversation was not found.")

        if approval.action == "same_day_booking":
            booking = await asyncio.to_thread(self._bookings.get, approval.booking_id)
            if booking is None:
                raise ApprovalContextMismatch("The approval booking was not found.")
            state = await self._workflow.process_appointment_action(
                approval.conversation_id,
                owner_reference=owner_reference,
                request_id=approval.id,
                action="book",
                slot_id=booking.slot_id,
            )
        elif approval.action == "cancel_booking":
            state = await self._workflow.process_appointment_action(
                approval.conversation_id,
                owner_reference=owner_reference,
                request_id=approval.id,
                action="cancel",
                booking_id=approval.booking_id,
            )
        else:
            raise ApprovalContextMismatch("The approval action is not supported.")

        if state.get("workflow_error"):
            raise ApprovalStateConflict("The saved appointment action could not be recovered.")
        if state.get("approval_status") == decision:
            return state
        if state.get("approval_status") == "pending":
            return await asyncio.wait_for(
                self._workflow.resume_approval(approval.conversation_id, approval.id, decision),
                timeout=self._resume_timeout_seconds,
            )
        if approval.action == "same_day_booking":
            expected_status = "confirmed" if decision == "approved" else "rejected"
        else:
            expected_status = "cancelled" if decision == "approved" else "confirmed"
        if state.get("booking_status") == expected_status:
            return state
        raise ApprovalStateConflict("The saved appointment action did not match its decision.")

    async def _get_approval(self, approval_id: UUID) -> Approval:
        approval = await asyncio.to_thread(self._approvals.get, approval_id)
        if approval is None:
            raise ApprovalNotFound("Approval not found.")
        return approval
