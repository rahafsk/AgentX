"""Tests for approval decision recovery, timeout, and shared locking."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import pytest
from langgraph.types import Command

from app.graphs.workflows import APPROVAL_HANDLER
from app.repositories.errors import (
    ApprovalAlreadyDecided,
    ApprovalContextMismatch,
)
from app.repositories.models import Approval, Conversation
from app.services.approvals import (
    ApprovalDecisionConflict,
    ApprovalResumeTimeout,
    ApprovalService,
)
from app.services.conversations import ConversationService

_NOW = datetime(2026, 1, 1, tzinfo=UTC)
_OWNER = "owner-1"


def _approval(
    conversation_id: UUID,
    *,
    status: str = "pending",
) -> Approval:
    return Approval(
        id=uuid4(),
        booking_id=uuid4(),
        conversation_id=conversation_id,
        action="cancel_booking",
        status=status,  # type: ignore[arg-type]
        requested_at=_NOW,
        decided_at=None if status == "pending" else _NOW,
        decided_by=None if status == "pending" else "staff",
        decision_note=None if status == "pending" else "Reviewed",
    )


def _conversation(conversation_id: UUID) -> Conversation:
    return Conversation(
        id=conversation_id,
        thread_id=str(conversation_id),
        created_at=_NOW,
        updated_at=_NOW,
        owner_reference=_OWNER,
    )


class _ApprovalRepository:
    def __init__(self, approvals: list[Approval]) -> None:
        self.approvals = {approval.id: approval for approval in approvals}
        self.decide_calls = 0

    def get(self, approval_id: UUID) -> Approval | None:
        return self.approvals.get(approval_id)

    def list_pending(self, *, limit: int = 50) -> list[Approval]:
        return [item for item in self.approvals.values() if item.status == "pending"][:limit]

    def decide(
        self,
        approval_id: UUID,
        decision: str,
        *,
        decided_by: str,
        decision_note: str | None = None,
    ) -> Approval:
        self.decide_calls += 1
        approval = self.approvals[approval_id]
        if approval.status != "pending":
            raise ApprovalAlreadyDecided()

        updated = replace(
            approval,
            status=decision,  # type: ignore[arg-type]
            decided_at=_NOW,
            decided_by=decided_by,
            decision_note=decision_note,
        )
        self.approvals[approval_id] = updated
        return updated


class _ConversationRepository:
    def __init__(self, conversations: list[Conversation]) -> None:
        self.conversations = {conversation.id: conversation for conversation in conversations}

    def get(self, conversation_id: UUID) -> Conversation | None:
        return self.conversations.get(conversation_id)


class _Snapshot:
    def __init__(
        self,
        values: Mapping[str, Any],
        next_nodes: tuple[str, ...] = (),
    ) -> None:
        self.values = values
        self.next = next_nodes


class _Workflow:
    def __init__(
        self,
        snapshots: Mapping[str, _Snapshot],
        *,
        delay: float = 0,
        barrier: asyncio.Barrier | None = None,
    ) -> None:
        self.snapshots = snapshots
        self.delay = delay
        self.barrier = barrier
        self.calls: list[tuple[Command[Any], Mapping[str, Any]]] = []
        self.active = 0
        self.max_active = 0
        self.started = asyncio.Event()

    async def aget_state(self, config: Mapping[str, Any]) -> _Snapshot:
        thread_id = config["configurable"]["thread_id"]
        return self.snapshots[thread_id]

    async def ainvoke(
        self,
        input: Any,
        config: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        if config is None:
            raise AssertionError("config is required.")
        self.calls.append((input, config))
        self.started.set()
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if self.barrier is not None:
                await asyncio.wait_for(self.barrier.wait(), timeout=1)
            if self.delay:
                await asyncio.sleep(self.delay)
            return {"response": "Workflow resumed."}
        finally:
            self.active -= 1


class _NoopSerializer:
    @asynccontextmanager
    async def serialize_conversation(
        self,
        conversation_id: UUID,
    ) -> AsyncIterator[None]:
        yield


class _ConversationServiceRepository:
    def __init__(self, conversation: Conversation) -> None:
        self.conversation = conversation

    def create(self, owner_reference: str) -> Conversation:
        return self.conversation

    def get_for_owner(
        self,
        conversation_id: UUID,
        owner_reference: str,
    ) -> Conversation | None:
        if (
            conversation_id == self.conversation.id
            and owner_reference == self.conversation.owner_reference
        ):
            return self.conversation
        return None

    def touch_for_owner(
        self,
        conversation_id: UUID,
        owner_reference: str,
    ) -> Conversation | None:
        return self.get_for_owner(conversation_id, owner_reference)


def _pending_snapshot(conversation_id: UUID) -> _Snapshot:
    return _Snapshot(
        {
            "conversation_id": str(conversation_id),
            "approval_status": "pending",
            "response": "",
        },
        (APPROVAL_HANDLER,),
    )


def _service(
    approvals: _ApprovalRepository,
    conversations: _ConversationRepository,
    workflow: _Workflow,
    *,
    locks: Any | None = None,
    timeout: float = 1,
) -> ApprovalService:
    return ApprovalService(
        approvals,
        conversations,
        workflow,
        locks or _NoopSerializer(),
        resume_timeout_seconds=timeout,
    )


async def test_timeout_keeps_saved_decision() -> None:
    conversation_id = uuid4()
    approval = _approval(conversation_id)
    approvals = _ApprovalRepository([approval])
    conversations = _ConversationRepository([_conversation(conversation_id)])
    workflow = _Workflow(
        {str(conversation_id): _pending_snapshot(conversation_id)},
        delay=0.1,
    )
    service = _service(approvals, conversations, workflow, timeout=0.01)

    with pytest.raises(ApprovalResumeTimeout):
        await service.decide(
            approval.id,
            "approved",
            decided_by="staff",
            decision_note="Reviewed",
        )

    assert approvals.decide_calls == 1
    assert approvals.approvals[approval.id].status == "approved"


async def test_matching_recorded_decision_recovers_without_deciding_again() -> None:
    conversation_id = uuid4()
    approval = _approval(conversation_id, status="approved")
    approvals = _ApprovalRepository([approval])
    conversations = _ConversationRepository([_conversation(conversation_id)])
    workflow = _Workflow({str(conversation_id): _pending_snapshot(conversation_id)})
    service = _service(approvals, conversations, workflow)

    result, response = await service.decide(
        approval.id,
        "approved",
        decided_by="staff",
        decision_note="Reviewed",
    )

    assert result.status == "approved"
    assert response == "Workflow resumed."
    assert approvals.decide_calls == 0
    assert len(workflow.calls) == 1


async def test_conflicting_retry_is_rejected() -> None:
    conversation_id = uuid4()
    approval = _approval(conversation_id, status="rejected")
    approvals = _ApprovalRepository([approval])
    conversations = _ConversationRepository([_conversation(conversation_id)])
    workflow = _Workflow({str(conversation_id): _pending_snapshot(conversation_id)})
    service = _service(approvals, conversations, workflow)

    with pytest.raises(ApprovalDecisionConflict):
        await service.decide(
            approval.id,
            "approved",
            decided_by="staff",
            decision_note="Reviewed",
        )

    assert approvals.decide_calls == 0
    assert workflow.calls == []


async def test_conversation_context_mismatch_is_rejected() -> None:
    conversation_id = uuid4()
    other_conversation_id = uuid4()
    approval = _approval(conversation_id, status="approved")
    approvals = _ApprovalRepository([approval])
    conversations = _ConversationRepository([_conversation(conversation_id)])
    workflow = _Workflow({str(conversation_id): _pending_snapshot(other_conversation_id)})
    service = _service(approvals, conversations, workflow)

    with pytest.raises(ApprovalContextMismatch):
        await service.decide(
            approval.id,
            "approved",
            decided_by="staff",
            decision_note="Reviewed",
        )

    assert workflow.calls == []


async def test_patient_message_and_decision_share_conversation_lock() -> None:
    conversation_id = uuid4()
    conversation = _conversation(conversation_id)
    approval = _approval(conversation_id)
    approvals = _ApprovalRepository([approval])
    conversations = _ConversationRepository([conversation])
    workflow = _Workflow(
        {conversation.thread_id: _pending_snapshot(conversation_id)},
        delay=0.05,
    )
    conversation_service = ConversationService(
        _ConversationServiceRepository(conversation),
        workflow,
    )
    approval_service = _service(
        approvals,
        conversations,
        workflow,
        locks=conversation_service,
    )

    patient = asyncio.create_task(
        conversation_service.process_message(
            conversation_id,
            "I need to cancel my appointment.",
            owner_reference=_OWNER,
            request_id=uuid4(),
        )
    )
    await asyncio.wait_for(workflow.started.wait(), timeout=1)

    staff = asyncio.create_task(
        approval_service.decide(
            approval.id,
            "approved",
            decided_by="staff",
            decision_note="Reviewed",
        )
    )
    await asyncio.sleep(0.01)

    assert approvals.decide_calls == 0

    await asyncio.gather(patient, staff)

    assert approvals.decide_calls == 1
    assert workflow.max_active == 1


async def test_different_conversations_run_independently() -> None:
    first_conversation_id = uuid4()
    second_conversation_id = uuid4()
    first_conversation = _conversation(first_conversation_id)
    second_conversation = _conversation(second_conversation_id)
    first_approval = _approval(first_conversation_id)
    second_approval = _approval(second_conversation_id)
    approvals = _ApprovalRepository([first_approval, second_approval])
    conversations = _ConversationRepository([first_conversation, second_conversation])
    workflow = _Workflow(
        {
            first_conversation.thread_id: _pending_snapshot(first_conversation_id),
            second_conversation.thread_id: _pending_snapshot(second_conversation_id),
        },
        barrier=asyncio.Barrier(2),
    )
    conversation_service = ConversationService(
        _ConversationServiceRepository(first_conversation),
        workflow,
    )
    approval_service = _service(
        approvals,
        conversations,
        workflow,
        locks=conversation_service,
    )

    await asyncio.gather(
        approval_service.decide(
            first_approval.id,
            "approved",
            decided_by="staff",
            decision_note="Reviewed",
        ),
        approval_service.decide(
            second_approval.id,
            "approved",
            decided_by="staff",
            decision_note="Reviewed",
        ),
    )

    assert workflow.max_active == 2
