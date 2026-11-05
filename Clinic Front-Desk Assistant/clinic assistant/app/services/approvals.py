"""Coordinate durable staff approval decisions and workflow resume."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Mapping
from contextlib import AbstractAsyncContextManager
from typing import Any, Protocol
from uuid import UUID

from langgraph.types import Command

from app.graphs.workflows import APPROVAL_HANDLER
from app.repositories.errors import (
    ApprovalAlreadyDecided,
    ApprovalContextMismatch,
    ApprovalNotFound,
    ConversationNotFound,
)
from app.repositories.models import (
    Approval,
    ApprovalDecision,
    Conversation,
)

DEFAULT_RESUME_TIMEOUT_SECONDS = 30.0


class ApprovalStore(Protocol):
    """Persistence operations needed to decide an approval."""

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


class ConversationStore(Protocol):
    """Conversation lookup used to resolve the LangGraph thread ID."""

    def get(self, conversation_id: UUID) -> Conversation | None: ...


class WorkflowSnapshot(Protocol):
    """Read-only workflow checkpoint state needed for resume validation."""

    values: Mapping[str, Any]
    next: tuple[str, ...]


class ApprovalWorkflow(Protocol):
    """Workflow operations used by the approval service."""

    async def aget_state(self, config: Any) -> WorkflowSnapshot: ...

    async def ainvoke(
        self,
        input: Command[Any],
        config: Any,
    ) -> Mapping[str, Any]: ...


class ConversationSerializer(Protocol):
    """Shared per-conversation lock used by patient and staff operations."""

    def serialize_conversation(
        self,
        conversation_id: UUID,
    ) -> AbstractAsyncContextManager[None]: ...


class ApprovalDecisionConflict(RuntimeError):
    """The saved decision conflicts with the requested decision."""


class ApprovalResumeTimeout(RuntimeError):
    """The decision was saved, but the workflow resume did not finish."""


class ApprovalService:
    """List approvals, record decisions, and resume their workflows safely."""

    def __init__(
        self,
        approvals: ApprovalStore,
        conversations: ConversationStore,
        workflow: ApprovalWorkflow,
        conversation_locks: ConversationSerializer,
        *,
        resume_timeout_seconds: float = DEFAULT_RESUME_TIMEOUT_SECONDS,
    ) -> None:
        if not math.isfinite(resume_timeout_seconds) or resume_timeout_seconds <= 0:
            raise ValueError("resume_timeout_seconds must be finite and greater than zero.")

        self._approvals = approvals
        self._conversations = conversations
        self._workflow = workflow
        self._conversation_locks = conversation_locks
        self._resume_timeout_seconds = resume_timeout_seconds

    async def list_pending(self, *, limit: int) -> list[Approval]:
        return await asyncio.to_thread(self._approvals.list_pending, limit=limit)

    async def decide(
        self,
        approval_id: UUID,
        decision: ApprovalDecision,
        *,
        decided_by: str,
        decision_note: str,
    ) -> tuple[Approval, str]:
        approval = await self._get_approval(approval_id)
        if approval.status != "pending" and approval.status != decision:
            raise ApprovalDecisionConflict("The approval was already decided differently.")

        expected_conversation_id = approval.conversation_id
        async with self._conversation_locks.serialize_conversation(expected_conversation_id):
            approval = await self._get_approval(approval_id)

            if approval.conversation_id != expected_conversation_id:
                raise ApprovalContextMismatch("The approval conversation changed.")

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
            elif approval.status != decision:
                raise ApprovalDecisionConflict("The approval was already decided differently.")

            conversation = await asyncio.to_thread(
                self._conversations.get,
                approval.conversation_id,
            )
            if conversation is None:
                raise ConversationNotFound("Conversation was not found.")

            config = {"configurable": {"thread_id": conversation.thread_id}}
            snapshot = await self._workflow.aget_state(config)
            values = snapshot.values

            if values.get("conversation_id") != str(approval.conversation_id):
                raise ApprovalContextMismatch(
                    "The workflow does not belong to this approval conversation."
                )

            is_pending = (
                values.get("approval_status") == "pending" or APPROVAL_HANDLER in snapshot.next
            )
            if not is_pending:
                if values.get("approval_status") == decision:
                    response = values.get("response")
                    if isinstance(response, str) and response.strip():
                        return approval, response
                raise ApprovalDecisionConflict("The workflow is not waiting for this approval.")

            try:
                async with asyncio.timeout(self._resume_timeout_seconds):
                    result = await self._workflow.ainvoke(
                        Command(resume={"decision": decision}),
                        config,
                    )
            except TimeoutError as exc:
                raise ApprovalResumeTimeout(
                    "The decision was recorded, but the workflow resume did not complete."
                ) from exc

            response = result.get("response")
            if not isinstance(response, str) or not response.strip():
                response = "The staff decision was recorded."

            return approval, response

    async def _get_approval(self, approval_id: UUID) -> Approval:
        approval = await asyncio.to_thread(self._approvals.get, approval_id)
        if approval is None:
            raise ApprovalNotFound("Approval not found.")
        return approval
