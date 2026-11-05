"""Coordinate durable staff decisions with checkpointed conversation resume."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any, Protocol
from uuid import UUID

from app.repositories.models import Approval, ApprovalDecision


class ApprovalStore(Protocol):
    """Synchronous PostgreSQL operations used by approval workflows."""

    def list_pending(self, *, limit: int = 50) -> list[Approval]: ...

    def decide(
        self,
        approval_id: UUID,
        decision: ApprovalDecision,
        *,
        decided_by: str,
        decision_note: str | None = None,
    ) -> Approval: ...


class ApprovalConversationWorkflow(Protocol):
    """Conversation operation required to continue a staff-paused turn."""

    async def resume_approval(
        self,
        conversation_id: UUID,
        approval_id: UUID,
        decision: ApprovalDecision,
    ) -> Mapping[str, Any]: ...


class ApprovalService:
    """Persist staff decisions and resume the corresponding LangGraph thread."""

    def __init__(
        self,
        repository: ApprovalStore,
        conversations: ApprovalConversationWorkflow,
    ) -> None:
        self._repository = repository
        self._conversations = conversations

    async def list_pending(self, *, limit: int = 50) -> list[Approval]:
        """Return the staff queue without blocking the event loop on PostgreSQL."""
        return await asyncio.to_thread(self._repository.list_pending, limit=limit)

    async def decide(
        self,
        approval_id: UUID,
        decision: ApprovalDecision,
        *,
        decided_by: str,
        decision_note: str,
    ) -> tuple[Approval, Mapping[str, Any]]:
        """Commit the decision, then resume its checkpointed patient conversation.

        Repeating the same decision is safe: repository and workflow operations
        recognize the completed result, allowing a caller to retry if the process
        stopped after the database commit but before returning the response.
        """
        decided_by = decided_by.strip()
        decision_note = decision_note.strip()
        if not decided_by:
            raise ValueError("decided_by cannot be empty.")
        if not decision_note:
            raise ValueError("A decision reason is required.")

        approval = await asyncio.to_thread(
            self._repository.decide,
            approval_id,
            decision,
            decided_by=decided_by,
            decision_note=decision_note,
        )
        state = await self._conversations.resume_approval(
            approval.conversation_id,
            approval.id,
            decision,
        )
        return approval, state
