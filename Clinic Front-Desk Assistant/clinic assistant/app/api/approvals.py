"""Staff endpoints for listing and deciding pending appointment approvals."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Annotated, Any, Literal, Protocol, cast
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field, field_validator

from app.api.dependencies import StaffReferenceDependency
from app.repositories.errors import (
    ApprovalAlreadyDecided,
    ApprovalContextMismatch,
    ApprovalNotFound,
    ApprovalStateConflict,
    ConversationNotFound,
)
from app.repositories.models import Approval, ApprovalAction, ApprovalStatus
from app.services.conversations import ConversationProcessingError

router = APIRouter(prefix="/approvals", tags=["approvals"])


class ApprovalServiceProtocol(Protocol):
    """Service operations required by the staff approvals API."""

    async def list_pending(self, *, limit: int = 50) -> list[Approval]: ...

    async def decide(
        self,
        approval_id: UUID,
        decision: Literal["approved", "rejected"],
        *,
        decided_by: str,
        decision_note: str,
    ) -> tuple[Approval, Mapping[str, Any]]: ...


class ApprovalDecisionRequest(BaseModel):
    """An authenticated staff decision and required audit explanation."""

    decision: Literal["approved", "rejected"]
    decision_note: str = Field(min_length=1, max_length=1000)

    @field_validator("decision_note")
    @classmethod
    def reject_blank_note(cls, value: str) -> str:
        note = value.strip()
        if not note:
            raise ValueError("A decision reason is required.")
        return note


class ApprovalResponse(BaseModel):
    """A pending approval or the result of a staff decision."""

    id: UUID
    booking_id: UUID
    conversation_id: UUID
    action: ApprovalAction
    status: ApprovalStatus
    requested_at: datetime
    decided_at: datetime | None
    response: str | None = None

    @classmethod
    def from_approval(
        cls,
        approval: Approval,
        *,
        response: str | None = None,
    ) -> ApprovalResponse:
        return cls(
            id=approval.id,
            booking_id=approval.booking_id,
            conversation_id=approval.conversation_id,
            action=approval.action,
            status=approval.status,
            requested_at=approval.requested_at,
            decided_at=approval.decided_at,
            response=response,
        )


def get_approval_service(request: Request) -> ApprovalServiceProtocol:
    """Return the configured staff-approval service."""
    service = getattr(request.app.state, "approval_service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Approval service is not configured.",
        )
    return cast(ApprovalServiceProtocol, service)


ApprovalServiceDependency = Annotated[
    ApprovalServiceProtocol,
    Depends(get_approval_service),
]


@router.get("/pending", response_model=list[ApprovalResponse])
async def list_pending_approvals(
    service: ApprovalServiceDependency,
    staff_reference: StaffReferenceDependency,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
) -> list[ApprovalResponse]:
    """List pending approval work for authenticated clinic staff."""
    del staff_reference
    approvals = await service.list_pending(limit=limit)
    return [ApprovalResponse.from_approval(item) for item in approvals]


@router.post("/{approval_id}/decision", response_model=ApprovalResponse)
async def decide_approval(
    approval_id: UUID,
    payload: ApprovalDecisionRequest,
    service: ApprovalServiceDependency,
    staff_reference: StaffReferenceDependency,
) -> ApprovalResponse:
    """Persist an audited staff decision and resume its paused graph thread."""
    try:
        approval, state = await service.decide(
            approval_id,
            payload.decision,
            decided_by=staff_reference,
            decision_note=payload.decision_note,
        )
    except ApprovalNotFound as exc:
        raise HTTPException(status_code=404, detail="Approval not found.") from exc
    except ConversationNotFound as exc:
        raise HTTPException(status_code=404, detail="Conversation not found.") from exc
    except (ApprovalAlreadyDecided, ApprovalContextMismatch, ApprovalStateConflict) as exc:
        raise HTTPException(status_code=409, detail="The approval cannot be decided.") from exc
    except ConversationProcessingError as exc:
        raise HTTPException(
            status_code=503,
            detail="The decision was saved; retry the same decision to resume the conversation.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    response_text = state.get("response")
    if not isinstance(response_text, str) or not response_text.strip():
        response_text = "The staff decision was recorded."
    return ApprovalResponse.from_approval(approval, response=response_text)
