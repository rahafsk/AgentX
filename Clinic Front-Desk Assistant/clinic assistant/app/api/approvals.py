"""HTTP endpoints for listing and deciding staff approvals."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, cast
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field, field_validator

from app.repositories.errors import (
    ApprovalAlreadyDecided,
    ApprovalContextMismatch,
    ApprovalNotFound,
    ApprovalStateConflict,
    ConversationNotFound,
)
from app.repositories.models import (
    Approval,
    ApprovalAction,
    ApprovalDecision,
    ApprovalStatus,
)
from app.services.approvals import (
    ApprovalDecisionConflict,
    ApprovalResumeTimeout,
    ApprovalService,
)

router = APIRouter(prefix="/approvals", tags=["approvals"])


class ApprovalResponse(BaseModel):
    """A staff approval item."""

    id: UUID
    booking_id: UUID
    conversation_id: UUID
    action: ApprovalAction
    status: ApprovalStatus
    requested_at: datetime
    decided_at: datetime | None
    decided_by: str | None
    decision_note: str | None

    @classmethod
    def from_approval(cls, approval: Approval) -> ApprovalResponse:
        return cls(
            id=approval.id,
            booking_id=approval.booking_id,
            conversation_id=approval.conversation_id,
            action=approval.action,
            status=approval.status,
            requested_at=approval.requested_at,
            decided_at=approval.decided_at,
            decided_by=approval.decided_by,
            decision_note=approval.decision_note,
        )


class ApprovalDecisionRequest(BaseModel):
    """A staff decision for a pending approval."""

    decision: ApprovalDecision
    decided_by: str = Field(min_length=1, max_length=200)
    decision_note: str = Field(min_length=1, max_length=1000)

    @field_validator("decided_by", "decision_note")
    @classmethod
    def reject_blank_values(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class ApprovalDecisionResponse(BaseModel):
    """The decided approval and the resumed workflow response."""

    approval: ApprovalResponse
    response: str


def get_approval_service(request: Request) -> ApprovalService:
    """Return the configured approval service from application state."""
    service = getattr(request.app.state, "approval_service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Approval service is not configured.",
        )
    return cast(ApprovalService, service)


ApprovalServiceDependency = Annotated[
    ApprovalService,
    Depends(get_approval_service),
]


@router.get("", response_model=list[ApprovalResponse])
async def list_pending_approvals(
    service: ApprovalServiceDependency,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
) -> list[ApprovalResponse]:
    """List pending staff approvals."""
    pending = await service.list_pending(limit=limit)
    return [ApprovalResponse.from_approval(approval) for approval in pending]


@router.post("/{approval_id}/decision", response_model=ApprovalDecisionResponse)
async def decide_approval(
    approval_id: UUID,
    payload: ApprovalDecisionRequest,
    service: ApprovalServiceDependency,
) -> ApprovalDecisionResponse:
    """Approve or reject a pending request and resume its workflow."""
    try:
        approval, response = await service.decide(
            approval_id,
            payload.decision,
            decided_by=payload.decided_by,
            decision_note=payload.decision_note,
        )
    except ApprovalNotFound as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Approval not found.",
        ) from exc
    except (
        ApprovalAlreadyDecided,
        ApprovalDecisionConflict,
        ApprovalStateConflict,
        ApprovalContextMismatch,
        ConversationNotFound,
    ) as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="The approval can no longer be decided.",
        ) from exc
    except ApprovalResumeTimeout as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=(
                "The decision was recorded, but the workflow resume did not complete. "
                "Retry the same decision."
            ),
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The approval decision could not be completed.",
        ) from exc

    return ApprovalDecisionResponse(
        approval=ApprovalResponse.from_approval(approval),
        response=response,
    )
