"""Authenticated staff endpoints for pending appointment approvals."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, cast
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.api.dependencies import StaffReferenceDependency
from app.repositories.errors import (
    ApprovalAlreadyDecided,
    ApprovalContextMismatch,
    ApprovalNotFound,
    ApprovalStateConflict,
    ConversationNotFound,
)
from app.repositories.models import Approval, ApprovalAction, ApprovalDecision, ApprovalStatus
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
            decided_by=approval.decided_by,
            decision_note=approval.decision_note,
            response=response,
        )


class ApprovalDecisionResponse(BaseModel):
    """A persisted staff decision and the resumed patient-facing response."""

    approval: ApprovalResponse
    response: str


class ApprovalDecisionRequest(BaseModel):
    """A staff decision and required audit explanation."""

    model_config = ConfigDict(extra="forbid")

    decision: ApprovalDecision
    decision_note: str = Field(min_length=1, max_length=1000)

    @field_validator("decision_note")
    @classmethod
    def reject_blank_note(cls, value: str) -> str:
        note = value.strip()
        if not note:
            raise ValueError("A decision reason is required.")
        return note


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
@router.get("/pending", response_model=list[ApprovalResponse])
async def list_pending_approvals(
    service: ApprovalServiceDependency,
    staff_reference: StaffReferenceDependency,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
) -> list[ApprovalResponse]:
    """List pending approvals for authenticated clinic staff."""
    del staff_reference
    approvals = await service.list_pending(limit=limit)
    return [ApprovalResponse.from_approval(approval) for approval in approvals]


@router.post("/{approval_id}/decision", response_model=ApprovalDecisionResponse)
async def decide_approval(
    approval_id: UUID,
    payload: ApprovalDecisionRequest,
    service: ApprovalServiceDependency,
    staff_reference: StaffReferenceDependency,
) -> ApprovalDecisionResponse:
    """Persist the authenticated staff decision and resume its graph thread."""
    try:
        approval, response = await service.decide(
            approval_id,
            payload.decision,
            decided_by=staff_reference,
            decision_note=payload.decision_note,
        )
    except ApprovalNotFound as exc:
        raise HTTPException(status_code=404, detail="Approval not found.") from exc
    except ConversationNotFound as exc:
        raise HTTPException(status_code=404, detail="Conversation not found.") from exc
    except (
        ApprovalAlreadyDecided,
        ApprovalDecisionConflict,
        ApprovalContextMismatch,
        ApprovalStateConflict,
    ) as exc:
        raise HTTPException(status_code=409, detail="The approval cannot be decided.") from exc
    except ApprovalResumeTimeout as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="The decision was recorded; retry the same decision to resume the workflow.",
        ) from exc
    except TimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="The decision was recorded; retry the same decision to resume the workflow.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail="The approval decision could not be completed; retry the same decision.",
        ) from exc

    return ApprovalDecisionResponse(
        approval=ApprovalResponse.from_approval(approval),
        response=response,
    )
