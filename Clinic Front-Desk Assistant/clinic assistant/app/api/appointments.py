"""HTTP endpoints for slot search and checkpointed appointment actions."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import datetime
from typing import Annotated, Literal, Protocol, cast
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from pydantic import BaseModel, ConfigDict

from app.api.dependencies import IdempotencyKey, OwnerReferenceDependency
from app.repositories.appointments import MAX_SEARCH_LIMIT
from app.repositories.errors import ConversationNotFound
from app.repositories.models import ApprovalStatus, AvailableAppointmentSlot, BookingStatus
from app.services.appointments import DEFAULT_SEARCH_LIMIT
from app.services.conversations import ConversationProcessingError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/appointments", tags=["appointments"])


class AppointmentServiceProtocol(Protocol):
    """Read-only slot-search operation used directly by this HTTP router."""

    def search_available_slots(
        self,
        starts_at: datetime,
        ends_at: datetime,
        *,
        doctor_id: UUID | None = None,
        limit: int = DEFAULT_SEARCH_LIMIT,
    ) -> list[AvailableAppointmentSlot]: ...


class AppointmentWorkflowProtocol(Protocol):
    """Checkpointed booking/cancellation operations routed through LangGraph."""

    async def process_appointment_action(
        self,
        conversation_id: UUID,
        *,
        owner_reference: str,
        request_id: UUID,
        action: Literal["book", "cancel"],
        slot_id: UUID | None = None,
        booking_id: UUID | None = None,
    ) -> Mapping[str, object]: ...


class AvailableSlotResponse(BaseModel):
    """An open appointment slot returned to the chat client."""

    id: UUID
    doctor_id: UUID
    doctor_name: str
    doctor_specialty: str
    starts_at: datetime
    ends_at: datetime

    @classmethod
    def from_slot(cls, slot: AvailableAppointmentSlot) -> AvailableSlotResponse:
        return cls(
            id=slot.id,
            doctor_id=slot.doctor_id,
            doctor_name=slot.doctor_name,
            doctor_specialty=slot.doctor_specialty,
            starts_at=slot.starts_at,
            ends_at=slot.ends_at,
        )


class BookingRequest(BaseModel):
    """A slot request; patient identity comes from authentication, not JSON."""

    model_config = ConfigDict(extra="forbid")

    slot_id: UUID
    conversation_id: UUID


class BookingResponse(BaseModel):
    """A booking outcome returned after its workflow turn is checkpointed."""

    id: UUID
    slot_id: UUID
    conversation_id: UUID
    status: BookingStatus
    response: str


class CancellationRequest(BaseModel):
    """A request to cancel an existing booking in its conversation."""

    model_config = ConfigDict(extra="forbid")

    booking_id: UUID
    conversation_id: UUID


class ApprovalResponse(BaseModel):
    """A checkpointed cancellation approval request."""

    id: UUID
    booking_id: UUID
    conversation_id: UUID
    status: ApprovalStatus
    response: str


def get_appointment_service(request: Request) -> AppointmentServiceProtocol:
    """Return the configured slot-search service."""
    service = getattr(request.app.state, "appointment_service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Appointment service is not configured.",
        )
    return cast(AppointmentServiceProtocol, service)


def get_appointment_workflow(request: Request) -> AppointmentWorkflowProtocol:
    """Return the conversation service that executes checkpointed appointment actions."""
    service = getattr(request.app.state, "conversation_service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Conversation workflow is not configured.",
        )
    return cast(AppointmentWorkflowProtocol, service)


AppointmentServiceDependency = Annotated[
    AppointmentServiceProtocol,
    Depends(get_appointment_service),
]
AppointmentWorkflowDependency = Annotated[
    AppointmentWorkflowProtocol,
    Depends(get_appointment_workflow),
]


def _require_aware_datetime(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"{field_name} must include a timezone.",
        )


def _validate_search_range(starts_at: datetime, ends_at: datetime) -> None:
    _require_aware_datetime(starts_at, "starts_at")
    _require_aware_datetime(ends_at, "ends_at")
    if ends_at <= starts_at:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="ends_at must be later than starts_at.",
        )


def _raise_workflow_error(error: object, request_id: UUID) -> None:
    """Translate appointment-domain outcomes without exposing raw DB errors."""
    if error == "booking_conflict":
        raise HTTPException(status_code=409, detail="The appointment slot is already booked.")
    if error == "slot_unavailable":
        raise HTTPException(status_code=409, detail="The selected appointment slot is unavailable.")
    if error == "conversation_not_found":
        raise HTTPException(status_code=404, detail="Conversation not found.")
    if error == "booking_not_found":
        raise HTTPException(status_code=404, detail="Booking not found.")
    if error == "booking_conversation_mismatch":
        raise HTTPException(status_code=409, detail="The booking belongs to another conversation.")
    if error == "approval_already_pending":
        raise HTTPException(status_code=409, detail="A staff request is already pending.")
    if error == "approval_state_conflict":
        raise HTTPException(status_code=409, detail="The booking is not in a cancellable state.")
    if error == "approval_workflow_required":
        raise HTTPException(
            status_code=503,
            detail="This request requires a configured staff-approval workflow.",
        )
    if error:
        logger.error(
            "Appointment workflow failed (request_id=%s, error_type=%s)",
            request_id,
            error if isinstance(error, str) else type(error).__name__,
        )
        raise HTTPException(
            status_code=500,
            detail="The assistant could not process the appointment request.",
        )


@router.get("/slots", response_model=list[AvailableSlotResponse])
def search_slots(
    starts_at: Annotated[datetime, Query(description="Start of the search range.")],
    ends_at: Annotated[datetime, Query(description="End of the search range.")],
    service: AppointmentServiceDependency,
    doctor_id: Annotated[
        UUID | None,
        Query(description="Optionally restrict results to one doctor."),
    ] = None,
    limit: Annotated[
        int,
        Query(ge=1, le=MAX_SEARCH_LIMIT, description="Maximum number of slots to return."),
    ] = DEFAULT_SEARCH_LIMIT,
) -> list[AvailableSlotResponse]:
    """Return open appointment slots within the requested time range."""
    _validate_search_range(starts_at, ends_at)
    try:
        slots = service.search_available_slots(
            starts_at,
            ends_at,
            doctor_id=doctor_id,
            limit=limit,
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return [AvailableSlotResponse.from_slot(slot) for slot in slots]


@router.post(
    "/bookings",
    response_model=BookingResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_booking(
    payload: BookingRequest,
    service: AppointmentWorkflowDependency,
    owner_reference: OwnerReferenceDependency,
    request_id: IdempotencyKey,
    response: Response,
) -> BookingResponse:
    """Submit through LangGraph so approval-required bookings are checkpointed."""
    try:
        result = await service.process_appointment_action(
            payload.conversation_id,
            owner_reference=owner_reference,
            request_id=request_id,
            action="book",
            slot_id=payload.slot_id,
        )
    except ConversationNotFound as exc:
        raise HTTPException(status_code=404, detail="Conversation not found.") from exc
    except ConversationProcessingError as exc:
        raise HTTPException(
            status_code=503,
            detail="The request could not be resumed; retry with the same Idempotency-Key.",
        ) from exc
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    _raise_workflow_error(result.get("workflow_error"), request_id)

    booking_id = result.get("booking_id")
    booking_status = result.get("booking_status")
    response_text = result.get("response")
    if not isinstance(booking_id, str) or booking_status not in {
        "pending_approval",
        "confirmed",
        "cancelled",
        "rejected",
    }:
        logger.error("Booking workflow returned incomplete state (request_id=%s)", request_id)
        raise HTTPException(status_code=500, detail="The assistant could not process the booking.")
    if not isinstance(response_text, str) or not response_text.strip():
        response_text = "Your booking request was processed."
    try:
        parsed_booking_id = UUID(booking_id)
    except ValueError as exc:
        logger.error("Booking workflow returned an invalid ID (request_id=%s)", request_id)
        raise HTTPException(
            status_code=500, detail="The assistant could not process the booking."
        ) from exc

    if booking_status == "pending_approval":
        response.status_code = status.HTTP_202_ACCEPTED
    return BookingResponse(
        id=parsed_booking_id,
        slot_id=payload.slot_id,
        conversation_id=payload.conversation_id,
        status=cast(BookingStatus, booking_status),
        response=response_text,
    )


@router.post("/cancellations", response_model=ApprovalResponse)
async def request_cancellation(
    payload: CancellationRequest,
    service: AppointmentWorkflowDependency,
    owner_reference: OwnerReferenceDependency,
    request_id: IdempotencyKey,
    response: Response,
) -> ApprovalResponse:
    """Submit cancellation through LangGraph and pause durably for staff review."""
    try:
        result = await service.process_appointment_action(
            payload.conversation_id,
            owner_reference=owner_reference,
            request_id=request_id,
            action="cancel",
            booking_id=payload.booking_id,
        )
    except ConversationNotFound as exc:
        raise HTTPException(status_code=404, detail="Conversation not found.") from exc
    except ConversationProcessingError as exc:
        raise HTTPException(
            status_code=503,
            detail="The request could not be resumed; retry with the same Idempotency-Key.",
        ) from exc
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    _raise_workflow_error(result.get("workflow_error"), request_id)

    approval_id = result.get("approval_id")
    approval_status = result.get("approval_status")
    response_text = result.get("response")
    if not isinstance(approval_id, str) or approval_status not in {
        "pending",
        "approved",
        "rejected",
    }:
        logger.error("Cancellation workflow returned incomplete state (request_id=%s)", request_id)
        raise HTTPException(
            status_code=500, detail="The assistant could not process the cancellation."
        )
    if not isinstance(response_text, str) or not response_text.strip():
        response_text = "Your cancellation request is waiting for staff review."
    try:
        parsed_approval_id = UUID(approval_id)
    except ValueError as exc:
        logger.error("Cancellation workflow returned an invalid ID (request_id=%s)", request_id)
        raise HTTPException(
            status_code=500, detail="The assistant could not process the cancellation."
        ) from exc

    if approval_status == "pending":
        response.status_code = status.HTTP_202_ACCEPTED
    return ApprovalResponse(
        id=parsed_approval_id,
        booking_id=payload.booking_id,
        conversation_id=payload.conversation_id,
        status=cast(ApprovalStatus, approval_status),
        response=response_text,
    )
