"""HTTP endpoints for searching slots, creating bookings, and requesting cancellations."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Protocol, cast
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field, field_validator

from app.repositories.appointments import MAX_SEARCH_LIMIT
from app.repositories.errors import (
    ApprovalAlreadyPending,
    ApprovalStateConflict,
    BookingConflict,
    ConversationNotFound,
    SlotUnavailable,
)
from app.repositories.models import (
    Approval,
    ApprovalAction,
    ApprovalStatus,
    AvailableAppointmentSlot,
    Booking,
    BookingStatus,
)
from app.services.appointments import (
    DEFAULT_SEARCH_LIMIT,
    BookingConversationMismatch,
    BookingNotFound,
)

router = APIRouter(prefix="/appointments", tags=["appointments"])


class AppointmentServiceProtocol(Protocol):
    """Service operations required by the appointment API."""

    def search_available_slots(
        self,
        starts_at: datetime,
        ends_at: datetime,
        *,
        doctor_id: UUID | None = None,
        limit: int = DEFAULT_SEARCH_LIMIT,
    ) -> list[AvailableAppointmentSlot]: ...

    def book_appointment(
        self,
        slot_id: UUID,
        patient_reference: str,
        conversation_id: UUID,
        *,
        requires_approval: bool = False,
    ) -> Booking: ...

    def request_cancellation(
        self,
        booking_id: UUID,
        conversation_id: UUID,
    ) -> Approval: ...


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
        """Build a response from a repository slot record."""
        return cls(
            id=slot.id,
            doctor_id=slot.doctor_id,
            doctor_name=slot.doctor_name,
            doctor_specialty=slot.doctor_specialty,
            starts_at=slot.starts_at,
            ends_at=slot.ends_at,
        )


class BookingRequest(BaseModel):
    """A request to reserve an open appointment slot."""

    slot_id: UUID
    patient_reference: str = Field(min_length=1)
    conversation_id: UUID
    requires_approval: bool = False

    @field_validator("patient_reference")
    @classmethod
    def reject_blank_patient_reference(cls, value: str) -> str:
        """Strip surrounding whitespace and reject an empty reference."""
        patient_reference = value.strip()
        if not patient_reference:
            raise ValueError("Patient reference cannot be blank.")
        return patient_reference


class BookingResponse(BaseModel):
    """A booking, including whether it is awaiting staff approval."""

    id: UUID
    slot_id: UUID
    conversation_id: UUID | None
    patient_reference: str
    status: BookingStatus
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_booking(cls, booking: Booking) -> BookingResponse:
        """Build a response from a repository booking record."""
        return cls(
            id=booking.id,
            slot_id=booking.slot_id,
            conversation_id=booking.conversation_id,
            patient_reference=booking.patient_reference,
            status=booking.status,
            created_at=booking.created_at,
            updated_at=booking.updated_at,
        )


class CancellationRequest(BaseModel):
    """A request to cancel a confirmed booking with staff approval."""

    booking_id: UUID
    conversation_id: UUID


class ApprovalResponse(BaseModel):
    """A staff approval request, including its pending or decided state."""

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
        """Build a response from a repository approval record."""
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


def get_appointment_service(request: Request) -> AppointmentServiceProtocol:
    """Return the configured appointment service from application state."""
    service = getattr(request.app.state, "appointment_service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Appointment service is not configured.",
        )
    return cast(AppointmentServiceProtocol, service)


AppointmentServiceDependency = Annotated[
    AppointmentServiceProtocol,
    Depends(get_appointment_service),
]


def _require_aware_datetime(value: datetime, field_name: str) -> None:
    """Reject a query datetime that does not include timezone information."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"{field_name} must include a timezone.",
        )


def _validate_search_range(starts_at: datetime, ends_at: datetime) -> None:
    """Validate the requested slot-search time range."""
    _require_aware_datetime(starts_at, "starts_at")
    _require_aware_datetime(ends_at, "ends_at")

    if ends_at <= starts_at:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="ends_at must be later than starts_at.",
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
        Query(
            ge=1,
            le=MAX_SEARCH_LIMIT,
            description="Maximum number of slots to return.",
        ),
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
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc

    return [AvailableSlotResponse.from_slot(slot) for slot in slots]


@router.post(
    "/bookings",
    response_model=BookingResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_booking(
    payload: BookingRequest,
    service: AppointmentServiceDependency,
) -> BookingResponse:
    """Book an open slot and preserve any required staff-approval status."""
    try:
        booking = service.book_appointment(
            payload.slot_id,
            payload.patient_reference,
            payload.conversation_id,
            requires_approval=payload.requires_approval,
        )
    except BookingConflict as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="The appointment slot already has an active booking.",
        ) from exc
    except SlotUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="The selected appointment slot is unavailable.",
        ) from exc
    except ConversationNotFound as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found.",
        ) from exc
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc

    return BookingResponse.from_booking(booking)


@router.post(
    "/cancellations",
    response_model=ApprovalResponse,
    status_code=status.HTTP_201_CREATED,
)
def request_cancellation(
    payload: CancellationRequest,
    service: AppointmentServiceDependency,
) -> ApprovalResponse:
    """Request staff approval to cancel a confirmed booking."""
    try:
        approval = service.request_cancellation(
            payload.booking_id,
            payload.conversation_id,
        )
    except BookingNotFound as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Booking not found.",
        ) from exc
    except BookingConversationMismatch as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="The booking belongs to a different conversation.",
        ) from exc
    except ApprovalAlreadyPending as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A cancellation request for this booking is already pending.",
        ) from exc
    except ApprovalStateConflict as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Only confirmed bookings can be cancelled.",
        ) from exc
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc

    return ApprovalResponse.from_approval(approval)
