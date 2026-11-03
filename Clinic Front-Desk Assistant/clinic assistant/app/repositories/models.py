"""Typed records shared by the PostgreSQL repositories and services."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal
from uuid import UUID

SlotStatus = Literal["open", "blocked"]
BookingStatus = Literal["pending_approval", "confirmed", "cancelled", "rejected"]
ApprovalAction = Literal["same_day_booking", "cancel_booking"]
ApprovalStatus = Literal["pending", "approved", "rejected"]
ApprovalDecision = Literal["approved", "rejected"]


@dataclass(frozen=True, slots=True)
class Doctor:
    """A doctor record from the clinic database."""

    id: UUID
    full_name: str
    specialty: str
    is_active: bool
    created_at: datetime


@dataclass(frozen=True, slots=True)
class AppointmentSlot:
    """A scheduled slot, including its database availability status."""

    id: UUID
    doctor_id: UUID
    starts_at: datetime
    ends_at: datetime
    status: SlotStatus
    created_at: datetime


@dataclass(frozen=True, slots=True)
class AvailableAppointmentSlot:
    """An open slot joined with the active doctor's display details."""

    id: UUID
    doctor_id: UUID
    doctor_name: str
    doctor_specialty: str
    starts_at: datetime
    ends_at: datetime


@dataclass(frozen=True, slots=True)
class Conversation:
    """Conversation metadata linked to its stable LangGraph thread ID."""

    id: UUID
    thread_id: str
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class Booking:
    """A slot reservation identified only by an opaque patient reference."""

    id: UUID
    slot_id: UUID
    conversation_id: UUID | None
    patient_reference: str
    status: BookingStatus
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class Approval:
    """A staff decision request associated with its booking conversation."""

    id: UUID
    booking_id: UUID
    conversation_id: UUID
    action: ApprovalAction
    status: ApprovalStatus
    requested_at: datetime
    decided_at: datetime | None
    decided_by: str | None
    decision_note: str | None
