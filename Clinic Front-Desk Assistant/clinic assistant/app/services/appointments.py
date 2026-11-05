"""Coordinate appointment searches, bookings, and cancellation approvals.

The service validates front-desk requests and composes the PostgreSQL
repositories. Atomic booking, slot locking, duplicate-reservation protection,
and same-day approval creation remain the responsibility of
``BookingRepository``; the service does not duplicate those database rules.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from psycopg import Connection

from app.repositories.appointments import MAX_SEARCH_LIMIT, AppointmentRepository
from app.repositories.approvals import ApprovalRepository
from app.repositories.bookings import BookingRepository
from app.repositories.conversations import ConversationRepository
from app.repositories.errors import (
    ApprovalContextMismatch,
    ConversationNotFound,
)
from app.repositories.models import Approval, AvailableAppointmentSlot, Booking

DEFAULT_SEARCH_LIMIT = 25


class BookingNotFound(ApprovalContextMismatch):
    """The booking referenced by a cancellation request does not exist."""


class BookingConversationMismatch(ApprovalContextMismatch):
    """A booking belongs to a conversation other than the one supplied."""


class AppointmentService:
    """Search slots and coordinate booking and cancellation workflows."""

    def __init__(
        self,
        appointment_repository: AppointmentRepository,
        booking_repository: BookingRepository,
        approval_repository: ApprovalRepository,
        conversation_repository: ConversationRepository | None = None,
    ) -> None:
        self._appointments = appointment_repository
        self._bookings = booking_repository
        self._approvals = approval_repository
        self._conversations = conversation_repository

    @classmethod
    def from_connection(
        cls,
        connection: Connection[Any],
        *,
        clinic_timezone: str = "Asia/Muscat",
    ) -> AppointmentService:
        """Build the service with the repositories that share one connection."""
        return cls(
            AppointmentRepository(connection),
            BookingRepository(connection, clinic_timezone=clinic_timezone),
            ApprovalRepository(connection),
            ConversationRepository(connection),
        )

    def search_available_slots(
        self,
        starts_at: datetime,
        ends_at: datetime,
        *,
        doctor_id: UUID | None = None,
        limit: int = DEFAULT_SEARCH_LIMIT,
    ) -> list[AvailableAppointmentSlot]:
        """Return open slots in the requested time range.

        The repository applies the database availability rules and caps the
        result set at ``MAX_SEARCH_LIMIT``. Inputs are validated here as well so
        callers get a clear error before a query is attempted.
        """
        starts_at = _require_aware_datetime(starts_at, "starts_at")
        ends_at = _require_aware_datetime(ends_at, "ends_at")

        if ends_at <= starts_at:
            raise ValueError("ends_at must be later than starts_at.")

        if isinstance(limit, bool) or not isinstance(limit, int):
            raise TypeError("limit must be an integer.")

        if not 1 <= limit <= MAX_SEARCH_LIMIT:
            raise ValueError(f"limit must be between 1 and {MAX_SEARCH_LIMIT}.")

        if doctor_id is not None:
            doctor_id = _require_uuid(doctor_id, "doctor_id")

        return self._appointments.search_open_slots(
            starts_at,
            ends_at,
            doctor_id=doctor_id,
            limit=limit,
        )

    def book_appointment(
        self,
        slot_id: UUID,
        patient_reference: str,
        conversation_id: UUID,
        *,
        requires_approval: bool = False,
    ) -> Booking:
        """Create a booking for an open slot.

        ``BookingRepository`` performs the atomic insert and automatically
        creates a pending staff approval for same-day bookings. The returned
        booking status therefore indicates whether approval is required.
        """
        slot_id = _require_uuid(slot_id, "slot_id")
        conversation_id = _require_uuid(conversation_id, "conversation_id")

        if not isinstance(patient_reference, str):
            raise TypeError("patient_reference must be text.")

        patient_reference = patient_reference.strip()
        if not patient_reference:
            raise ValueError("patient_reference cannot be empty.")

        if not isinstance(requires_approval, bool):
            raise TypeError("requires_approval must be a boolean.")

        self._require_conversation(conversation_id)

        return self._bookings.create(
            slot_id,
            patient_reference,
            conversation_id=conversation_id,
            requires_approval=requires_approval,
        )

    def request_cancellation(
        self,
        booking_id: UUID,
        conversation_id: UUID,
    ) -> Approval:
        """Request staff approval to cancel a booking in its own conversation.

        The approval repository enforces the confirmed-booking rule and the
        single-pending-approval constraint. Staff decisions are handled by
        ``ApprovalRepository.decide`` rather than by this service.
        """
        booking_id = _require_uuid(booking_id, "booking_id")
        conversation_id = _require_uuid(conversation_id, "conversation_id")

        booking = self._bookings.get(booking_id)
        if booking is None:
            raise BookingNotFound("The booking does not exist.")

        if booking.conversation_id != conversation_id:
            raise BookingConversationMismatch("The booking belongs to a different conversation.")

        return self._approvals.create(
            booking_id,
            conversation_id,
            "cancel_booking",
        )

    def _require_conversation(self, conversation_id: UUID) -> None:
        """Validate the conversation when conversation validation is enabled."""
        if self._conversations is None:
            return

        if self._conversations.get(conversation_id) is None:
            raise ConversationNotFound(f"Conversation {conversation_id} was not found.")


def _require_uuid(value: object, field_name: str) -> UUID:
    """Return a UUID or raise a clear type error for a service input."""
    if not isinstance(value, UUID):
        raise TypeError(f"{field_name} must be a UUID.")
    return value


def _require_aware_datetime(value: object, field_name: str) -> datetime:
    """Return a timezone-aware datetime or raise a clear input error."""
    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be a datetime.")

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must include a timezone.")

    return value


__all__ = [
    "AppointmentService",
    "BookingConversationMismatch",
    "BookingNotFound",
    "DEFAULT_SEARCH_LIMIT",
]
