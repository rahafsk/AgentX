"""PostgreSQL repositories and typed records for the clinic assistant."""

from app.repositories.appointments import AppointmentRepository
from app.repositories.approvals import ApprovalDecision, ApprovalRepository
from app.repositories.bookings import BookingRepository
from app.repositories.conversations import ConversationRepository
from app.repositories.doctors import DoctorRepository
from app.repositories.errors import (
    AppointmentSlotConflict,
    ApprovalAlreadyDecided,
    ApprovalAlreadyPending,
    ApprovalContextMismatch,
    ApprovalNotFound,
    ApprovalStateConflict,
    BookingConflict,
    ConversationNotFound,
    DoctorNotFound,
    InvalidAppointmentSlot,
    RepositoryError,
    SlotUnavailable,
)
from app.repositories.models import (
    AppointmentSlot,
    Approval,
    ApprovalAction,
    ApprovalStatus,
    AvailableAppointmentSlot,
    Booking,
    BookingStatus,
    Conversation,
    Doctor,
    SlotStatus,
)

__all__ = [
    "Approval",
    "ApprovalAction",
    "ApprovalAlreadyDecided",
    "ApprovalAlreadyPending",
    "ApprovalContextMismatch",
    "ApprovalDecision",
    "ApprovalNotFound",
    "ApprovalRepository",
    "ApprovalStateConflict",
    "ApprovalStatus",
    "AppointmentRepository",
    "AppointmentSlot",
    "AppointmentSlotConflict",
    "AvailableAppointmentSlot",
    "Booking",
    "BookingConflict",
    "BookingRepository",
    "BookingStatus",
    "Conversation",
    "ConversationNotFound",
    "ConversationRepository",
    "Doctor",
    "DoctorNotFound",
    "DoctorRepository",
    "InvalidAppointmentSlot",
    "RepositoryError",
    "SlotStatus",
    "SlotUnavailable",
]
