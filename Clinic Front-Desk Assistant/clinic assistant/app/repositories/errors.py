
"""Domain errors raised by PostgreSQL repository operations."""


class RepositoryError(RuntimeError):
    """Base class for persistence errors that services can handle safely."""


class DoctorNotFound(RepositoryError):
    """A requested doctor does not exist."""


class ConversationNotFound(RepositoryError):
    """A requested conversation does not exist."""


class AppointmentSlotConflict(RepositoryError):
    """A doctor already has an overlapping appointment slot."""


class InvalidAppointmentSlot(RepositoryError):
    """An appointment slot has invalid times or status."""


class BookingConflict(RepositoryError):
    """A slot already has an active booking."""


class SlotUnavailable(RepositoryError):
    """A booking targets a blocked slot or a slot for an inactive doctor."""


class ApprovalContextMismatch(RepositoryError):
    """An approval does not reference the booking's conversation."""


class ApprovalAlreadyPending(RepositoryError):
    """An approval for the same booking action is already pending."""


class ApprovalNotFound(RepositoryError):
    """The requested approval does not exist."""


class ApprovalAlreadyDecided(RepositoryError):
    """The approval is no longer pending."""


class ApprovalStateConflict(RepositoryError):
    """The booking is not in the state required by its approval action."""
