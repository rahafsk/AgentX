"""Shared locking order for booking and approval operations."""

# _locks.py is a helper file shared by bookings.py and approvals.py.
# A lock temporarily prevents another transaction from changing the same database row. Both repositories should acquire locks in the same order:
# Doctor → Appointment slot → Booking → Approval
# Using different orders can cause a deadlock, where two transactions wait for each other.
# The shared file keeps this logic in one place. It does not create a new database table or repository.
from typing import Any
from uuid import UUID

from psycopg import Cursor

from app.repositories.errors import SlotUnavailable


def lock_slot(
    cursor: Cursor[dict[str, Any]],
    slot_id: UUID,
) -> dict[str, Any]:
    """Lock the doctor first, then the appointment slot."""
    cursor.execute(
        "SELECT doctor_id FROM appointment_slots WHERE id = %s",
        (slot_id,),
    )
    reference = cursor.fetchone()

    if reference is None:
        raise SlotUnavailable("The appointment slot does not exist.")

    doctor_id = reference["doctor_id"]

    cursor.execute(
        """
        SELECT is_active
        FROM doctors
        WHERE id = %s
        FOR UPDATE
        """,
        (doctor_id,),
    )
    doctor = cursor.fetchone()

    if doctor is None:
        raise SlotUnavailable("The doctor does not exist.")

    cursor.execute(
        """
        SELECT id, doctor_id, starts_at, status
        FROM appointment_slots
        WHERE id = %s
        FOR UPDATE
        """,
        (slot_id,),
    )
    slot = cursor.fetchone()

    if slot is None or slot["doctor_id"] != doctor_id:
        raise SlotUnavailable("The appointment slot changed. Retry the request.")

    slot["doctor_is_active"] = doctor["is_active"]
    return slot
