"""LangGraph workflow connecting routing to clinic branches."""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, date, datetime, time, timedelta
from typing import Any, Protocol, cast
from uuid import UUID
from zoneinfo import ZoneInfo

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt

from app.graphs.routing import (
    APPOINTMENT_HANDLER,
    CLINIC_QUESTION_HANDLER,
    EMERGENCY_HANDLER,
    FALLBACK_HANDLER,
    classify_intent_node,
    latest_patient_message,
    route_intent,
)
from app.graphs.states import ApprovalStatus, ConversationState, trim_messages_node
from app.repositories.errors import (
    ApprovalAlreadyPending,
    ApprovalStateConflict,
    ApprovalWorkflowRequired,
    BookingConflict,
    ConversationNotFound,
    SlotUnavailable,
)
from app.repositories.models import AppointmentSlot, Approval, AvailableAppointmentSlot, Booking
from app.services.appointments import BookingConversationMismatch, BookingNotFound
from app.services.clinic_knowledge import ClinicKnowledgeService

APPROVAL_HANDLER = "approval_handler"

EMERGENCY_RESPONSE = (
    "This may be a medical emergency. Call your local emergency number or go to the "
    "nearest emergency department now. Do not wait for an appointment."
)
FALLBACK_RESPONSE = (
    "I can help with clinic information and appointments. Please tell me what you need."
)
PENDING_APPROVAL_RESPONSE = "Your request is waiting for staff approval."
APPROVED_RESPONSE = "Clinic staff approved your request."
REJECTED_RESPONSE = (
    "Clinic staff did not approve your request. Please contact the front desk for help."
)

AppointmentHandler = Callable[[ConversationState], Awaitable[Mapping[str, Any]]]
_CLINIC_TIMEZONE = ZoneInfo("Asia/Muscat")
_MAX_OFFERED_SLOTS = 5
_DAYPARTS = {
    "morning": (time(5), time(12)),
    "afternoon": (time(12), time(17)),
    "evening": (time(17), time(23, 59, 59)),
}
_WEEKDAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}


class AppointmentOperations(Protocol):
    """Synchronous appointment operations run by the adapter in worker threads."""

    def search_available_slots(
        self,
        starts_at: datetime,
        ends_at: datetime,
        *,
        doctor_id: UUID | None = None,
        limit: int = _MAX_OFFERED_SLOTS,
    ) -> list[AvailableAppointmentSlot]: ...

    def get_slot(self, slot_id: UUID) -> AppointmentSlot | None: ...

    def book_appointment(
        self,
        slot_id: UUID,
        owner_reference: str,
        conversation_id: UUID,
        *,
        approval_workflow: bool,
        request_id: UUID | None,
    ) -> Booking: ...

    def request_cancellation(
        self,
        booking_id: UUID,
        conversation_id: UUID,
        owner_reference: str,
        *,
        request_id: UUID | None,
    ) -> Approval: ...


class BookingOperations(Protocol):
    """Read existing conversation bookings for explicit patient selection."""

    def get(self, booking_id: UUID) -> Booking | None: ...

    def list_for_conversation(self, conversation_id: UUID) -> list[Booking]: ...


def build_workflow(
    *,
    knowledge_service: ClinicKnowledgeService,
    appointment_handler: AppointmentHandler,
    checkpointer: BaseCheckpointSaver[Any],
) -> CompiledStateGraph[ConversationState, Any, ConversationState, ConversationState]:
    """Build and compile the clinic assistant workflow."""
    if checkpointer is None:
        raise ValueError("A checkpointer is required for approval pause/resume.")

    graph: StateGraph[
        ConversationState,
        None,
        ConversationState,
        ConversationState,
    ] = StateGraph(ConversationState)

    async def clinic_question_node(state: ConversationState) -> dict[str, Any]:
        message = latest_patient_message(state.get("messages"))
        question = message.text.strip() if message is not None else ""

        if not question:
            return {"response": FALLBACK_RESPONSE}

        try:
            answer = await knowledge_service.answer(question)
        except Exception:
            return {"response": FALLBACK_RESPONSE}

        # The service already includes "Source: ..." in the answer.
        return {"response": answer.answer}

    async def appointment_node(state: ConversationState) -> dict[str, Any]:
        try:
            update = dict(await appointment_handler(state))
        except Exception as exc:
            logging.getLogger(__name__).error(
                "Appointment workflow failed (error_type=%s)",
                type(exc).__name__,
            )
            return {
                "workflow_error": "appointment_handler_failed",
                "response": (
                    "I couldn't complete that appointment request. "
                    "Please contact the clinic front desk for help."
                ),
            }

        response = update.get("response")
        if not isinstance(response, str) or not response.strip():
            if update.get("approval_status") == "pending":
                update["response"] = PENDING_APPROVAL_RESPONSE
            else:
                update["response"] = FALLBACK_RESPONSE

        return update

    graph.add_node("trim_messages", trim_messages_node)
    graph.add_node("classify_intent", classify_intent_node)
    graph.add_node(
        CLINIC_QUESTION_HANDLER,
        clinic_question_node,
        input_schema=ConversationState,
    )
    graph.add_node(
        APPOINTMENT_HANDLER,
        appointment_node,
        input_schema=ConversationState,
    )
    graph.add_node(EMERGENCY_HANDLER, _emergency_node)
    graph.add_node(FALLBACK_HANDLER, _fallback_node)
    graph.add_node(APPROVAL_HANDLER, _approval_node)

    graph.add_edge(START, "trim_messages")
    graph.add_edge("trim_messages", "classify_intent")

    graph.add_conditional_edges(
        "classify_intent",
        route_intent,
        {
            CLINIC_QUESTION_HANDLER: CLINIC_QUESTION_HANDLER,
            APPOINTMENT_HANDLER: APPOINTMENT_HANDLER,
            EMERGENCY_HANDLER: EMERGENCY_HANDLER,
            FALLBACK_HANDLER: FALLBACK_HANDLER,
        },
    )

    graph.add_edge(CLINIC_QUESTION_HANDLER, END)
    graph.add_edge(EMERGENCY_HANDLER, END)
    graph.add_edge(FALLBACK_HANDLER, END)

    graph.add_conditional_edges(
        APPOINTMENT_HANDLER,
        _route_after_appointment,
        {
            APPROVAL_HANDLER: APPROVAL_HANDLER,
            END: END,
        },
    )
    graph.add_edge(APPROVAL_HANDLER, END)

    return graph.compile(checkpointer=checkpointer)


def make_appointment_handler(
    appointments: AppointmentOperations,
    bookings: BookingOperations,
) -> AppointmentHandler:
    """Search, select, book, and cancel only the appointment named by the patient."""

    async def handle(state: ConversationState) -> Mapping[str, Any]:
        conversation_text = state.get("conversation_id")
        owner_reference = state.get("patient_id")
        if not isinstance(conversation_text, str) or not isinstance(owner_reference, str):
            return {
                "workflow_error": "conversation_not_found",
                "response": "Conversation not found.",
            }

        try:
            conversation_id = UUID(conversation_text)
        except ValueError:
            return {
                "workflow_error": "conversation_not_found",
                "response": "Conversation not found.",
            }

        raw_request_id = state.get("request_id")
        request_id = UUID(raw_request_id) if isinstance(raw_request_id, str) else None
        message = latest_patient_message(state.get("messages"))
        text = message.text.strip() if message is not None else ""
        intent = state.get("intent")

        if intent == "reschedule_appointment":
            return {
                "response": (
                    "I can help find a replacement time, but I won't change an existing booking "
                    "until you select the appointment and a new slot. Please tell me which "
                    "appointment to move, or contact the front desk for help."
                )
            }

        if intent in {"find_slots", "book_appointment"}:
            slot_id = state.get("requested_slot")
            offered_slots = state.get("available_slots", [])
            if not isinstance(offered_slots, list):
                offered_slots = []
            if not isinstance(slot_id, str):
                slot_id = _selected_id(text, offered_slots, id_field="id")
            if not isinstance(slot_id, str):
                stored_selection = state.get("selected_slot_id")
                slot_id = stored_selection if isinstance(stored_selection, str) else None

            if intent == "find_slots" and slot_id is not None:
                selected = next(
                    (item for item in offered_slots if item.get("id") == slot_id),
                    None,
                )
                if selected is not None:
                    return {
                        "available_slots": offered_slots,
                        "selected_slot_id": slot_id,
                        "approval_status": "not_required",
                        "response": (
                            f"You selected {selected['starts_at']} with {selected['doctor_name']}. "
                            "Say 'book this slot' if you want me to reserve it."
                        ),
                    }

            if slot_id is not None:
                try:
                    booking = await asyncio.to_thread(
                        appointments.book_appointment,
                        UUID(slot_id),
                        owner_reference,
                        conversation_id,
                        approval_workflow=True,
                        request_id=request_id,
                    )
                except Exception as exc:
                    return _appointment_failure(exc)
                pending = booking.status == "pending_approval"
                response = (
                    PENDING_APPROVAL_RESPONSE
                    if pending
                    else "Your appointment is confirmed."
                    if booking.status == "confirmed"
                    else "The booking is no longer active. Please choose another slot."
                )
                return {
                    "booking_id": str(booking.id),
                    "booking_status": booking.status,
                    "approval_id": str(booking.id) if pending else None,
                    "approval_status": "pending" if pending else "not_required",
                    "available_slots": [],
                    "selected_slot_id": None,
                    "response": response,
                }

            starts_at, ends_at, daypart = _search_window(text)
            preferred_time = _requested_time(text)
            try:
                slots = await asyncio.to_thread(
                    appointments.search_available_slots,
                    starts_at,
                    ends_at,
                    limit=25,
                )
            except Exception as exc:
                return _appointment_failure(exc)
            selected_slots = _filter_daypart(slots, daypart)
            if preferred_time is not None:
                selected_slots = [
                    slot
                    for slot in selected_slots
                    if slot.starts_at.astimezone(_CLINIC_TIMEZONE).hour == preferred_time.hour
                ]
            selected_slots = selected_slots[:_MAX_OFFERED_SLOTS]
            options = [
                {
                    "id": str(slot.id),
                    "starts_at": slot.starts_at.isoformat(),
                    "doctor_name": slot.doctor_name,
                    "doctor_specialty": slot.doctor_specialty,
                }
                for slot in selected_slots
            ]
            if not options:
                return {
                    "available_slots": [],
                    "selected_slot_id": None,
                    "approval_status": "not_required",
                    "response": (
                        "I couldn't find an available slot matching that preference. "
                        "Please try another date or contact the front desk."
                    ),
                }
            lines = [
                f"{index}. {option['starts_at']} — {option['doctor_name']} "
                f"({option['doctor_specialty']}); slot ID {option['id']}"
                for index, option in enumerate(options, start=1)
            ]
            return {
                "available_slots": options,
                "selected_slot_id": None,
                "approval_status": "not_required",
                "response": "Available appointments:\n"
                + "\n".join(lines)
                + "\nReply with an option number or slot ID to book.",
            }

        if intent == "cancel_appointment":
            booking_id = state.get("requested_booking_id")
            offered_bookings = state.get("available_bookings", [])
            if not isinstance(offered_bookings, list):
                offered_bookings = []
            if not isinstance(booking_id, str):
                booking_id = _selected_id(text, offered_bookings, id_field="booking_id")

            if booking_id is None:
                options = await _future_booking_options(
                    bookings,
                    appointments,
                    conversation_id,
                )
                if not options:
                    return {
                        "available_bookings": [],
                        "approval_status": "not_required",
                        "response": "I couldn't find a future confirmed appointment to cancel.",
                    }
                if len(options) == 1 and _is_explicit_single_cancellation(text):
                    booking_id = options[0]["booking_id"]
                else:
                    lines = [
                        f"{index}. {item['starts_at']} — booking ID {item['booking_id']}"
                        for index, item in enumerate(options, start=1)
                    ]
                    return {
                        "available_bookings": options,
                        "approval_status": "not_required",
                        "response": "Which appointment should I request to cancel?\n"
                        + "\n".join(lines)
                        + "\nReply with an option number or booking ID.",
                    }

            try:
                approval = await asyncio.to_thread(
                    appointments.request_cancellation,
                    UUID(booking_id),
                    conversation_id,
                    owner_reference,
                    request_id=request_id,
                )

                def get_booking() -> tuple[bool, Booking | None]:
                    booking = bookings.get(approval.booking_id)
                    return booking is not None, booking

                _, cancellation_booking = await asyncio.to_thread(get_booking)
            except Exception as exc:
                return _appointment_failure(exc)
            is_pending = approval.status == "pending"
            response = (
                PENDING_APPROVAL_RESPONSE
                if is_pending
                else "The cancellation was approved."
                if approval.status == "approved"
                else "The cancellation was not approved. Your appointment remains active."
            )
            return {
                "booking_id": str(approval.booking_id),
                "booking_status": (
                    cancellation_booking.status if cancellation_booking is not None else None
                ),
                "approval_id": str(approval.id),
                "approval_status": approval.status,
                "available_bookings": [],
                "response": response,
            }

        return {"response": FALLBACK_RESPONSE}

    return handle


def _search_window(text: str) -> tuple[datetime, datetime, str | None]:
    """Resolve ISO dates, today/tomorrow, and daypart preferences safely."""
    now = datetime.now(_CLINIC_TIMEZONE)
    lowered = text.casefold()
    requested_date: date | None = None
    date_match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", lowered)
    if date_match:
        try:
            requested_date = date.fromisoformat(date_match.group(1))
        except ValueError:
            requested_date = None
    elif re.search(r"\btomorrow\b", lowered):
        requested_date = now.date() + timedelta(days=1)
    elif re.search(r"\btoday\b", lowered):
        requested_date = now.date()
    else:
        for weekday, weekday_index in _WEEKDAYS.items():
            if re.search(rf"\b{weekday}\b", lowered):
                days_ahead = (weekday_index - now.weekday()) % 7
                requested_date = now.date() + timedelta(days=days_ahead)
                break

    if requested_date is None:
        start = now
        end = now + timedelta(days=30)
    else:
        start = datetime.combine(requested_date, time.min, tzinfo=_CLINIC_TIMEZONE)
        end = start + timedelta(days=1)
        if start < now:
            start = now

    daypart = next((part for part in _DAYPARTS if part in lowered), None)
    return start, end, daypart


def _filter_daypart(
    slots: list[AvailableAppointmentSlot],
    daypart: str | None,
) -> list[AvailableAppointmentSlot]:
    if daypart is None:
        return slots
    start, end = _DAYPARTS[daypart]
    return [
        slot
        for slot in slots
        if start <= slot.starts_at.astimezone(_CLINIC_TIMEZONE).time().replace(tzinfo=None) < end
    ]


def _requested_time(text: str) -> time | None:
    """Parse common 12-hour appointment-time preferences without guessing."""
    match = re.search(r"\b(1[0-2]|0?[1-9])(?::([0-5][0-9]))?\s*(am|pm)\b", text, re.IGNORECASE)
    if match is None:
        return None
    hour = int(match.group(1)) % 12
    if match.group(3).casefold() == "pm":
        hour += 12
    return time(hour, int(match.group(2) or "0"))


def _selected_id(
    text: str,
    options: list[dict[str, str]],
    *,
    id_field: str,
) -> str | None:
    if not options:
        return None
    available = {
        item[id_field].casefold(): item[id_field]
        for item in options
        if isinstance(item.get(id_field), str)
    }
    for value in re.findall(
        r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
        text,
        re.IGNORECASE,
    ):
        if value.casefold() in available:
            return available[value.casefold()]

    lowered = text.casefold()
    selection = re.search(r"\b(?:option|choice|number)\s*([1-5])\b", lowered)
    if selection is not None:
        index = int(selection.group(1)) - 1
        return options[index][id_field] if 0 <= index < len(options) else None

    ordinal = re.fullmatch(
        r"\s*(?:(?:book|reserve|cancel|choose|select)\s+)?(?:the\s+)?"
        r"(first|one|second|two|third|three|fourth|four|fifth|five)"
        r"(?:\s+(?:one|slot|option|appointment|booking))?\s*[.!]?\s*",
        lowered,
    )
    if ordinal is not None:
        index = {
            "first": 0,
            "one": 0,
            "second": 1,
            "two": 1,
            "third": 2,
            "three": 2,
            "fourth": 3,
            "four": 3,
            "fifth": 4,
            "five": 4,
        }[ordinal.group(1)]
        return options[index][id_field] if index < len(options) else None
    return None


async def _future_booking_options(
    bookings: BookingOperations,
    appointments: AppointmentOperations,
    conversation_id: UUID,
) -> list[dict[str, str]]:
    current_time = datetime.now(UTC)
    conversation_bookings = await asyncio.to_thread(bookings.list_for_conversation, conversation_id)
    options: list[dict[str, str]] = []
    for booking in conversation_bookings:
        if booking.status != "confirmed":
            continue
        slot = await asyncio.to_thread(appointments.get_slot, booking.slot_id)
        if slot is None or slot.starts_at <= current_time:
            continue
        options.append(
            {
                "booking_id": str(booking.id),
                "slot_id": str(slot.id),
                "starts_at": slot.starts_at.astimezone(_CLINIC_TIMEZONE).isoformat(),
            }
        )
    return options


def _is_explicit_single_cancellation(text: str) -> bool:
    lowered = text.casefold()
    return bool(re.search(r"\b(cancel|cancellation|call off)\b", lowered)) and bool(
        re.search(r"\b(my|the|this)\s+(appointment|booking)\b", lowered)
    )


def _appointment_failure(error: Exception) -> Mapping[str, Any]:
    if isinstance(error, ConversationNotFound):
        return {"workflow_error": "conversation_not_found", "response": "Conversation not found."}
    if isinstance(error, BookingNotFound):
        return {"workflow_error": "booking_not_found", "response": "Booking not found."}
    if isinstance(error, BookingConversationMismatch):
        return {
            "workflow_error": "booking_conversation_mismatch",
            "response": "That booking does not belong to this conversation.",
        }
    if isinstance(error, BookingConflict):
        return {"workflow_error": "booking_conflict", "response": "That slot is already booked."}
    if isinstance(error, SlotUnavailable):
        return {"workflow_error": "slot_unavailable", "response": "That slot is unavailable."}
    if isinstance(error, ApprovalAlreadyPending):
        return {
            "workflow_error": "approval_already_pending",
            "response": "A staff request is already pending for that booking.",
        }
    if isinstance(error, ApprovalStateConflict):
        return {
            "workflow_error": "approval_state_conflict",
            "response": "That booking cannot be changed in its current state.",
        }
    if isinstance(error, ApprovalWorkflowRequired):
        return {
            "workflow_error": "approval_workflow_required",
            "response": "This request requires a configured staff-approval workflow.",
        }
    return {
        "workflow_error": "appointment_handler_failed",
        "response": (
            "I couldn't complete that appointment request. Please contact the clinic front desk."
        ),
    }


def _route_after_appointment(state: ConversationState) -> str:
    if state.get("approval_status") == "pending":
        return APPROVAL_HANDLER
    return END


def _approval_node(state: ConversationState) -> dict[str, Any]:
    """Pause for staff approval and resume with the staff decision."""
    decision = interrupt(
        {
            "type": "staff_approval",
            "conversation_id": state.get("conversation_id"),
            "approval_id": state.get("approval_id"),
            "booking_id": state.get("booking_id"),
            "intent": state.get("intent"),
            "requested_slot": state.get("requested_slot"),
            "requested_booking_id": state.get("requested_booking_id"),
        }
    )

    status = _approval_status(decision)
    update: dict[str, Any] = {
        "approval_status": status,
        "response": APPROVED_RESPONSE if status == "approved" else REJECTED_RESPONSE,
    }
    if state.get("intent") == "book_appointment":
        update["booking_status"] = "confirmed" if status == "approved" else "rejected"
    elif state.get("intent") == "cancel_appointment":
        update["booking_status"] = "cancelled" if status == "approved" else "confirmed"
    return update


def _approval_status(decision: object) -> ApprovalStatus:
    if not isinstance(decision, Mapping):
        raise ValueError("Approval decision must be a mapping.")

    value = decision.get("decision")
    if not isinstance(value, str) or value not in {"approved", "rejected"}:
        raise ValueError("Approval decision must be 'approved' or 'rejected'.")

    return cast(ApprovalStatus, value)


def _emergency_node(state: ConversationState) -> dict[str, str]:
    return {"response": EMERGENCY_RESPONSE}


def _fallback_node(state: ConversationState) -> dict[str, str]:
    return {"response": FALLBACK_RESPONSE}


__all__ = [
    "APPROVAL_HANDLER",
    "APPROVED_RESPONSE",
    "AppointmentHandler",
    "EMERGENCY_RESPONSE",
    "FALLBACK_RESPONSE",
    "PENDING_APPROVAL_RESPONSE",
    "REJECTED_RESPONSE",
    "build_workflow",
    "make_appointment_handler",
]
