"""LangGraph workflow connecting routing to clinic branches."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, cast
from uuid import UUID

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
from app.repositories.bookings import BookingRepository
from app.repositories.models import Booking
from app.services.appointments import AppointmentService
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
        except Exception:
            return {"response": FALLBACK_RESPONSE}

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
    appointments: AppointmentService,
    bookings: BookingRepository,
) -> AppointmentHandler:
    """Adapt the appointment service to the workflow's handler contract."""

    async def handle(state: ConversationState) -> Mapping[str, Any]:
        raw_conversation_id = state.get("conversation_id")
        if not isinstance(raw_conversation_id, str):
            return {"response": FALLBACK_RESPONSE}

        try:
            conversation_id = UUID(raw_conversation_id)
        except ValueError:
            return {"response": FALLBACK_RESPONSE}

        intent = state.get("intent")

        if intent == "cancel_appointment":
            booking = await _active_booking(bookings, conversation_id)
            if booking is None:
                return {"response": "I couldn't find an active appointment to cancel."}

            await asyncio.to_thread(
                appointments.request_cancellation,
                booking.id,
                conversation_id,
            )
            return {
                "response": PENDING_APPROVAL_RESPONSE,
                "approval_status": "pending",
            }

        if intent == "book_appointment":
            slot_value = state.get("requested_slot")
            if not isinstance(slot_value, str):
                return {"response": "Please provide the appointment slot you want."}

            try:
                slot_id = UUID(slot_value)
            except ValueError:
                return {"response": "Please provide a valid appointment slot."}

            patient_reference = state.get("patient_id") or str(conversation_id)
            booking = await asyncio.to_thread(
                appointments.book_appointment,
                slot_id,
                patient_reference,
                conversation_id,
            )

            if booking.status == "pending_approval":
                return {
                    "response": PENDING_APPROVAL_RESPONSE,
                    "approval_status": "pending",
                }

            return {
                "response": "Your appointment is confirmed.",
                "approval_status": "not_required",
            }

        if intent == "find_slots":
            return {"response": "Please tell me the date and time you prefer."}

        return {"response": FALLBACK_RESPONSE}

    return handle


async def _active_booking(
    bookings: BookingRepository,
    conversation_id: UUID,
) -> Booking | None:
    conversation_bookings = await asyncio.to_thread(
        bookings.list_for_conversation,
        conversation_id,
    )
    return next(
        (booking for booking in conversation_bookings if booking.status == "confirmed"),
        None,
    )


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
            "intent": state.get("intent"),
            "requested_slot": state.get("requested_slot"),
        }
    )

    status = _approval_status(decision)
    return {
        "approval_status": status,
        "response": APPROVED_RESPONSE if status == "approved" else REJECTED_RESPONSE,
    }


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
