"""Shared state definitions for the clinic assistant workflow."""

from typing import Annotated, Literal, NotRequired, TypedDict

from langchain_core.messages import AnyMessage, RemoveMessage
from langgraph.graph.message import add_messages

# Maximum number of recent messages to retain.
MAX_MESSAGES = 20


Intent = Literal[
    "clinic_question",
    "find_slots",
    "book_appointment",
    "cancel_appointment",
    "reschedule_appointment",
    "emergency",
    "other",
]

ApprovalStatus = Literal[
    "not_required",
    "pending",
    "approved",
    "rejected",
]

BookingStatus = Literal["pending_approval", "confirmed", "cancelled", "rejected"]


class ConversationState(TypedDict):
    """State shared between conversation workflow nodes."""

    # Shared conversation identifier
    conversation_id: str

    # Stable client idempotency key for this turn; nodes should pass it to side effects.
    request_id: NotRequired[str]

    # Optional minimal patient reference, if available
    patient_id: NotRequired[str | None]

    # Conversation messages managed by LangGraph
    messages: Annotated[list[AnyMessage], add_messages]

    # Set by the routing node (Member 3)
    intent: NotRequired[Intent]

    # Requested appointment slot
    requested_slot: NotRequired[str | None]
    selected_slot_id: NotRequired[str | None]

    # Structured appointment request and handler results.
    requested_booking_id: NotRequired[str | None]
    booking_id: NotRequired[str | None]
    booking_status: NotRequired[BookingStatus | None]
    approval_id: NotRequired[str | None]
    workflow_error: NotRequired[str | None]
    available_slots: NotRequired[list[dict[str, str]]]
    available_bookings: NotRequired[list[dict[str, str]]]

    # Approval information (Member 4)
    approval_status: NotRequired[ApprovalStatus]

    # Final response returned to the API
    response: NotRequired[str]


def trim_messages_node(state: ConversationState) -> dict:
    """Remove older messages and retain only recent conversation history."""

    messages = state.get("messages", [])
    if len(messages) <= MAX_MESSAGES:
        return {}

    old_messages = messages[:-MAX_MESSAGES]

    return {
        "messages": [
            RemoveMessage(id=message.id) for message in old_messages if message.id is not None
        ]
    }
