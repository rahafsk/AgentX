
"""Shared state definitions for the clinic assistant workflow."""

from typing import Annotated, Literal, NotRequired, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


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


class ConversationState(TypedDict):
    """State shared between conversation workflow nodes."""

    # Shared conversation identifiers
    conversation_id: str
    patient_id: str

    # Conversation messages managed by LangGraph
    messages: Annotated[list[AnyMessage], add_messages]

    # Set by the routing node (Member 3)
    intent: NotRequired[Intent]

    # Requested appointment slot, coordinated with Member 3
    requested_slot: NotRequired[str | None]

    # Approval information, coordinated with Member 4
    approval_status: NotRequired[ApprovalStatus]

    # Final response returned to the API
    response: NotRequired[str]


   