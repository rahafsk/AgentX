"""Intent classification and conditional routing for the clinic assistant graph."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Final, Literal

from langchain_core.messages import AnyMessage, HumanMessage

from app.graphs.states import ConversationState, Intent

RouteDestination = Literal[
    "clinic_question_handler",
    "appointment_handler",
    "emergency_handler",
    "fallback_handler",
]

CLINIC_QUESTION_HANDLER: Final[RouteDestination] = "clinic_question_handler"
APPOINTMENT_HANDLER: Final[RouteDestination] = "appointment_handler"
EMERGENCY_HANDLER: Final[RouteDestination] = "emergency_handler"
FALLBACK_HANDLER: Final[RouteDestination] = "fallback_handler"

INTENT_DESTINATIONS: Final[dict[Intent, RouteDestination]] = {
    "clinic_question": CLINIC_QUESTION_HANDLER,
    "find_slots": APPOINTMENT_HANDLER,
    "book_appointment": APPOINTMENT_HANDLER,
    "cancel_appointment": APPOINTMENT_HANDLER,
    "reschedule_appointment": APPOINTMENT_HANDLER,
    "emergency": EMERGENCY_HANDLER,
    "other": FALLBACK_HANDLER,
}

# Emergency detection is deliberately checked before every other intent.
_EMERGENCY_PATTERN = re.compile(
    r"\b(?:"
    r"emergency|911|ambulance|unconscious|unresponsive|"
    r"not breathing|cannot breathe|can't breathe|difficulty breathing|"
    r"shortness of breath|chest pain|heart attack|stroke|seizure|"
    r"severe bleeding|overdose|suicidal|anaphylaxis"
    r")\b",
    re.IGNORECASE,
)

_CANCEL_PATTERN = re.compile(
    r"\b(?:cancel(?:ed|led|ing|ling)?|call(?:ing)? off)\b",
    re.IGNORECASE,
)

_RESCHEDULE_PATTERN = re.compile(
    r"\b(?:"
    r"re-?schedule(?:d|ing)?|"
    r"move (?:my|the|this) appointment|"
    r"change (?:my|the|this) appointment"
    r")\b",
    re.IGNORECASE,
)

_BOOK_PATTERN = re.compile(
    r"\b(?:book|booking|schedule|scheduling)\b",
    re.IGNORECASE,
)

_FIND_SLOTS_PATTERN = re.compile(
    r"\b(?:"
    r"available|availability|slots?|openings?\b(?!\s+hours?\b)|free times?|"
    r"earliest|soonest|what times?"
    r")\b",
    re.IGNORECASE,
)

_CLINIC_QUESTION_PATTERN = re.compile(
    r"\b(?:"
    r"hours?|open|opening|close|closing|located|location|address|phone|"
    r"parking|insurance|services?|prices?|pricing|costs?|fees?|"
    r"polic(?:y|ies)|doctors?|providers?|treatments?|contact"
    r")\b",
    re.IGNORECASE,
)


def _normalized_text(text: str) -> str:
    """Normalize patient text for deterministic keyword matching."""

    return " ".join(text.replace("\u2019", "'").lower().split())


def classify_intent(text: str) -> Intent:
    """Classify free-form patient text into one of the supported intents.

    The classifier is deterministic and intentionally conservative. Emergency
    language is checked first for safety; anything unrecognized becomes
    ``"other"`` so the workflow can use its safe fallback response.
    """

    normalized = _normalized_text(text)
    if not normalized:
        return "other"

    if _EMERGENCY_PATTERN.search(normalized):
        return "emergency"
    if _CANCEL_PATTERN.search(normalized):
        return "cancel_appointment"
    if _RESCHEDULE_PATTERN.search(normalized):
        return "reschedule_appointment"
    if _BOOK_PATTERN.search(normalized):
        return "book_appointment"
    if _FIND_SLOTS_PATTERN.search(normalized):
        return "find_slots"
    if _CLINIC_QUESTION_PATTERN.search(normalized):
        return "clinic_question"
    return "other"


def latest_patient_message(messages: Sequence[AnyMessage] | None) -> HumanMessage | None:
    """Return the most recent patient message, ignoring assistant messages."""

    if not messages:
        return None

    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message
    return None


def _message_text(message: AnyMessage) -> str:
    """Extract text from a message whose content may be a string or block list."""

    content: object = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return " ".join(parts)
    return ""


def classify_intent_node(state: ConversationState) -> dict[str, Intent]:
    """Classify the latest patient message and store the intent in graph state.

    Missing or empty message history is handled as ``"other"`` rather than
    raising, so the graph can continue to its safe fallback handler.
    """

    message = latest_patient_message(state.get("messages"))
    text = _message_text(message) if message is not None else ""
    return {"intent": classify_intent(text)}


def route_intent(state: ConversationState) -> RouteDestination:
    """Return the workflow destination for the state's classified intent."""

    intent = state.get("intent")
    if intent is None:
        return FALLBACK_HANDLER
    return INTENT_DESTINATIONS.get(intent, FALLBACK_HANDLER)
