"""Unit tests for deterministic intent classification and graph routing."""

from __future__ import annotations

from typing import cast, get_args

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.graphs.routing import (
    APPOINTMENT_HANDLER,
    CLINIC_QUESTION_HANDLER,
    EMERGENCY_HANDLER,
    FALLBACK_HANDLER,
    INTENT_DESTINATIONS,
    classify_intent,
    classify_intent_node,
    latest_patient_message,
    route_intent,
)
from app.graphs.states import ConversationState, Intent


def _state(
    *messages: HumanMessage | AIMessage,
    intent: str | None = None,
) -> ConversationState:
    state: dict[str, object] = {
        "conversation_id": "conversation-1",
        "messages": list(messages),
    }
    if intent is not None:
        state["intent"] = intent
    return cast(ConversationState, state)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("What are the clinic's opening hours?", "clinic_question"),
        ("Do you have any available slots next week?", "find_slots"),
        ("I would like to book an appointment.", "book_appointment"),
        ("I need to cancel my appointment.", "cancel_appointment"),
        ("Can I reschedule my appointment to Friday?", "reschedule_appointment"),
        ("I have chest pain and need help.", "emergency"),
        ("Tell me a joke.", "other"),
    ],
)
def test_classify_intent_supports_every_intent(text: str, expected: Intent) -> None:
    assert classify_intent(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "I need to cancel my appointment because I have chest pain.",
        "Can I book an appointment? I can't breathe.",
        "I want to reschedule, but I am having a seizure.",
    ],
)
def test_emergency_takes_precedence_over_other_intents(text: str) -> None:
    assert classify_intent(text) == "emergency"


@pytest.mark.parametrize(
    "text",
    ["", "   ", "hello there", "12345"],
)
def test_unknown_or_empty_text_is_other(text: str) -> None:
    assert classify_intent(text) == "other"


def test_intent_destinations_cover_every_supported_intent() -> None:
    assert set(INTENT_DESTINATIONS) == set(get_args(Intent))


@pytest.mark.parametrize(
    ("intent", "destination"),
    [
        ("clinic_question", CLINIC_QUESTION_HANDLER),
        ("find_slots", APPOINTMENT_HANDLER),
        ("book_appointment", APPOINTMENT_HANDLER),
        ("cancel_appointment", APPOINTMENT_HANDLER),
        ("reschedule_appointment", APPOINTMENT_HANDLER),
        ("emergency", EMERGENCY_HANDLER),
        ("other", FALLBACK_HANDLER),
    ],
)
def test_route_intent_maps_every_intent(intent: Intent, destination: str) -> None:
    assert route_intent(_state(intent=intent)) == destination


def test_all_appointment_intents_route_to_appointment_handler() -> None:
    appointment_intents: tuple[Intent, ...] = (
        "find_slots",
        "book_appointment",
        "cancel_appointment",
        "reschedule_appointment",
    )

    assert {route_intent(_state(intent=intent)) for intent in appointment_intents} == {
        APPOINTMENT_HANDLER
    }


def test_route_intent_falls_back_when_intent_is_missing() -> None:
    assert route_intent(_state()) == FALLBACK_HANDLER


def test_route_intent_falls_back_for_unknown_intent() -> None:
    assert route_intent(_state(intent="not_a_real_intent")) == FALLBACK_HANDLER


def test_classify_intent_node_updates_only_intent() -> None:
    state = _state(HumanMessage(content="I need to cancel my appointment."))

    assert classify_intent_node(state) == {"intent": "cancel_appointment"}


def test_classify_intent_node_handles_missing_messages() -> None:
    state = cast(ConversationState, {"conversation_id": "conversation-1"})

    assert classify_intent_node(state) == {"intent": "other"}


def test_classify_intent_node_handles_empty_messages() -> None:
    assert classify_intent_node(_state()) == {"intent": "other"}


def test_classify_intent_node_handles_empty_message_text() -> None:
    assert classify_intent_node(_state(HumanMessage(content="   "))) == {"intent": "other"}


def test_classify_intent_node_ignores_assistant_only_history() -> None:
    state = _state(AIMessage(content="How can I help you today?"))

    assert classify_intent_node(state) == {"intent": "other"}


def test_latest_patient_message_skips_assistant_messages() -> None:
    patient_message = HumanMessage(content="What are your hours?")
    messages = [patient_message, AIMessage(content="Our hours are 9 to 5.")]

    assert latest_patient_message(messages) is patient_message


def test_classify_intent_node_uses_latest_patient_message() -> None:
    state = _state(
        HumanMessage(content="I need to cancel my appointment."),
        AIMessage(content="I can help with that."),
        HumanMessage(content="Actually, what are your opening hours?"),
    )

    assert classify_intent_node(state) == {"intent": "clinic_question"}


def test_classify_intent_node_reads_block_list_content() -> None:
    message = HumanMessage(
        content=[{"type": "text", "text": "I need to reschedule my appointment."}]
    )

    assert classify_intent_node(_state(message)) == {"intent": "reschedule_appointment"}


def test_classify_intent_node_reads_string_blocks_in_content() -> None:
    message = HumanMessage(content=["I would like to", "book an appointment."])

    assert classify_intent_node(_state(message)) == {"intent": "book_appointment"}


def test_latest_patient_message_handles_missing_or_empty_history() -> None:
    assert latest_patient_message(None) is None
    assert latest_patient_message([]) is None
