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


@pytest.mark.parametrize(
    "text",
    [
        "I am having trouble breathing.",
        "I'm struggling to breathe.",
        "It is hard to breathe.",
        "My child is choking.",
        "I think my throat is closing.",
        "My throat is swelling.",
        "I can't catch my breath.",
        "I am having trouble catching my breath.",
        "I think I'm having a severe allergic reaction.",
        "He isn't breathing.",
        "He stopped breathing.",
        "I am unable to breathe.",
        "I am not able to breathe.",
        "Breathing is difficult.",
        "My breathing is labored.",
        "I am short of breath.",
        "I am bleeding heavily.",
        "I want to kill myself.",
        "I want to hurt myself.",
        "I want to die.",
    ],
)
def test_common_emergency_phrasings_are_detected(text: str) -> None:
    assert classify_intent(text) == "emergency"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("I do not have chest pain.", "other"),
        ("I'm not choking.", "other"),
        ("No trouble breathing.", "other"),
        ("I have no history of anaphylaxis.", "other"),
        ("I had chest pain last week.", "other"),
        ("I had trouble breathing yesterday.", "other"),
        ("I have a history of anaphylaxis.", "other"),
        ("I previously had a seizure.", "other"),
        ("I used to have chest pain.", "other"),
        ("I am not short of breath.", "other"),
        ("I am not bleeding heavily.", "other"),
        ("I stopped breathing last week.", "other"),
        ("I don't want to die.", "other"),
        ("I had chest pain yesterday and feel fine now.", "other"),
        ("I have no fever or chest pain.", "other"),
        ("I have no fever or trouble breathing.", "other"),
    ],
)
def test_negated_and_historical_emergencies_are_not_current(text: str, expected: Intent) -> None:
    assert classify_intent(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "I had chest pain last week but now I can't breathe.",
            "emergency",
        ),
        (
            "I don't have chest pain, but I am choking.",
            "emergency",
        ),
        (
            "I have no chest pain and no trouble breathing.",
            "other",
        ),
        (
            "I had chest pain last week, but I feel fine now.",
            "other",
        ),
        (
            "I have chest pain since yesterday.",
            "emergency",
        ),
        (
            "I had chest pain last week and still have it.",
            "emergency",
        ),
        (
            "I had chest pain last week and I have chest pain now.",
            "emergency",
        ),
        (
            "I don't have trouble breathing, but my throat is swelling.",
            "emergency",
        ),
        (
            "I have no fever and chest pain.",
            "emergency",
        ),
        (
            "I have no fever and I have chest pain.",
            "emergency",
        ),
        (
            "I have no chest pain and trouble breathing.",
            "other",
        ),
        (
            "I had chest pain yesterday and I have chest pain.",
            "emergency",
        ),
        (
            "I don't have chest pain but I have trouble breathing.",
            "emergency",
        ),
        (
            "I have no fever, but I have chest pain.",
            "emergency",
        ),
        (
            "I have no chest pain, but I am struggling to breathe.",
            "emergency",
        ),
    ],
)
def test_mixed_historical_and_current_emergencies(text: str, expected: Intent) -> None:
    assert classify_intent(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "I want to cancel my appointment because I am choking.",
        "Can I book an appointment? I'm having trouble breathing.",
        "What are your hours? I am having a severe allergic reaction.",
        "I need an appointment because I can't breathe.",
        "I need to see a doctor because I am short of breath.",
        "I need an appointment because my throat is swelling.",
    ],
)
def test_current_emergency_overrides_ordinary_appointment_keywords(text: str) -> None:
    assert classify_intent(text) == "emergency"


@pytest.mark.parametrize(
    "text",
    [
        "What are the clinic's working hours?",
        "What times are you open?",
        "What is the clinic schedule?",
        "When does the clinic open?",
        "What time do you close?",
        "What are your hours of operation?",
        "What are the clinic's opening hours?",
        "Are you open on Saturday?",
        "I don't need to see a doctor, what is your address?",
        "I don't want an appointment, what are your hours?",
    ],
)
def test_clinic_hours_and_schedule_questions_are_not_bookings(text: str) -> None:
    assert classify_intent(text) == "clinic_question"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("I need an appointment.", "book_appointment"),
        ("I would like an appointment.", "book_appointment"),
        ("I want an appointment.", "book_appointment"),
        ("I need a doctor's appointment.", "book_appointment"),
        ("I need to see a doctor.", "book_appointment"),
        ("I need to find an appointment.", "find_slots"),
        ("I don't want an appointment, do you have available slots?", "find_slots"),
        ("I don't have chest pain but I need an appointment.", "book_appointment"),
        ("I'd like to make an appointment.", "book_appointment"),
        ("How do I get an appointment?", "book_appointment"),
        ("Can I get an appointment with a doctor?", "book_appointment"),
        ("Please schedule me for Friday.", "book_appointment"),
        ("Can I schedule an appointment for Friday?", "book_appointment"),
        ("I need to book an appointment with a doctor.", "book_appointment"),
        ("I need to reschedule.", "reschedule_appointment"),
        ("I want to reschedule my appointment.", "reschedule_appointment"),
        ("I am rescheduling my appointment.", "reschedule_appointment"),
        ("I want to change my appointment time.", "reschedule_appointment"),
        ("I need to move my appointment to Friday.", "reschedule_appointment"),
        ("I need to postpone my appointment.", "reschedule_appointment"),
        ("I need to request a cancellation.", "cancel_appointment"),
        ("Please call off my appointment.", "cancel_appointment"),
        ("Can I cancel my appointment?", "cancel_appointment"),
        ("What appointment times are available?", "find_slots"),
    ],
)
def test_common_appointment_phrases_are_recognized(text: str, expected: Intent) -> None:
    assert classify_intent(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("I don't want to cancel my appointment.", "other"),
        ("I do not want to cancel my appointment.", "other"),
        ("Please don't cancel my appointment.", "other"),
        ("I'm not cancelling my appointment.", "other"),
        ("I am not going to cancel my appointment.", "other"),
        ("I want to keep my appointment.", "other"),
        (
            "I don't want to cancel, I want to reschedule.",
            "reschedule_appointment",
        ),
        (
            "Please don't cancel my appointment; I'd like to reschedule.",
            "reschedule_appointment",
        ),
        (
            "I want to book an appointment, not cancel it.",
            "book_appointment",
        ),
        (
            "I need to reschedule my appointment, not cancel it.",
            "reschedule_appointment",
        ),
    ],
)
def test_cancellation_refusals_do_not_trigger_cancellation(text: str, expected: Intent) -> None:
    assert classify_intent(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "I don't want an appointment.",
        "I don't need an appointment.",
        "I don't want to book an appointment.",
        "I don't need to see a doctor.",
    ],
)
def test_appointment_refusals_do_not_trigger_booking(text: str) -> None:
    assert classify_intent(text) == "other"


@pytest.mark.parametrize(
    "text",
    [
        "I don't want to reschedule my appointment.",
        "I do not need to reschedule my appointment.",
        "I don't want to move my appointment.",
        "I don't need to change my appointment.",
        "I never said I wanted to reschedule.",
    ],
)
def test_rescheduling_refusals_do_not_trigger_rescheduling(text: str) -> None:
    assert classify_intent(text) == "other"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "I don't want to reschedule, but I need to book a new appointment.",
            "book_appointment",
        ),
        (
            "I don't want to reschedule, but I want to cancel.",
            "cancel_appointment",
        ),
    ],
)
def test_rescheduling_refusal_preserves_separate_affirmative_action(
    text: str, expected: Intent
) -> None:
    assert classify_intent(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "I need to cancel my appointment and book a new one.",
            "cancel_appointment",
        ),
        (
            "I want to check available slots and book an appointment.",
            "book_appointment",
        ),
        (
            "Can I reschedule my appointment to Friday afternoon?",
            "reschedule_appointment",
        ),
    ],
)
def test_multiple_appointment_keywords_keep_priority(text: str, expected: Intent) -> None:
    assert classify_intent(text) == expected


@pytest.mark.parametrize(
    ("previous_intent", "text", "expected"),
    [
        ("book_appointment", "Friday afternoon", "book_appointment"),
        ("find_slots", "next Tuesday at 3 pm", "find_slots"),
        ("reschedule_appointment", "tomorrow morning", "reschedule_appointment"),
        ("cancel_appointment", "Friday", "cancel_appointment"),
        ("book_appointment", "Let's do Friday afternoon, please", "book_appointment"),
        ("book_appointment", "the first one", "book_appointment"),
        ("cancel_appointment", "option 2", "cancel_appointment"),
        ("find_slots", "choice 1", "find_slots"),
    ],
)
def test_classify_intent_node_preserves_active_appointment_intent(
    previous_intent: Intent, text: str, expected: Intent
) -> None:
    state = _state(HumanMessage(content=text), intent=previous_intent)

    assert classify_intent_node(state) == {"intent": expected}


@pytest.mark.parametrize(
    ("previous_intent", "text"),
    [
        ("book_appointment", "Tell me a joke."),
        ("book_appointment", "I had chest pain last week"),
        ("book_appointment", "I have a question about Friday."),
        ("clinic_question", "Friday afternoon"),
        (None, "Friday afternoon"),
    ],
)
def test_classify_intent_node_does_not_preserve_for_unrelated_context(
    previous_intent: Intent | None, text: str
) -> None:
    state = _state(HumanMessage(content=text), intent=previous_intent)

    assert classify_intent_node(state) == {"intent": "other"}


def test_classify_intent_node_allows_new_request_to_change_intent() -> None:
    state = _state(
        HumanMessage(content="I need to cancel my appointment."),
        intent="book_appointment",
    )

    assert classify_intent_node(state) == {"intent": "cancel_appointment"}


def test_classify_intent_node_lets_emergency_override_active_appointment_intent() -> None:
    state = _state(
        HumanMessage(content="I can't breathe."),
        intent="book_appointment",
    )

    assert classify_intent_node(state) == {"intent": "emergency"}


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
