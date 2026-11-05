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
    r"not breathing|cannot breathe|can't breathe|"
    r"(?:isn't|aren't|wasn't|weren't)\s+breathing|stopped\s+breathing|"
    r"unable\s+to\s+breathe|not\s+able\s+to\s+breathe|"
    r"breathing\s+is\s+(?:difficult|hard|labored|laborious)|"
    r"(?:trouble|difficulty|difficulties)\s+breathing|"
    r"(?:struggling|struggle|hard|difficult)\s+to\s+breathe|"
    r"(?:can't|cannot)\s+catch\s+(?:a|my|his|her|their)\s+breath|"
    r"(?:trouble|difficulty|difficulties)\s+catching\s+"
    r"(?:a|my|his|her|their)\s+breath|"
    r"gasping|choking|"
    r"severe\s+allergic\s+reaction(?:s)?|"
    r"throat\s+(?:is\s+)?(?:closing|swelling)|"
    r"shortness\s+of\s+breath|short\s+of\s+breath|"
    r"chest\s+pain|heart\s+attack|stroke|seizure|"
    r"severe\s+bleeding|bleeding\s+heavily|overdose|suicidal|anaphylaxis|"
    r"kill\s+myself|hurt\s+myself|end\s+my\s+life|want\s+to\s+die"
    r")\b",
    re.IGNORECASE,
)

_STATEMENT_BOUNDARY_PATTERN = re.compile(
    r"[.;!?,]|\b(?:but|however|yet|although|though|except)\b|"
    r"\band\b(?=\s+(?:(?:now|then|also)\s+)?"
    r"(?:i|you|he|she|we|they|my|his|her|their|our|the\s+patient|"
    r"have|has|had|am|is|are|was|were|feel|feels|felt|can|could|do|does|did|"
    r"need|want|would|will|stopped|struggling|struggle|choking|gasping|"
    r"bleeding)\b)",
    re.IGNORECASE,
)

_NEGATION_WORD_PATTERN = re.compile(
    r"\b(?:no|not|never|without|deny|denies|denied|free\s+of|negative\s+for|"
    r"isn't|aren't|wasn't|weren't|don't|doesn't|didn't|haven't|hasn't|hadn't|"
    r"can't|cannot|won't|wouldn't|no\s+longer)\b",
    re.IGNORECASE,
)

_NEGATION_FILLER_WORDS: Final[frozenset[str]] = frozenset(
    {
        "a",
        "all",
        "also",
        "am",
        "an",
        "and",
        "any",
        "are",
        "at",
        "be",
        "been",
        "being",
        "believe",
        "currently",
        "did",
        "do",
        "does",
        "even",
        "ever",
        "experience",
        "experiencing",
        "feel",
        "feeling",
        "feels",
        "felt",
        "for",
        "had",
        "has",
        "have",
        "having",
        "her",
        "his",
        "history",
        "i",
        "if",
        "in",
        "is",
        "much",
        "my",
        "now",
        "of",
        "on",
        "our",
        "really",
        "right",
        "severe",
        "sharp",
        "some",
        "sure",
        "terrible",
        "that",
        "the",
        "their",
        "them",
        "think",
        "this",
        "to",
        "today",
        "was",
        "we",
        "were",
        "with",
        "your",
    }
)

_HISTORICAL_PATTERN = re.compile(
    r"\b(?:"
    r"last\s+(?:night|week|month|year|time|summer|winter|spring|fall)|"
    r"yesterday|the\s+day\s+before|ago|previously|in\s+the\s+past|used\s+to|"
    r"history\s+of|resolved|no\s+longer|earlier"
    r")\b",
    re.IGNORECASE,
)

_ONGOING_PATTERN = re.compile(
    r"\b(?:now|currently|right\s+now|at\s+the\s+moment|still|since|ongoing|"
    r"today|tonight|again)\b",
    re.IGNORECASE,
)

_STRONG_ONGOING_PATTERN = re.compile(
    r"\b(?:currently|right\s+now|at\s+the\s+moment|still|ongoing)\b",
    re.IGNORECASE,
)

_IMMEDIATE_ONGOING_PATTERN = re.compile(
    r"^\s*(?:right\s+)?(?:now|currently|still|ongoing)\b",
    re.IGNORECASE,
)

_CANCEL_PATTERN = re.compile(
    r"\b(?:"
    r"cancel(?:led|ed|ing|ling|s)?|"
    r"cancell?ations?|"
    r"call(?:ed|ing)?\s+off|"
    r"request(?:ing)?\s+(?:an?\s+|the\s+)?cancellation"
    r")\b",
    re.IGNORECASE,
)

_CANCEL_REFUSAL_PATTERN = re.compile(
    r"\b(?:"
    r"(?:do\s+not|don't|does\s+not|doesn't|did\s+not|didn't|will\s+not|won't|"
    r"would\s+not|wouldn't|never|not)"
    r"(?:\s+\w+){0,4}\s+"
    r"(?:cancel(?:led|ed|ing|ling)?|cancellations?|call(?:ed|ing)?\s+off)|"
    r"no\s+(?:need|intention|plans?)\s+to\s+(?:cancel|cancellations?)|"
    r"(?:want|wish|plan|intend|need)\s+to\s+(?:keep|maintain|leave)\s+"
    r"(?:my|the)\s+appointment"
    r")\b",
    re.IGNORECASE,
)

_RESCHEDULE_PATTERN = re.compile(
    r"\b(?:"
    r"re-?schedul(?:e|ed|es|ing)|"
    r"move\s+(?:my|the|this)\s+(?:appointment|booking)|"
    r"change\s+(?:my|the|this)\s+(?:appointment|booking)"
    r"(?:\s+(?:time|date|slot))?|"
    r"(?:push|postpone)\s+(?:back\s+)?(?:my|the|this)\s+(?:appointment|booking)"
    r")\b",
    re.IGNORECASE,
)

_RESCHEDULE_REFUSAL_PATTERN = re.compile(
    r"\b(?:"
    r"(?:do\s+not|don't|does\s+not|doesn't|did\s+not|didn't|will\s+not|won't|"
    r"would\s+not|wouldn't|never|not)"
    r"(?:\s+\w+){0,4}\s+"
    r"(?:reschedul(?:e|ed|es|ing)|"
    r"move\s+(?:my|the|this)\s+(?:appointment|booking)|"
    r"change\s+(?:my|the|this)\s+(?:appointment|booking))|"
    r"no\s+(?:need|intention|plans?)\s+to\s+(?:reschedule|move|change)\b"
    r")\b",
    re.IGNORECASE,
)

_BOOK_PATTERN = re.compile(
    r"\b(?:"
    r"book(?:ed|ing|s)?|"
    r"(?:make|making|get|getting|request|requesting|arrange|arranging|set\s+up)"
    r"\s+(?:me\s+|us\s+)?(?:an?|the|my|another)?\s*(?:appointment|booking)|"
    r"(?:schedule|scheduling)\s+(?:me|us)\b|"
    r"(?:schedule|scheduling)\s+(?:an?|the|my|another)\s+"
    r"(?:appointment|booking)|"
    r"(?:to|i'?ll|i\s+will|i\s+want\s+to|i\s+need\s+to|i'?d\s+like\s+to|"
    r"i\s+would\s+like\s+to|can\s+i|could\s+i|please|let'?s)\s+schedule\b"
    r")\b",
    re.IGNORECASE,
)

_APPOINTMENT_REQUEST_PATTERN = re.compile(
    r"\b(?:"
    r"(?:i\s+)?(?:need|want|would\s+like|i'?d\s+like)\s+"
    r"(?:(?:an?|the|another|my)\s+)?(?:doctor'?s?\s+)?appointment|"
    r"(?:i\s+)?(?:need|want|would\s+like|i'?d\s+like)\s+"
    r"to\s+see\s+(?:a|the)\s+doctor"
    r")\b",
    re.IGNORECASE,
)

_APPOINTMENT_REFUSAL_PATTERN = re.compile(
    r"\b(?:"
    r"(?:do\s+not|don't|does\s+not|doesn't|did\s+not|didn't|will\s+not|won't|"
    r"would\s+not|wouldn't|never|not|can't|cannot)\s+"
    r"(?:want|need|like|get|book|make|schedule|find)\s+"
    r"(?:to\s+(?:book|make|get|schedule|find)\s+)?"
    r"(?:(?:an?|the|another|my)\s+)?(?:doctor'?s?\s+)?appointment|"
    r"(?:do\s+not|don't|does\s+not|doesn't|did\s+not|didn't|will\s+not|won't|"
    r"would\s+not|wouldn't|never|not|can't|cannot)\s+"
    r"(?:want|need|would\s+like)\s+to\s+see\s+(?:a|the)\s+doctor"
    r")\b",
    re.IGNORECASE,
)

_FIND_SLOTS_PATTERN = re.compile(
    r"\b(?:"
    r"available|availability|slots?|openings?\b(?!\s+hours?\b)|free\s+times?|"
    r"earliest|soonest|find\s+(?:me\s+)?(?:an?\s+)?appointment"
    r")\b",
    re.IGNORECASE,
)

_CLINIC_SCHEDULE_PATTERN = re.compile(
    r"\b(?:"
    r"(?:clinic|opening|working|business|office|weekly|daily|your|the)\s+"
    r"(?:hours?|schedule|timetable)|"
    r"hours?\s+of\s+operation|"
    r"(?:what|which)\s+times?\b[^.!?]*\b(?:open|opening|close|closing|hours?)\b|"
    r"(?:when|what\s+time)\s+(?:do|does|are|is|will)\s+"
    r"(?:you|the\s+clinic|the\s+office)\s+(?:open|close)|"
    r"(?:are|is)\s+(?:you|the\s+clinic|the\s+office)\s+open|"
    r"(?:open|opening|close|closing)\s+hours?"
    r")\b",
    re.IGNORECASE,
)

_CLINIC_QUESTION_PATTERN = re.compile(
    r"\b(?:"
    r"hours?|open|opening|close|closing|located|location|address|phone|"
    r"parking|insurance|services?|prices?|pricing|costs?|fees?|"
    r"polic(?:y|ies)|doctors?|providers?|treatments?|contact|"
    r"schedules?|timetables?"
    r")\b",
    re.IGNORECASE,
)

_APPOINTMENT_INTENTS: Final[frozenset[Intent]] = frozenset(
    {
        "find_slots",
        "book_appointment",
        "cancel_appointment",
        "reschedule_appointment",
    }
)

_CONTEXTUAL_FOLLOW_UP_TOKENS: Final[frozenset[str]] = frozenset(
    {
        "a",
        "about",
        "actually",
        "an",
        "around",
        "at",
        "can",
        "could",
        "do",
        "early",
        "for",
        "how",
        "i",
        "in",
        "instead",
        "it",
        "late",
        "let's",
        "like",
        "make",
        "maybe",
        "next",
        "on",
        "one",
        "please",
        "prefer",
        "sounds",
        "sure",
        "thanks",
        "that",
        "the",
        "this",
        "we",
        "would",
        "works",
        "good",
        "fine",
        "ok",
        "okay",
        "coming",
    }
)

_CONTEXTUAL_TIME_TOKENS: Final[frozenset[str]] = frozenset(
    {
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
        "mon",
        "tue",
        "tues",
        "wed",
        "thu",
        "thur",
        "thurs",
        "fri",
        "sat",
        "sun",
        "today",
        "tomorrow",
        "tonight",
        "morning",
        "afternoon",
        "evening",
        "noon",
        "midnight",
        "am",
        "pm",
        "o'clock",
        "week",
        "weekend",
        "month",
        "year",
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
        "jan",
        "feb",
        "mar",
        "apr",
        "jun",
        "jul",
        "aug",
        "sep",
        "sept",
        "oct",
        "nov",
        "dec",
        "first",
        "second",
        "third",
        "fourth",
        "fifth",
        "earliest",
        "soonest",
        "latest",
    }
)


def _normalized_text(text: str) -> str:
    """Normalize patient text for deterministic keyword matching."""

    return " ".join(text.replace("\u2019", "'").lower().split())


def _statement_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    """Return statement boundaries around a match in normalized text.

    Conjunctions only start a new statement when they introduce a new subject
    or verb. This keeps one negation or historical marker from leaking into a
    later affirmative symptom while preserving shared modifiers such as
    "no chest pain and trouble breathing".
    """

    statement_start = 0
    statement_end = len(text)

    for boundary in _STATEMENT_BOUNDARY_PATTERN.finditer(text):
        if boundary.end() <= start:
            statement_start = boundary.end()
        elif boundary.start() >= end:
            statement_end = boundary.start()
            break

    return statement_start, statement_end


def _is_historical_mention(clause: str, match_start: int, match_end: int) -> bool:
    """Return whether an emergency mention is clearly historical."""

    historical = _HISTORICAL_PATTERN.search(clause)
    if historical is None:
        return False

    if _STRONG_ONGOING_PATTERN.search(clause):
        return False

    if _IMMEDIATE_ONGOING_PATTERN.search(clause[match_end:]):
        return False

    if _ONGOING_PATTERN.search(clause[:match_start]):
        return False

    if historical.start() >= match_end:
        between = clause[match_end : historical.start()]
        if _ONGOING_PATTERN.search(between):
            return False

    return True


def _negation_applies(prefix: str) -> bool:
    """Return whether the nearest negation applies to the upcoming symptom."""

    negation = None
    for candidate in _NEGATION_WORD_PATTERN.finditer(prefix):
        negation = candidate

    if negation is None:
        return False

    between = prefix[negation.end() :]
    if not between.strip():
        return True

    if re.search(r"\bor\b", between, re.IGNORECASE):
        return True

    if _EMERGENCY_PATTERN.search(between):
        return True

    words = re.findall(r"[a-z']+", between.lower())
    return all(word in _NEGATION_FILLER_WORDS for word in words)


def _has_current_emergency(text: str) -> bool:
    """Return whether the text contains an unnegated, current emergency mention."""

    for match in _EMERGENCY_PATTERN.finditer(text):
        start, end = match.span()
        statement_start, statement_end = _statement_bounds(text, start, end)
        prefix = text[statement_start:start]

        if _negation_applies(prefix):
            continue

        statement = text[statement_start:statement_end]
        match_start = start - statement_start
        match_end = end - statement_start

        if _is_historical_mention(statement, match_start, match_end):
            continue

        return True

    return False


def _is_numeric_contextual_token(token: str) -> bool:
    """Return whether a token looks like a time or ordinal date."""

    return (
        re.fullmatch(r"\d{1,2}(?::\d{2})?(?:am|pm)?", token) is not None
        or re.fullmatch(r"\d{1,2}(?:st|nd|rd|th)", token) is not None
    )


def _is_contextual_follow_up(text: str) -> bool:
    """Return whether text looks like a date, time, or slot-selection reply."""

    normalized = _normalized_text(text)
    if not normalized:
        return False

    normalized = normalized.replace("a.m.", "am").replace("p.m.", "pm")
    cleaned = re.sub(r"[^a-z0-9:']+", " ", normalized).strip()
    tokens = cleaned.split()

    if not tokens or len(tokens) > 8:
        return False

    has_time_anchor = False
    for token in tokens:
        if token in _CONTEXTUAL_TIME_TOKENS or _is_numeric_contextual_token(token):
            has_time_anchor = True
        elif token not in _CONTEXTUAL_FOLLOW_UP_TOKENS:
            return False

    return has_time_anchor


def classify_intent(text: str) -> Intent:
    """Classify free-form patient text into one of the supported intents.

    The classifier is deterministic and intentionally conservative. Emergency
    language is checked first for safety; anything unrecognized becomes
    ``"other"`` so the workflow can use its safe fallback response.
    """

    normalized = _normalized_text(text)
    if not normalized:
        return "other"

    if _has_current_emergency(normalized):
        return "emergency"

    cancel_requested = bool(_CANCEL_PATTERN.search(normalized)) and not bool(
        _CANCEL_REFUSAL_PATTERN.search(normalized)
    )
    reschedule_requested = bool(_RESCHEDULE_PATTERN.search(normalized)) and not bool(
        _RESCHEDULE_REFUSAL_PATTERN.search(normalized)
    )
    appointment_refused = bool(_APPOINTMENT_REFUSAL_PATTERN.search(normalized))
    book_requested = (
        bool(_BOOK_PATTERN.search(normalized))
        or bool(_APPOINTMENT_REQUEST_PATTERN.search(normalized))
    ) and not appointment_refused
    slots_requested = bool(_FIND_SLOTS_PATTERN.search(normalized))
    has_appointment_action = (
        cancel_requested or reschedule_requested or book_requested or slots_requested
    )

    if _CLINIC_SCHEDULE_PATTERN.search(normalized) and not has_appointment_action:
        return "clinic_question"

    if cancel_requested:
        return "cancel_appointment"
    if reschedule_requested:
        return "reschedule_appointment"
    if book_requested:
        return "book_appointment"
    if slots_requested:
        return "find_slots"

    # Do not let a negated doctor/appointment request leave a stray "doctor"
    # keyword that the clinic-question fallback would otherwise match.
    clinic_text = (
        _APPOINTMENT_REFUSAL_PATTERN.sub(" ", normalized) if appointment_refused else normalized
    )
    if _CLINIC_QUESTION_PATTERN.search(clinic_text):
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
    raising, so the graph can continue to its safe fallback handler. When the
    latest message is only a date, time, or slot-selection reply and an
    appointment workflow is already active, the previous appointment intent is
    preserved so the workflow can continue.
    """

    message = latest_patient_message(state.get("messages"))
    text = _message_text(message) if message is not None else ""
    intent = classify_intent(text)

    if intent == "other" and _is_contextual_follow_up(text):
        previous_intent = state.get("intent")
        if previous_intent is not None and previous_intent in _APPOINTMENT_INTENTS:
            return {"intent": previous_intent}

    return {"intent": intent}


def route_intent(state: ConversationState) -> RouteDestination:
    """Return the workflow destination for the state's classified intent."""

    intent = state.get("intent")
    if intent is None:
        return FALLBACK_HANDLER
    return INTENT_DESTINATIONS.get(intent, FALLBACK_HANDLER)
