"""Answer clinic questions using only approved clinic documents.

The service retrieves evidence through ``app.rag``, cites source document names,
and falls back honestly when evidence is missing or retrieval fails. It never
diagnoses conditions or recommends treatment.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from app.llm import ChatMessage, ChatModel, LLMError
from app.rag import DocumentLoadError, RetrievedChunk

FALLBACK_ANSWER = (
    "I don't have enough information in the approved clinic documents to answer that. "
    "Please contact the clinic front desk for help."
)

MEDICAL_ADVICE_ANSWER = (
    "I can't provide a diagnosis or recommend treatment. "
    "Please contact the clinic or a qualified healthcare professional for medical advice."
)

_SYSTEM_PROMPT = (
    "You answer questions about clinic information using only the approved document "
    "excerpts supplied by the user. Do not use outside knowledge. Do not diagnose, "
    "recommend treatment, or give medical advice. If the excerpts do not answer the "
    "question, say that you do not have enough information."
)

_MAX_TOP_K = 5
_MAX_PROMPT_EVIDENCE_CHARS = 8_000
_MAX_QUESTION_CHARS = 1_000

_MEDICAL_TERMS = re.compile(
    r"\b(?:"
    r"diagnos\w*|prescri\w*|medicat\w*|medicines?|drugs?|dosage|doses?|"
    r"tablets?|pills?|antibiotics?|insulin|inhalers?|cream|ointment|"
    r"ibuprofen|aspirin|paracetamol|acetaminophen|painkillers?|"
    r"symptoms?|side[-\s]?effects?|pain|fever|cough|cold|flu|covid|"
    r"rash|itch\w*|infection|disease|illness|condition|allerg\w*|"
    r"cancer|diabetes|pregnan\w*|depression|anxiety|asthma|migraine|"
    r"headache|nausea|vomit\w*|dizz\w*|bleed\w*|wound|burn|injur\w*|"
    r"fracture|sprain|swelling|swollen|lump|tumou?r|surgery|therap\w*|"
    r"treat(?:s|ed|ing|ment|ments)?|cure|heal(?:s|ed|ing)?|"
    r"blood pressure|heart attack|stroke|"
    r"chest pain|breath\w*|sore throat|mental health|doctor|hospital|"
    r"mg|ml|milligrams?|milliliters?"
    r")\b",
    re.IGNORECASE,
)

_ADMIN_TOPICS = re.compile(
    r"\b(?:"
    r"appointments?|book|booking|cancel|cancellation|reschedule|"
    r"hours?|open|opening|close|closed|"
    r"address|location|directions?|nearby|near|phone|contact|email|website|"
    r"parking|insurance|billing|payment|cost|price|fees?|"
    r"services?|accessibility|wheelchair|"
    r"documents?|forms?|polic(?:y|ies)|holiday|pharmacy|refill|copy|"
    r"records?|waiting|walk-in|staff"
    r")\b",
    re.IGNORECASE,
)

_EXPLICIT_ADVICE_REQUEST = re.compile(
    r"\bdiagnos(?:e|ed|ing)\b"
    r"|\bprescri(?:be|bed)\b"
    r"|\b(?:treat|cure)\s+(?:my|me|this)\b"
    r"|\b(?:treatment|therapy)\s+for\s+(?:my|me|this)\b"
    r"|\b(?:what|which)\s+(?:treatment|medication|medicine|dose|dosage)\s+"
    r"(?:should|can|could|do|would)\s+(?:i|we)\b"
    r"|\b(?:should|can|could|may|might|do|does|did)\s+(?:i|we)\s+"
    r"(?:take|use|apply|stop|start|increase|decrease)\s+(?:taking\s+)?"
    r"(?:a\s+|an\s+|the\s+|my\s+|your\s+|this\s+)?"
    r"(?:medications?|medicines?|drugs?|pills?|tablets?|doses?|dosages?|"
    r"antibiotics?|insulin|inhalers?|creams?|ointments?|supplements?|"
    r"ibuprofen|aspirin|paracetamol|acetaminophen)\b"
    r"|\b(?:take|use|apply|stop|start|increase|decrease)\s+"
    r"(?:my|your|the|this)\s+(?:medications?|medicines?|drugs?|doses?|"
    r"inhalers?|insulin|creams?|ointments?)\b"
    r"|\b(?:side effects?|symptoms?)\b"
    r"|\b(?:is|are)\s+(?:it|this|that|my|these|those)\b.*"
    r"\b(?:safe|normal|serious|dangerous|contagious|urgent|infected|broken|cancer)\b"
    r"|\b(?:how much|how many|how often)\b.*"
    r"\b(?:take|use|give|dose|dosage|medicine|medication|tablet|pill|mg|ml)\b"
    r"|\b(?:recommend|suggest)\b.*"
    r"\b(?:treatment|medication|medicine|therapy|dose|doctor)\b"
    r"|\b(?:do|can|could|should|may)\s+i\s+have\s+(?:a\s+|an\s+|the\s+)?"
    r"(?:diagnosis|condition|disease|infection|illness|symptom|allergy|"
    r"allergic reaction|fever|rash|pain|headache|migraine|nausea|dizziness|"
    r"diabetes|cancer|cold|flu|covid|pregnan\w*|depression|anxiety)\b",
    re.IGNORECASE,
)

_PERSONAL_ADVICE = re.compile(
    r"\b(?:you|your|the patient|patients?|people)\b[^.]*\b"
    r"(?:should|must|need to|can|may|could)\b[^.]*\b"
    r"(?:take|use|apply|stop|start|increase|decrease|avoid|rest|exercise|"
    r"eat|drink|carry)\b",
    re.IGNORECASE,
)

_UNSAFE_RESPONSE = re.compile(
    r"\bdiagnos\w*\b"
    r"|\bprescri\w*\b"
    r"|\b(?:take|use|apply|give)\s+"
    r"(?:\d+(?:\.\d+)?\s*(?:mg|ml|g|tablets?|pills?|doses?)|"
    r"(?:one|two|three|four|five|six|seven|eight|nine|ten)\s*"
    r"(?:tablets?|pills?|doses?))\b"
    r"|\b(?:take|use|give|apply)\s+"
    r"(?:a\s+|an\s+|the\s+|some\s+)?"
    r"(?:ibuprofen|aspirin|paracetamol|acetaminophen|painkillers?|"
    r"antibiotics?|insulin|inhalers?|medications?|medicines?|drugs?|"
    r"pills?|tablets?)\b"
    r"|\b(?:start|stop|increase|decrease)\s+(?:taking\s+)?"
    r"(?:your\s+|the\s+)?(?:medications?|medicines?|antibiotics?|insulin|doses?)\b"
    r"|\b(?:you|the patient|patients?)\s+(?:should|must|need to)\s+"
    r"(?:rest|drink\s+fluids?|eat\s+light)\b"
    r"|\b(?:you|the patient)\s+(?:may|might|could)\s+have\b"
    r"|\b(?:you|the patient)\s+(?:have|has)\s+(?:a\s+|an\s+|the\s+)?"
    r"(?:diagnosis|condition|disease|infection|illness|symptom|allergy|"
    r"fever|rash|pain|diabetes|cancer|cold|flu|covid|pregnan\w*)\b"
    r"|\b(?:it|this)\s+(?:looks|seems)\s+like\b"
    r"|\b(?:it|this)\s+(?:is|looks like|seems like)\s+(?:a\s+|an\s+|the\s+)?"
    r"(?:infection|disease|illness|condition|cancer|flu|covid|pregnan\w*|"
    r"migraine|rash)\b"
    r"|\b(?:signs?|symptoms?)\s+of\s+(?:a\s+|an\s+|the\s+)?"
    r"(?:infection|disease|illness|condition|cancer|flu|covid|pregnan\w*)\b"
    r"|\b(?:apply|use)\s+(?:the\s+|this\s+)?"
    r"(?:creams?|ointments?|inhalers?|injections?)\b"
    r"|\b(?:take|use)\s+(?:this|the)\s+"
    r"(?:medicines?|medications?|drugs?|inhalers?|creams?)\b"
    r"|\b(?:recommended|recommend|suggested|suggest)\s+"
    r"(?:treatment|therapy|medication|medicine|dose|surgery)\b"
    r"|\b(?:treatment|therapy)\s+for\s+(?:my|me|this|your)\b",
    re.IGNORECASE,
)

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "by",
        "can",
        "for",
        "from",
        "has",
        "have",
        "if",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "our",
        "that",
        "the",
        "their",
        "this",
        "to",
        "was",
        "were",
        "will",
        "with",
        "you",
        "your",
    }
)


class DocumentRetriever(Protocol):
    """Minimal retrieval contract required by the clinic knowledge service."""

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 3,
    ) -> Sequence[RetrievedChunk]:
        """Return relevant chunks from clinic-approved documents."""
        ...


@dataclass(frozen=True, slots=True)
class KnowledgeAnswer:
    """A grounded answer with the source documents used as evidence."""

    answer: str
    sources: tuple[str, ...]
    grounded: bool


class ClinicKnowledgeService:
    """Answer clinic-information questions from approved documents only."""

    def __init__(
        self,
        retriever: DocumentRetriever,
        *,
        llm: ChatModel | None = None,
        top_k: int = 3,
    ) -> None:
        if top_k < 1 or top_k > _MAX_TOP_K:
            raise ValueError(f"top_k must be between 1 and {_MAX_TOP_K}.")
        self._retriever = retriever
        self._llm = llm
        self._top_k = top_k

    async def answer(
        self,
        question: str,
        *,
        top_k: int | None = None,
    ) -> KnowledgeAnswer:
        """Answer a clinic question using only retrieved approved-document evidence."""
        if not isinstance(question, str):
            raise TypeError("question must be text.")

        if len(question) > _MAX_QUESTION_CHARS:
            raise ValueError(f"question must be {_MAX_QUESTION_CHARS} characters or fewer.")

        normalized = question.strip()
        if not normalized:
            raise ValueError("question must not be blank.")

        # Refuse diagnosis/treatment requests before retrieval or model access.
        if _is_medical_advice_request(normalized):
            return KnowledgeAnswer(
                answer=MEDICAL_ADVICE_ANSWER,
                sources=(),
                grounded=False,
            )

        resolved_top_k = self._top_k if top_k is None else top_k
        if resolved_top_k < 1 or resolved_top_k > _MAX_TOP_K:
            raise ValueError(f"top_k must be between 1 and {_MAX_TOP_K}.")

        try:
            chunks = await asyncio.to_thread(
                self._retriever.retrieve,
                normalized,
                top_k=resolved_top_k,
            )
        except (DocumentLoadError, OSError):
            return _fallback_answer()

        chunks = list(chunks)[:resolved_top_k]
        if not chunks:
            return _fallback_answer()

        if self._llm is not None:
            llm_answer = await self._generate_answer(normalized, chunks)

            # Use the model answer only when it is safe and supported by the
            # retrieved evidence. Otherwise, use a document excerpt instead.
            if (
                llm_answer is not None
                and not _contains_unsafe_medical_advice(llm_answer)
                and _answer_is_supported(llm_answer, chunks)
            ):
                used_chunks = _used_chunks(llm_answer, chunks)
                if used_chunks:
                    return _grounded_answer(llm_answer, used_chunks)

        return _excerpt_answer(chunks[0])

    async def _generate_answer(
        self,
        question: str,
        chunks: Sequence[RetrievedChunk],
    ) -> str | None:
        llm = self._llm
        if llm is None:
            return None

        evidence = _evidence_text(chunks)
        messages: list[ChatMessage] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Approved clinic document excerpts:\n\n{evidence}\n\nQuestion: {question}"
                ),
            },
        ]

        try:
            response = (await llm.complete(messages)).strip()
        except LLMError:
            return None

        return response or None


def _evidence_text(chunks: Sequence[RetrievedChunk]) -> str:
    """Build a prompt-size-limited evidence block."""
    parts: list[str] = []
    remaining = _MAX_PROMPT_EVIDENCE_CHARS

    for chunk in chunks:
        part = f"[{chunk.source}] {chunk.text}".strip()
        if len(part) > remaining:
            part = part[:remaining].rstrip()
        if part:
            parts.append(part)
            remaining -= len(part) + 2
        if remaining <= 0:
            break

    return "\n\n".join(parts)


def _tokens(text: str) -> set[str]:
    """Return lowercase alphanumeric tokens without common stop words."""
    return {token for token in _TOKEN_PATTERN.findall(text.casefold()) if token not in _STOP_WORDS}


def _is_medical_advice_request(text: str) -> bool:
    """Return whether a request is asking for diagnosis or treatment advice."""
    if _EXPLICIT_ADVICE_REQUEST.search(text):
        return True
    if not _MEDICAL_TERMS.search(text):
        return False

    # Fail closed: a medical term without a clear clinic-admin topic is treated
    # as a medical-advice request, even if the exact wording is new.
    return not _ADMIN_TOPICS.search(text)


def _contains_unsafe_medical_advice(text: str) -> bool:
    """Return whether text contains diagnosis or treatment advice."""
    if _UNSAFE_RESPONSE.search(text):
        return True
    return bool(_MEDICAL_TERMS.search(text) and _PERSONAL_ADVICE.search(text))


def _is_safe_excerpt(text: str) -> bool:
    """Return whether text is administrative clinic content without medical advice."""
    return bool(_ADMIN_TOPICS.search(text)) and not _contains_unsafe_medical_advice(text)


def _answer_is_supported(
    answer: str,
    chunks: Sequence[RetrievedChunk],
) -> bool:
    """Return whether the model answer is grounded in the retrieved evidence.

    This is intentionally simple: most meaningful answer terms must appear in the
    evidence, and every number in the answer must appear in the evidence.
    """
    answer_tokens = _tokens(answer)
    if not answer_tokens:
        return False

    evidence_tokens = _tokens(" ".join(chunk.text for chunk in chunks))
    overlap = answer_tokens & evidence_tokens
    if len(overlap) / len(answer_tokens) < 0.5:
        return False

    answer_numbers = {token for token in answer_tokens if token.isdigit()}
    return answer_numbers <= evidence_tokens


def _used_chunks(
    answer: str,
    chunks: Sequence[RetrievedChunk],
) -> list[RetrievedChunk]:
    """Return only the chunks whose text overlaps the model answer."""
    answer_tokens = _tokens(answer)
    return [chunk for chunk in chunks if answer_tokens & _tokens(chunk.text)]


def _fallback_answer() -> KnowledgeAnswer:
    return KnowledgeAnswer(
        answer=FALLBACK_ANSWER,
        sources=(),
        grounded=False,
    )


def _grounded_answer(
    text: str,
    chunks: Sequence[RetrievedChunk],
) -> KnowledgeAnswer:
    sources = tuple(dict.fromkeys(chunk.source for chunk in chunks))
    if not sources:
        return _fallback_answer()

    source_suffix = (
        f"Source: {sources[0]}" if len(sources) == 1 else f"Sources: {', '.join(sources)}"
    )
    return KnowledgeAnswer(
        answer=f"{text}\n\n{source_suffix}",
        sources=sources,
        grounded=True,
    )


def _excerpt_answer(chunk: RetrievedChunk) -> KnowledgeAnswer:
    """Return a safe document excerpt, or the fallback when the excerpt is unsafe."""
    if not _is_safe_excerpt(chunk.text):
        return _fallback_answer()

    excerpt = " ".join(chunk.text.split())
    return _grounded_answer(
        f"According to the approved clinic documents: {excerpt}",
        [chunk],
    )


__all__ = [
    "ClinicKnowledgeService",
    "DocumentRetriever",
    "FALLBACK_ANSWER",
    "KnowledgeAnswer",
    "MEDICAL_ADVICE_ANSWER",
]
