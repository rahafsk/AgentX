"""Answer clinic questions using only approved clinic documents.

The service retrieves evidence through ``app.rag``, cites source document names,
and falls back honestly when evidence is missing or retrieval fails. It never
diagnoses conditions or recommends treatment.
"""

from __future__ import annotations

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

_MEDICAL_ADVICE_REQUEST = re.compile(
    r"\bdiagnos(?:e|is|ed|ing)\b"
    r"|\bprescri(?:be|bed|ption)\b"
    r"|\b(?:should|can|may|do)\s+i\s+(?:take|use|stop|start)\b"
    r"|\b(?:what|which)\s+(?:treatment|medication|medicine|dose|dosage)\s+"
    r"(?:should|can|do|would)\s+i\b"
    r"|\b(?:is it safe|side effects?|symptoms?)\b"
    r"|\b(?:do|can)\s+i\s+have\b"
    r"|\b(?:treat|cure)\s+(?:my|me|this|a|an)\b"
    r"|\b(?:treatment|therapy)\s+(?:for|recommendation|advice)\b"
    r"|\b(?:recommend|suggest)\s+(?:a\s+)?(?:treatment|medication|medicine|therapy)\b",
    re.IGNORECASE,
)

_TREATMENT_ADVICE = re.compile(
    r"\bdiagnos(?:e|is|ed|ing)\b"
    r"|\bprescri(?:be|bed|ption)\b"
    r"|\b(?:you|the patient)\s+(?:should|must|need to|can)\s+"
    r"(?:take|use|apply|stop|start|increase|decrease)\b"
    r"|\b(?:take|use)\s+\d+(?:\.\d+)?\s*(?:mg|ml|tablets?|pills?|doses?)\b"
    r"|\b(?:recommended|recommend|suggested|suggest)\s+"
    r"(?:treatment|therapy|medication|medicine)\b"
    r"|\b(?:treatment|therapy)\s+(?:for|recommendation|advice)\b",
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


class KnowledgeRetrievalError(RuntimeError):
    """Raised when approved clinic documents cannot be retrieved safely."""


class ClinicKnowledgeService:
    """Answer clinic-information questions from approved documents only."""

    def __init__(
        self,
        retriever: DocumentRetriever,
        *,
        llm: ChatModel | None = None,
        top_k: int = 3,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be greater than zero.")
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

        normalized = question.strip()
        if not normalized:
            raise ValueError("question must not be blank.")

        # Refuse diagnosis/treatment requests before retrieval or model access.
        if _MEDICAL_ADVICE_REQUEST.search(normalized):
            return KnowledgeAnswer(
                answer=MEDICAL_ADVICE_ANSWER,
                sources=(),
                grounded=False,
            )

        resolved_top_k = self._top_k if top_k is None else top_k
        if resolved_top_k < 1:
            raise ValueError("top_k must be greater than zero.")

        try:
            chunks = self._retriever.retrieve(normalized, top_k=resolved_top_k)
        except (DocumentLoadError, OSError) as exc:
            raise KnowledgeRetrievalError("Could not retrieve approved clinic documents.") from exc

        if not chunks:
            return _fallback_answer()

        if self._llm is not None:
            llm_answer = await self._generate_answer(normalized, chunks)

            # Use the model answer only when it is safe and supported by the
            # retrieved evidence. Otherwise, use a document excerpt instead.
            if (
                llm_answer is not None
                and not _TREATMENT_ADVICE.search(llm_answer)
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

        evidence = "\n\n".join(f"[{chunk.source}] {chunk.text}" for chunk in chunks)
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


def _tokens(text: str) -> set[str]:
    """Return lowercase alphanumeric tokens without common stop words."""
    return {token for token in _TOKEN_PATTERN.findall(text.casefold()) if token not in _STOP_WORDS}


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
    if _TREATMENT_ADVICE.search(chunk.text):
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
    "KnowledgeRetrievalError",
    "MEDICAL_ADVICE_ANSWER",
]
