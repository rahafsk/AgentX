"""Unit tests for the clinic knowledge service."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import NoReturn

import pytest

from app.llm import ChatMessage, ChatModel, FakeLLM, LLMError
from app.rag import ClinicDocumentRetriever, DocumentLoadError, RetrievedChunk
from app.services.clinic_knowledge import (
    FALLBACK_ANSWER,
    MEDICAL_ADVICE_ANSWER,
    ClinicKnowledgeService,
    KnowledgeRetrievalError,
)

OPENING_HOURS = "Opening hours: the clinic opens Monday to Friday, 9 AM to 6 PM."
INSURANCE = "Contact reception to confirm your insurance coverage."


def _docs(tmp_path: Path) -> Path:
    (tmp_path / "opening_hours.md").write_text(OPENING_HOURS, encoding="utf-8")
    (tmp_path / "insurance.txt").write_text(INSURANCE, encoding="utf-8")
    return tmp_path


def _service(
    tmp_path: Path,
    llm: ChatModel | None = None,
    *,
    retriever: ClinicDocumentRetriever | None = None,
) -> ClinicKnowledgeService:
    documents = _docs(tmp_path)
    resolved_retriever = retriever or ClinicDocumentRetriever(documents)
    return ClinicKnowledgeService(retriever=resolved_retriever, llm=llm)


class _NeverCalledLLM:
    async def complete(self, messages: Sequence[ChatMessage]) -> str:
        raise AssertionError("LLM must not be called without evidence.")


class _FailingLLM:
    async def complete(self, messages: Sequence[ChatMessage]) -> NoReturn:
        raise LLMError("provider down")


class _BrokenRetriever:
    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 3,
    ) -> Sequence[RetrievedChunk]:
        raise DocumentLoadError("unreadable file: /secret/clinic-profile.md")


class _NeverCalledRetriever:
    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 3,
    ) -> Sequence[RetrievedChunk]:
        raise AssertionError("Retriever must not be called for medical advice.")


async def test_answer_uses_evidence_and_names_source(tmp_path: Path) -> None:
    service = _service(tmp_path)

    result = await service.answer("what are the opening hours")

    assert result.grounded
    assert result.sources == ("opening_hours.md",)
    assert "Source: opening_hours.md" in result.answer
    assert "9 AM to 6 PM" in result.answer


async def test_missing_evidence_returns_honest_fallback_without_llm(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path, _NeverCalledLLM())

    result = await service.answer("quantum spacecraft purple")

    assert not result.grounded
    assert result.sources == ()
    assert FALLBACK_ANSWER in result.answer


async def test_blank_question_is_rejected(tmp_path: Path) -> None:
    service = _service(tmp_path)

    with pytest.raises(ValueError, match="question must not be blank"):
        await service.answer("   ")


async def test_non_text_question_is_rejected(tmp_path: Path) -> None:
    service = _service(tmp_path)

    with pytest.raises(TypeError, match="question must be text"):
        await service.answer(42)  # type: ignore[arg-type]


async def test_retrieval_error_fails_safely_without_internal_details(
    tmp_path: Path,
) -> None:
    service = ClinicKnowledgeService(retriever=_BrokenRetriever())

    with pytest.raises(KnowledgeRetrievalError) as excinfo:
        await service.answer("what are the opening hours")

    assert "/secret/clinic-profile.md" not in str(excinfo.value)


async def test_fake_llm_answer_keeps_source_suffix(tmp_path: Path) -> None:
    llm = FakeLLM("The clinic opens at 9 AM on weekdays.")
    service = _service(tmp_path, llm)

    result = await service.answer("what are the opening hours")

    assert "The clinic opens at 9 AM on weekdays." in result.answer
    assert "Source: opening_hours.md" in result.answer
    assert result.grounded


async def test_llm_error_degrades_to_sourced_excerpt(tmp_path: Path) -> None:
    service = _service(tmp_path, _FailingLLM())

    result = await service.answer("what are the opening hours")

    assert result.grounded
    assert "9 AM to 6 PM" in result.answer
    assert "Source: opening_hours.md" in result.answer


async def test_blank_llm_response_degrades_to_sourced_excerpt(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path, FakeLLM("   "))

    result = await service.answer("what are the opening hours")

    assert result.grounded
    assert "9 AM to 6 PM" in result.answer
    assert "Source: opening_hours.md" in result.answer


async def test_empty_document_corpus_returns_fallback(tmp_path: Path) -> None:
    service = ClinicKnowledgeService(
        retriever=ClinicDocumentRetriever(tmp_path),
        llm=_NeverCalledLLM(),
    )

    result = await service.answer("what are the opening hours")

    assert not result.grounded
    assert result.sources == ()
    assert result.answer == FALLBACK_ANSWER


@pytest.mark.parametrize(
    "question",
    [
        "Do I have diabetes?",
        "What treatment should I take for a headache?",
        "Can you diagnose this rash?",
        "Should I take ibuprofen?",
    ],
)
async def test_medical_advice_questions_are_refused_without_retrieval_or_llm(
    question: str,
) -> None:
    service = ClinicKnowledgeService(
        retriever=_NeverCalledRetriever(),
        llm=_NeverCalledLLM(),
    )

    result = await service.answer(question)

    assert not result.grounded
    assert result.sources == ()
    assert result.answer == MEDICAL_ADVICE_ANSWER


@pytest.mark.parametrize(
    "unsupported_answer",
    [
        "The clinic offers free parking and is open 24 hours.",
        "The clinic opens at 8 AM on weekdays.",
        "the and is",
    ],
)
async def test_unsupported_llm_answer_uses_document_excerpt(
    tmp_path: Path,
    unsupported_answer: str,
) -> None:
    service = _service(tmp_path, FakeLLM(unsupported_answer))

    result = await service.answer("what are the opening hours")

    assert result.grounded
    assert unsupported_answer not in result.answer
    assert "9 AM to 6 PM" in result.answer
    assert "Source: opening_hours.md" in result.answer


async def test_unsafe_llm_answer_uses_safe_excerpt(tmp_path: Path) -> None:
    service = _service(
        tmp_path,
        FakeLLM("You should take 2 tablets of aspirin daily."),
    )

    result = await service.answer("what are the opening hours")

    assert result.grounded
    assert "aspirin" not in result.answer
    assert "9 AM to 6 PM" in result.answer
    assert "Source: opening_hours.md" in result.answer


async def test_unsafe_excerpt_returns_fallback(tmp_path: Path) -> None:
    (tmp_path / "unsafe.md").write_text(
        "You should take 2 tablets of aspirin daily.",
        encoding="utf-8",
    )
    service = ClinicKnowledgeService(
        retriever=ClinicDocumentRetriever(tmp_path),
    )

    result = await service.answer("aspirin tablets daily")

    assert not result.grounded
    assert result.sources == ()
    assert result.answer == FALLBACK_ANSWER


async def test_llm_answer_names_only_sources_it_uses(tmp_path: Path) -> None:
    service = _service(
        tmp_path,
        FakeLLM("The clinic opens at 9 AM on weekdays."),
    )

    result = await service.answer("insurance coverage and opening hours")

    assert result.sources == ("opening_hours.md",)
    assert "Source: opening_hours.md" in result.answer
    assert "Sources:" not in result.answer


async def test_invalid_top_k_is_rejected(tmp_path: Path) -> None:
    service = _service(tmp_path)

    with pytest.raises(ValueError, match="top_k"):
        await service.answer("what are the opening hours", top_k=0)


def test_invalid_default_top_k_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="top_k"):
        ClinicKnowledgeService(
            retriever=ClinicDocumentRetriever(_docs(tmp_path)),
            top_k=0,
        )
