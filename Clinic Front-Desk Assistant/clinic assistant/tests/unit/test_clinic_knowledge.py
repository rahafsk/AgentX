"""Unit tests for the clinic knowledge service."""

from __future__ import annotations

import threading
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


class _ThreadRecordingRetriever:
    def __init__(self) -> None:
        self.thread_id: int | None = None

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 3,
    ) -> Sequence[RetrievedChunk]:
        self.thread_id = threading.get_ident()
        return [
            RetrievedChunk(
                text="Opening hours are 9 AM to 5 PM.",
                source="hours.md",
                score=1.0,
            )
        ]


class _CapturingLLM:
    def __init__(self) -> None:
        self.messages: list[ChatMessage] | None = None

    async def complete(self, messages: Sequence[ChatMessage]) -> str:
        self.messages = list(messages)
        return "The clinic opens at 9 AM."


class _StaticRetriever:
    def __init__(self, *chunks: RetrievedChunk) -> None:
        self._chunks = chunks

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 3,
    ) -> Sequence[RetrievedChunk]:
        return list(self._chunks[:top_k])


class _HugeChunkRetriever:
    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 3,
    ) -> Sequence[RetrievedChunk]:
        return [
            RetrievedChunk(
                text="clinic opens 9 AM " + ("x" * 5_000),
                source=f"doc-{index}.md",
                score=1.0,
            )
            for index in range(top_k)
        ]


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


async def test_retrieval_error_returns_fallback(tmp_path: Path) -> None:
    service = ClinicKnowledgeService(retriever=_BrokenRetriever())

    result = await service.answer("what are the opening hours")

    assert result.answer == FALLBACK_ANSWER
    assert result.sources == ()
    assert not result.grounded
    assert "/secret/clinic-profile.md" not in result.answer


async def test_oversized_question_is_rejected() -> None:
    service = ClinicKnowledgeService(
        retriever=_NeverCalledRetriever(),
        llm=_NeverCalledLLM(),
    )

    with pytest.raises(ValueError, match="1000 characters or fewer"):
        await service.answer("x" * 1001)


async def test_retrieval_runs_outside_event_loop() -> None:
    retriever = _ThreadRecordingRetriever()
    service = ClinicKnowledgeService(retriever)

    await service.answer("opening hours")

    assert retriever.thread_id is not None
    assert retriever.thread_id != threading.get_ident()


async def test_llm_prompt_is_size_limited() -> None:
    llm = _CapturingLLM()
    service = ClinicKnowledgeService(_HugeChunkRetriever(), llm=llm)

    await service.answer("opening hours", top_k=5)

    assert llm.messages is not None
    user_message = llm.messages[1]["content"]
    assert len(user_message) < 10_000


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


async def test_routine_insurance_question_reaches_retrieval(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)

    result = await service.answer("Do I have to bring my insurance card?")

    assert result.grounded
    assert result.sources == ("insurance.txt",)
    assert "insurance coverage" in result.answer
    assert result.answer != MEDICAL_ADVICE_ANSWER


async def test_routine_appointment_question_is_not_medical_refusal(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)

    result = await service.answer("Can I have an appointment?")

    assert result.answer == FALLBACK_ANSWER
    assert result.sources == ()
    assert not result.grounded


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
        "Clinic notice: You should take 2 tablets of aspirin daily.",
        encoding="utf-8",
    )
    service = ClinicKnowledgeService(
        retriever=ClinicDocumentRetriever(tmp_path),
    )

    result = await service.answer("clinic notice")

    assert not result.grounded
    assert result.sources == ()
    assert result.answer == FALLBACK_ANSWER


async def test_standalone_medication_instruction_excerpt_returns_fallback() -> None:
    service = ClinicKnowledgeService(
        retriever=_StaticRetriever(
            RetrievedChunk(
                text="Take ibuprofen for headaches.",
                source="unsafe.md",
                score=1.0,
            )
        ),
    )

    result = await service.answer("what are the clinic hours")

    assert result.answer == FALLBACK_ANSWER
    assert result.sources == ()
    assert not result.grounded


async def test_administrative_content_does_not_make_medication_instruction_safe() -> None:
    service = ClinicKnowledgeService(
        retriever=_StaticRetriever(
            RetrievedChunk(
                text="Clinic hours are 9 AM to 5 PM. Take ibuprofen for headaches.",
                source="unsafe.md",
                score=1.0,
            )
        ),
    )

    result = await service.answer("what are the clinic hours")

    assert result.answer == FALLBACK_ANSWER
    assert result.sources == ()
    assert not result.grounded


async def test_safe_administrative_excerpt_still_succeeds() -> None:
    service = ClinicKnowledgeService(
        retriever=_StaticRetriever(
            RetrievedChunk(
                text="Clinic hours are 9 AM to 5 PM.",
                source="hours.md",
                score=1.0,
            )
        ),
    )

    result = await service.answer("what are the clinic hours")

    assert result.grounded
    assert result.sources == ("hours.md",)
    assert "9 AM to 5 PM" in result.answer


@pytest.mark.parametrize(
    "question",
    [
        "I have chest pain, what should I do?",
        "Is my rash serious?",
        "How many antibiotics should I give my child?",
        "What can I put on a burn?",
        "My doctor told me to double my dose, is that okay?",
        "Could I be pregnant?",
        "What medicine is best for a cough?",
        "What medicine is available for my cough?",
        "Can you provide a diagnosis?",
        "What is my diagnosis?",
        "Is it safe to exercise with asthma?",
    ],
)
async def test_paraphrased_medical_requests_are_refused(question: str) -> None:
    service = ClinicKnowledgeService(
        retriever=_NeverCalledRetriever(),
        llm=_NeverCalledLLM(),
    )

    result = await service.answer(question)

    assert result.answer == MEDICAL_ADVICE_ANSWER
    assert result.sources == ()
    assert not result.grounded


@pytest.mark.parametrize(
    "unsafe_text",
    [
        "Take 500 mg of paracetamol every six hours.",
        "Apply the cream twice daily.",
        "You may have an infection; start antibiotics.",
        "Diagnosis: influenza.",
        "Use this inhaler when wheezing.",
        "Patients with asthma should carry an inhaler.",
        "You should rest and drink fluids.",
    ],
)
async def test_unsafe_excerpt_phrasings_return_fallback(
    tmp_path: Path,
    unsafe_text: str,
) -> None:
    (tmp_path / "notice.md").write_text(
        f"Clinic notice: {unsafe_text}",
        encoding="utf-8",
    )
    service = ClinicKnowledgeService(
        retriever=ClinicDocumentRetriever(tmp_path),
    )

    result = await service.answer("clinic notice")

    assert result.answer == FALLBACK_ANSWER
    assert result.sources == ()
    assert not result.grounded


@pytest.mark.parametrize(
    "unsafe_response",
    [
        "You may have an infection. Start antibiotics and take 500 mg.",
        "Apply the cream twice daily.",
        "Diagnosis: influenza. You should rest and drink fluids.",
    ],
)
async def test_unsafe_model_responses_are_not_returned(
    tmp_path: Path,
    unsafe_response: str,
) -> None:
    service = _service(tmp_path, FakeLLM(unsafe_response))

    result = await service.answer("what are the opening hours")

    assert unsafe_response not in result.answer
    assert "9 AM to 6 PM" in result.answer
    assert "Source: opening_hours.md" in result.answer


@pytest.mark.parametrize(
    "question",
    [
        "Do I have to bring my insurance card?",
        "Can I have an appointment?",
        "What treatment services do you offer?",
        "What physical therapy services do you offer?",
        "Is physical therapy covered by insurance?",
        "Can I have a copy of my prescription?",
    ],
)
async def test_routine_clinic_questions_are_not_refused(
    tmp_path: Path,
    question: str,
) -> None:
    service = _service(tmp_path)

    result = await service.answer(question)

    assert result.answer != MEDICAL_ADVICE_ANSWER


async def test_llm_answer_names_only_sources_it_uses(tmp_path: Path) -> None:
    service = _service(
        tmp_path,
        FakeLLM("The clinic opens at 9 AM on weekdays."),
    )

    result = await service.answer("insurance coverage and opening hours")

    assert result.sources == ("opening_hours.md",)
    assert "Source: opening_hours.md" in result.answer
    assert "Sources:" not in result.answer


@pytest.mark.parametrize("top_k", [0, 1000])
async def test_invalid_answer_top_k_is_rejected(
    tmp_path: Path,
    top_k: int,
) -> None:
    service = _service(tmp_path)

    with pytest.raises(ValueError, match="top_k"):
        await service.answer("what are the opening hours", top_k=top_k)


@pytest.mark.parametrize("top_k", [0, 1000])
def test_invalid_default_top_k_is_rejected(
    tmp_path: Path,
    top_k: int,
) -> None:
    with pytest.raises(ValueError, match="top_k"):
        ClinicKnowledgeService(
            retriever=ClinicDocumentRetriever(_docs(tmp_path)),
            top_k=top_k,
        )
