"""Unit tests for the clinic assistant workflow branches."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast
from uuid import uuid4

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.graphs.states import ConversationState
from app.graphs.workflows import (
    APPROVAL_HANDLER,
    EMERGENCY_RESPONSE,
    FALLBACK_RESPONSE,
    build_workflow,
)
from app.rag import RetrievedChunk
from app.services.clinic_knowledge import ClinicKnowledgeService


class _NeverCalledRetriever:
    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 3,
    ) -> Sequence[RetrievedChunk]:
        raise AssertionError("Retriever must not be called.")


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


async def _fail_appointment(state: ConversationState) -> Mapping[str, Any]:
    raise AssertionError("Appointment handler must not be called.")


def _never_called_service() -> ClinicKnowledgeService:
    return ClinicKnowledgeService(_NeverCalledRetriever())


def _state(text: str) -> ConversationState:
    return cast(
        ConversationState,
        {
            "conversation_id": "conversation-1",
            "messages": [HumanMessage(content=text)],
        },
    )


def _config(thread_id: str) -> RunnableConfig:
    return {"configurable": {"thread_id": thread_id}}


async def test_emergency_returns_safe_response_without_other_handlers() -> None:
    workflow = build_workflow(
        knowledge_service=_never_called_service(),
        appointment_handler=_fail_appointment,
        checkpointer=InMemorySaver(),
    )

    result = await workflow.ainvoke(
        _state("I have chest pain and need help."),
        _config("emergency-1"),
    )

    assert result["response"] == EMERGENCY_RESPONSE
    assert "__interrupt__" not in result


async def test_clinic_question_uses_service_and_preserves_source_name() -> None:
    service = ClinicKnowledgeService(
        _StaticRetriever(
            RetrievedChunk(
                text="Opening hours: the clinic opens Monday to Friday, 9 AM to 6 PM.",
                source="opening_hours.md",
                score=1.0,
            )
        )
    )
    workflow = build_workflow(
        knowledge_service=service,
        appointment_handler=_fail_appointment,
        checkpointer=InMemorySaver(),
    )

    result = await workflow.ainvoke(
        _state("What are the opening hours?"),
        _config("clinic-1"),
    )

    assert "Source: opening_hours.md" in result["response"]
    assert "9 AM to 6 PM" in result["response"]


async def test_other_intent_uses_fallback_without_other_handlers() -> None:
    workflow = build_workflow(
        knowledge_service=_never_called_service(),
        appointment_handler=_fail_appointment,
        checkpointer=InMemorySaver(),
    )

    result = await workflow.ainvoke(
        _state("Tell me a joke."),
        _config("fallback-1"),
    )

    assert result["response"] == FALLBACK_RESPONSE


async def test_normal_appointment_completes_without_approval() -> None:
    calls: list[str] = []

    async def appointment_handler(state: ConversationState) -> Mapping[str, Any]:
        calls.append(str(state.get("intent")))
        return {
            "response": "Your appointment is confirmed.",
            "approval_status": "not_required",
        }

    workflow = build_workflow(
        knowledge_service=_never_called_service(),
        appointment_handler=appointment_handler,
        checkpointer=InMemorySaver(),
    )

    result = await workflow.ainvoke(
        _state("I need an appointment."),
        _config("booking-1"),
    )

    assert result["response"] == "Your appointment is confirmed."
    assert result["approval_status"] == "not_required"
    assert "__interrupt__" not in result
    assert calls == ["book_appointment"]


@pytest.mark.parametrize(
    ("text", "expected_intent"),
    [
        ("I need to cancel my appointment.", "cancel_appointment"),
        ("I need an appointment today.", "book_appointment"),
    ],
)
async def test_approval_requests_pause_and_resume(
    text: str,
    expected_intent: str,
) -> None:
    calls: list[str] = []

    async def appointment_handler(state: ConversationState) -> Mapping[str, Any]:
        calls.append(str(state.get("intent")))
        return {
            "response": "Waiting for staff approval.",
            "approval_status": "pending",
        }

    workflow = build_workflow(
        knowledge_service=_never_called_service(),
        appointment_handler=appointment_handler,
        checkpointer=InMemorySaver(),
    )
    config = _config(f"approval-{expected_intent}")

    first = await workflow.ainvoke(_state(text), config)

    assert first["response"] == "Waiting for staff approval."
    assert first["approval_status"] == "pending"
    assert "__interrupt__" in first

    snapshot = workflow.get_state(config)
    assert snapshot.values["approval_status"] == "pending"
    assert snapshot.next == (APPROVAL_HANDLER,)

    resumed = await workflow.ainvoke(
        Command(resume={"decision": "approved"}),
        config,
    )

    assert resumed["approval_status"] == "approved"
    assert resumed["response"]
    assert calls == [expected_intent]


async def test_rejected_approval_resumes_with_rejected_status() -> None:
    async def appointment_handler(state: ConversationState) -> Mapping[str, Any]:
        return {
            "response": "Waiting for staff approval.",
            "approval_status": "pending",
        }

    workflow = build_workflow(
        knowledge_service=_never_called_service(),
        appointment_handler=appointment_handler,
        checkpointer=InMemorySaver(),
    )
    config = _config("approval-rejected")

    first = await workflow.ainvoke(
        _state("I need to cancel my appointment."),
        config,
    )
    assert "__interrupt__" in first

    resumed = await workflow.ainvoke(
        Command(resume={"decision": "rejected"}),
        config,
    )

    assert resumed["approval_status"] == "rejected"
    assert resumed["response"]


def test_build_workflow_requires_checkpointer_argument() -> None:
    with pytest.raises(TypeError):
        build_workflow(  # type: ignore[call-arg]
            knowledge_service=_never_called_service(),
            appointment_handler=_fail_appointment,
        )


def test_build_workflow_rejects_none_checkpointer() -> None:
    with pytest.raises(ValueError, match="checkpointer is required"):
        build_workflow(
            knowledge_service=_never_called_service(),
            appointment_handler=_fail_appointment,
            checkpointer=None,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "invalid_decision",
    [
        None,
        "maybe",
        ["approved"],
        {"decision": "approved"},
    ],
)
async def test_invalid_approval_decisions_raise_value_error(
    invalid_decision: object,
) -> None:
    async def appointment_handler(state: ConversationState) -> Mapping[str, Any]:
        return {
            "response": "Waiting for staff approval.",
            "approval_status": "pending",
        }

    workflow = build_workflow(
        knowledge_service=_never_called_service(),
        appointment_handler=appointment_handler,
        checkpointer=InMemorySaver(),
    )
    config = _config(f"invalid-approval-{uuid4()}")

    first = await workflow.ainvoke(
        _state("I need to cancel my appointment."),
        config,
    )
    assert "__interrupt__" in first

    with pytest.raises(ValueError, match="Approval decision"):
        await workflow.ainvoke(
            Command(resume={"decision": invalid_decision}),
            config,
        )
