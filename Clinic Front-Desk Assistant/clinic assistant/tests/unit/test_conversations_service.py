"""Unit tests for the conversation service."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import pytest
from langchain_core.messages import HumanMessage

from app.repositories.errors import ConversationNotFound
from app.repositories.models import Conversation
from app.services.conversations import (
    ConversationProcessingError,
    ConversationService,
)

_CONVERSATION_ID = UUID("12345678-1234-5678-1234-567812345678")
_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _conversation() -> Conversation:
    return Conversation(
        id=_CONVERSATION_ID,
        thread_id=str(_CONVERSATION_ID),
        created_at=_NOW,
        updated_at=_NOW,
    )


class _FakeRepository:
    def __init__(self, conversations: list[Conversation] | None = None) -> None:
        self._conversations = {
            conversation.id: conversation for conversation in conversations or []
        }
        self.created = 0
        self.touched: list[UUID] = []

    def create(self) -> Conversation:
        self.created += 1
        conversation = _conversation()
        self._conversations[conversation.id] = conversation
        return conversation

    def get(self, conversation_id: UUID) -> Conversation | None:
        return self._conversations.get(conversation_id)

    def touch(self, conversation_id: UUID) -> Conversation | None:
        self.touched.append(conversation_id)
        return self._conversations.get(conversation_id)


class _FakeWorkflow:
    def __init__(self, result: dict[str, Any] | None = None) -> None:
        self.result = result if result is not None else {"response": "We can help."}
        self.inputs: list[dict[str, Any]] = []
        self.configs: list[dict[str, Any] | None] = []

    async def ainvoke(
        self,
        input: dict[str, Any],
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.inputs.append(input)
        self.configs.append(config)
        return self.result


def test_create_conversation_delegates_to_repository() -> None:
    repository = _FakeRepository()
    service = ConversationService(repository, _FakeWorkflow())

    conversation = service.create_conversation()

    assert conversation.id == _CONVERSATION_ID
    assert repository.created == 1


def test_get_conversation_returns_existing_conversation() -> None:
    conversation = _conversation()
    service = ConversationService(_FakeRepository([conversation]), _FakeWorkflow())

    assert service.get_conversation(conversation.id) == conversation


def test_get_conversation_raises_when_missing() -> None:
    service = ConversationService(_FakeRepository(), _FakeWorkflow())

    with pytest.raises(ConversationNotFound):
        service.get_conversation(_CONVERSATION_ID)


async def test_process_message_uses_thread_id_and_returns_response() -> None:
    conversation = _conversation()
    repository = _FakeRepository([conversation])
    workflow = _FakeWorkflow({"response": "Appointments are available."})
    service = ConversationService(repository, workflow)

    response = await service.process_message(conversation.id, "I need an appointment.")

    assert response == "Appointments are available."
    assert repository.touched == [conversation.id]
    assert workflow.configs == [
        {"configurable": {"thread_id": conversation.thread_id}},
    ]
    assert workflow.inputs[0]["conversation_id"] == str(conversation.id)
    messages = workflow.inputs[0]["messages"]
    assert len(messages) == 1
    assert isinstance(messages[0], HumanMessage)
    assert messages[0].content == "I need an appointment."


async def test_process_message_rejects_empty_message() -> None:
    repository = _FakeRepository([_conversation()])
    workflow = _FakeWorkflow()
    service = ConversationService(repository, workflow)

    with pytest.raises(ValueError):
        await service.process_message(_CONVERSATION_ID, "   ")

    assert workflow.inputs == []
    assert repository.touched == []


async def test_process_message_raises_when_conversation_is_missing() -> None:
    service = ConversationService(_FakeRepository(), _FakeWorkflow())

    with pytest.raises(ConversationNotFound):
        await service.process_message(_CONVERSATION_ID, "Hello")


async def test_process_message_raises_when_workflow_returns_no_response() -> None:
    conversation = _conversation()
    repository = _FakeRepository([conversation])
    workflow = _FakeWorkflow({"response": ""})
    service = ConversationService(repository, workflow)

    with pytest.raises(ConversationProcessingError):
        await service.process_message(conversation.id, "Hello")

    assert repository.touched == []
