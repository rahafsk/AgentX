"""Unit tests for owner-safe, bounded conversation service behavior."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import pytest
from langchain_core.messages import HumanMessage, RemoveMessage

from app.graphs.states import MAX_MESSAGES
from app.repositories.conversations import IdempotencyKeyConflict
from app.repositories.errors import ConversationNotFound
from app.repositories.models import Conversation
from app.services.conversations import (
    ConversationProcessingError,
    ConversationService,
)

_CONVERSATION_ID = UUID("12345678-1234-5678-1234-567812345678")
_NOW = datetime(2026, 1, 1, tzinfo=UTC)
_OWNER = "opaque-authenticated-owner"


def _conversation(owner_reference: str = _OWNER) -> Conversation:
    return Conversation(
        id=_CONVERSATION_ID,
        thread_id="graph-thread-not-the-conversation-uuid",
        created_at=_NOW,
        updated_at=_NOW,
        owner_reference=owner_reference,
    )


class _FakeRepository:
    def __init__(self, conversations: list[Conversation] | None = None) -> None:
        self._conversations = {item.id: item for item in conversations or []}
        self.created_owners: list[str] = []
        self.touched: list[tuple[UUID, str]] = []
        self.idempotency: dict[tuple[str, str], Conversation] = {}
        self.request_hashes: dict[tuple[str, str], str] = {}

    def create(self, owner_reference: str) -> Conversation:
        self.created_owners.append(owner_reference)
        conversation = replace(
            _conversation(owner_reference),
            id=uuid4(),
            thread_id=str(uuid4()),
        )
        self._conversations[conversation.id] = conversation
        return conversation

    def create_idempotent(
        self,
        owner_reference: str,
        idempotency_key: str,
        request_hash: str,
    ) -> Conversation:
        key = (owner_reference, idempotency_key)
        existing = self.idempotency.get(key)
        if existing is not None:
            if self.request_hashes[key] != request_hash:
                raise IdempotencyKeyConflict("different request")
            return existing

        conversation = self.create(owner_reference)
        self.idempotency[key] = conversation
        self.request_hashes[key] = request_hash
        return conversation

    def get_for_owner(
        self,
        conversation_id: UUID,
        owner_reference: str,
    ) -> Conversation | None:
        conversation = self._conversations.get(conversation_id)
        if conversation is None or conversation.owner_reference != owner_reference:
            return None
        return conversation

    def touch_for_owner(
        self,
        conversation_id: UUID,
        owner_reference: str,
    ) -> Conversation | None:
        conversation = self.get_for_owner(conversation_id, owner_reference)
        if conversation is not None:
            self.touched.append((conversation_id, owner_reference))
        return conversation


class _Snapshot:
    def __init__(self, values: Mapping[str, Any]) -> None:
        self.values = values


class _FakeWorkflow:
    def __init__(
        self,
        result: dict[str, Any] | None = None,
        *,
        delay_seconds: float = 0,
        state: dict[str, Any] | None = None,
    ) -> None:
        self.result = result if result is not None else {"response": "We can help."}
        self.delay_seconds = delay_seconds
        self.state = dict(state or {})
        self.inputs: list[Any] = []
        self.configs: list[Mapping[str, Any] | None] = []
        self.read_configs: list[dict[str, Any]] = []
        self.active_calls = 0
        self.max_active_calls = 0

    async def aget_state(self, config: Mapping[str, Any]) -> _Snapshot:
        self.read_configs.append(dict(config))
        return _Snapshot(dict(self.state))

    async def ainvoke(
        self,
        input: Any,
        config: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.active_calls += 1
        self.max_active_calls = max(self.max_active_calls, self.active_calls)
        self.inputs.append(input)
        self.configs.append(config)
        try:
            if self.delay_seconds:
                await asyncio.sleep(self.delay_seconds)

            messages = list(self.state.get("messages", []))
            for update in input.get("messages", []):
                if isinstance(update, RemoveMessage):
                    messages = [item for item in messages if item.id != update.id]
                else:
                    match = next(
                        (index for index, item in enumerate(messages) if item.id == update.id),
                        None,
                    )
                    if match is None:
                        messages.append(update)
                    else:
                        messages[match] = update

            self.state.update({key: value for key, value in input.items() if key != "messages"})
            self.state["messages"] = messages
            self.state.update(self.result)
            return dict(self.state)
        finally:
            self.active_calls -= 1


def test_create_conversation_binds_authenticated_owner() -> None:
    repository = _FakeRepository()
    service = ConversationService(repository, _FakeWorkflow())

    conversation = asyncio.run(
        service.create_conversation(
            _OWNER,
            idempotency_key="create-1",
            request_hash="hash-1",
        )
    )

    assert conversation.owner_reference == _OWNER
    assert repository.created_owners == [_OWNER]


def test_create_conversation_retry_returns_original_conversation() -> None:
    repository = _FakeRepository()
    service = ConversationService(repository, _FakeWorkflow())

    first = asyncio.run(
        service.create_conversation(
            _OWNER,
            idempotency_key="create-1",
            request_hash="hash-1",
        )
    )
    second = asyncio.run(
        service.create_conversation(
            _OWNER,
            idempotency_key="create-1",
            request_hash="hash-1",
        )
    )

    assert second == first
    assert repository.created_owners == [_OWNER]


def test_create_conversation_rejects_key_reuse_with_different_content() -> None:
    repository = _FakeRepository()
    service = ConversationService(repository, _FakeWorkflow())

    asyncio.run(
        service.create_conversation(
            _OWNER,
            idempotency_key="create-1",
            request_hash="hash-1",
        )
    )

    with pytest.raises(IdempotencyKeyConflict):
        asyncio.run(
            service.create_conversation(
                _OWNER,
                idempotency_key="create-1",
                request_hash="hash-2",
            )
        )


def test_create_conversation_key_is_scoped_to_owner() -> None:
    repository = _FakeRepository()
    service = ConversationService(repository, _FakeWorkflow())

    first = asyncio.run(
        service.create_conversation(
            "owner-1",
            idempotency_key="shared-key",
            request_hash="hash-1",
        )
    )
    second = asyncio.run(
        service.create_conversation(
            "owner-2",
            idempotency_key="shared-key",
            request_hash="hash-1",
        )
    )

    assert first.id != second.id
    assert repository.created_owners == ["owner-1", "owner-2"]


def test_create_conversation_rejects_empty_owner() -> None:
    service = ConversationService(_FakeRepository(), _FakeWorkflow())

    with pytest.raises(ValueError, match="owner_reference"):
        asyncio.run(
            service.create_conversation(
                "  ",
                idempotency_key="create-1",
                request_hash="hash-1",
            )
        )


def test_get_conversation_checks_owner_without_leaking_existence() -> None:
    conversation = _conversation()
    service = ConversationService(_FakeRepository([conversation]), _FakeWorkflow())

    with pytest.raises(ConversationNotFound, match="was not found"):
        asyncio.run(
            service.get_conversation(
                conversation.id,
                owner_reference="somebody-else",
            )
        )


def test_process_message_uses_stored_thread_id_and_request_id() -> None:
    conversation = _conversation()
    repository = _FakeRepository([conversation])
    workflow = _FakeWorkflow({"response": "Appointments are available."})
    service = ConversationService(repository, workflow)
    request_id = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")

    response = asyncio.run(
        service.process_message(
            conversation.id,
            "I need an appointment.",
            owner_reference=_OWNER,
            request_id=request_id,
        )
    )

    assert response == "Appointments are available."
    assert repository.touched == [(conversation.id, _OWNER)]
    assert workflow.configs == [
        {"configurable": {"thread_id": conversation.thread_id}},
    ]
    assert workflow.inputs[0]["conversation_id"] == str(conversation.id)
    assert workflow.inputs[0]["request_id"] == str(request_id)
    message = workflow.inputs[0]["messages"][-1]
    assert isinstance(message, HumanMessage)
    assert message.id == str(request_id)
    assert message.content == "I need an appointment."


def test_retry_with_same_request_id_does_not_invoke_workflow_twice() -> None:
    conversation = _conversation()
    repository = _FakeRepository([conversation])
    workflow = _FakeWorkflow({"response": "We can help."})
    service = ConversationService(repository, workflow)
    request_id = UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")

    first = asyncio.run(
        service.process_message(
            conversation.id,
            "Hello",
            owner_reference=_OWNER,
            request_id=request_id,
        )
    )
    second = asyncio.run(
        service.process_message(
            conversation.id,
            "Hello",
            owner_reference=_OWNER,
            request_id=request_id,
        )
    )

    assert first == second == "We can help."
    assert len(workflow.inputs) == 1


async def _process_two_messages(
    service: ConversationService,
    conversation_id: UUID,
) -> None:
    await asyncio.gather(
        service.process_message(
            conversation_id,
            "First",
            owner_reference=_OWNER,
            request_id=UUID("cccccccc-cccc-4ccc-8ccc-cccccccccccc"),
        ),
        service.process_message(
            conversation_id,
            "Second",
            owner_reference=_OWNER,
            request_id=UUID("dddddddd-dddd-4ddd-8ddd-dddddddddddd"),
        ),
    )


def test_same_conversation_workflow_turns_are_serialized() -> None:
    conversation = _conversation()
    workflow = _FakeWorkflow({"response": "Done."}, delay_seconds=0.01)
    service = ConversationService(_FakeRepository([conversation]), workflow)

    asyncio.run(_process_two_messages(service, conversation.id))

    assert workflow.max_active_calls == 1


def test_checkpointed_history_is_trimmed_to_configured_limit() -> None:
    conversation = _conversation()
    old_messages = [HumanMessage(content=f"old-{i}", id=f"old-{i}") for i in range(MAX_MESSAGES)]
    workflow = _FakeWorkflow(
        {"response": "The newest turn is handled."},
        state={"messages": old_messages, "response": "stale", "approval_status": "approved"},
    )
    service = ConversationService(_FakeRepository([conversation]), workflow)

    asyncio.run(
        service.process_message(
            conversation.id,
            "Newest turn",
            owner_reference=_OWNER,
            request_id=UUID("eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee"),
        )
    )

    assert len(workflow.state["messages"]) == MAX_MESSAGES
    assert workflow.state["messages"][0].id == "old-1"
    assert workflow.inputs[0]["response"] == ""
    assert workflow.inputs[0]["approval_status"] == "not_required"


def test_approval_pause_returns_pending_response_when_workflow_has_no_text() -> None:
    conversation = _conversation()
    workflow = _FakeWorkflow({"approval_status": "pending", "response": ""})
    service = ConversationService(_FakeRepository([conversation]), workflow)

    response = asyncio.run(
        service.process_message(
            conversation.id,
            "Please book today.",
            owner_reference=_OWNER,
            request_id=UUID("ffffffff-ffff-4fff-8fff-ffffffffffff"),
        )
    )

    assert "waiting for staff review" in response


def test_process_message_rejects_empty_and_oversized_messages() -> None:
    conversation = _conversation()
    workflow = _FakeWorkflow()
    service = ConversationService(
        _FakeRepository([conversation]),
        workflow,
        max_message_characters=5,
    )

    with pytest.raises(ValueError, match="empty"):
        asyncio.run(
            service.process_message(
                conversation.id,
                "   ",
                owner_reference=_OWNER,
                request_id=UUID("11111111-1111-4111-8111-111111111111"),
            )
        )
    with pytest.raises(ValueError, match="5 characters"):
        asyncio.run(
            service.process_message(
                conversation.id,
                "message too long",
                owner_reference=_OWNER,
                request_id=UUID("22222222-2222-4222-8222-222222222222"),
            )
        )
    assert workflow.inputs == []


async def _process_timed_out(service: ConversationService, conversation_id: UUID) -> None:
    await service.process_message(
        conversation_id,
        "Hello",
        owner_reference=_OWNER,
        request_id=UUID("33333333-3333-4333-8333-333333333333"),
    )


def test_workflow_timeout_is_bounded_and_actionable() -> None:
    conversation = _conversation()
    workflow = _FakeWorkflow(delay_seconds=0.05)
    service = ConversationService(
        _FakeRepository([conversation]),
        workflow,
        workflow_timeout_seconds=0.001,
    )

    with pytest.raises(ConversationProcessingError, match="same request_id"):
        asyncio.run(_process_timed_out(service, conversation.id))
