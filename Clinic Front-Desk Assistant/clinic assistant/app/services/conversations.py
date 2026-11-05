"""Coordinate conversation metadata and LangGraph workflow execution."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol
from uuid import UUID

from langchain_core.messages import HumanMessage

from app.graphs.states import ConversationState
from app.repositories.errors import ConversationNotFound
from app.repositories.models import Conversation


class ConversationStore(Protocol):
    """Persistence operations required by the conversation service."""

    def create(self) -> Conversation: ...

    def get(self, conversation_id: UUID) -> Conversation | None: ...

    def touch(self, conversation_id: UUID) -> Conversation | None: ...


class ConversationWorkflow(Protocol):
    """Compiled LangGraph workflow used to process patient messages."""

    async def ainvoke(
        self,
        input: ConversationState,
        config: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]: ...


class ConversationProcessingError(RuntimeError):
    """Raised when a workflow does not return a usable patient response."""


class ConversationService:
    """Create conversations and run messages through the LangGraph workflow."""

    def __init__(
        self,
        repository: ConversationStore,
        workflow: ConversationWorkflow,
    ) -> None:
        self._repository = repository
        self._workflow = workflow

    def create_conversation(self) -> Conversation:
        """Create conversation metadata; its UUID string is the graph thread ID."""
        return self._repository.create()

    def get_conversation(self, conversation_id: UUID) -> Conversation:
        """Return a conversation or raise a domain error when it is missing."""
        conversation = self._repository.get(conversation_id)
        if conversation is None:
            raise ConversationNotFound(f"Conversation {conversation_id} was not found.")
        return conversation

    async def process_message(self, conversation_id: UUID, message: str) -> str:
        """Run one patient message through the workflow and return its response."""
        if not message or not message.strip():
            raise ValueError("Conversation messages cannot be empty.")

        conversation = self.get_conversation(conversation_id)
        state: ConversationState = {
            "conversation_id": str(conversation.id),
            "messages": [HumanMessage(content=message)],
        }
        config = {"configurable": {"thread_id": conversation.thread_id}}
        result = await self._workflow.ainvoke(state, config)

        response = result.get("response")
        if not isinstance(response, str) or not response.strip():
            raise ConversationProcessingError(
                "The conversation workflow did not return a response."
            )

        self._repository.touch(conversation.id)
        return response
