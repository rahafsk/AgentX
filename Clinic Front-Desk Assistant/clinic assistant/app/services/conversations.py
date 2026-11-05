"""Coordinate owner-scoped conversation metadata and LangGraph execution."""

from __future__ import annotations

import asyncio
import math
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar, cast
from uuid import UUID

from langchain_core.messages import AnyMessage, HumanMessage

from app.graphs.states import MAX_MESSAGES, ConversationState, trim_messages_node
from app.repositories.errors import ConversationNotFound
from app.repositories.models import Conversation

DEFAULT_WORKFLOW_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_MESSAGE_CHARACTERS = 4000
_APPROVAL_PENDING_RESPONSE = (
    "Your request is waiting for staff review. The clinic will update you after a decision."
)
_UNAVAILABLE_RESPONSE = "I couldn't prepare a reply. Please contact the clinic front desk."
_ALREADY_PROCESSED_RESPONSE = (
    "This message was already processed. Please refer to the latest conversation response."
)
_Result = TypeVar("_Result")


class ConversationStore(Protocol):
    """Short, synchronous persistence calls; each call closes its transaction."""

    def create(self, owner_reference: str) -> Conversation: ...

    def get_for_owner(
        self,
        conversation_id: UUID,
        owner_reference: str,
    ) -> Conversation | None: ...

    def touch_for_owner(
        self,
        conversation_id: UUID,
        owner_reference: str,
    ) -> Conversation | None: ...


class ConversationSnapshot(Protocol):
    """Read-only LangGraph checkpoint values for a conversation thread."""

    values: Mapping[str, Any]


class ConversationWorkflow(Protocol):
    """Compiled LangGraph workflow with checkpoint inspection and invocation."""

    async def aget_state(self, config: Mapping[str, Any]) -> ConversationSnapshot: ...

    async def ainvoke(
        self,
        input: ConversationState,
        config: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]: ...


class ConversationProcessingError(RuntimeError):
    """Raised when a workflow times out before a request is checkpointed."""


@dataclass(slots=True)
class _LockEntry:
    lock: asyncio.Lock
    users: int = 0


class ConversationService:
    """Create owner-scoped conversations and invoke their stable LangGraph threads.

    The service serializes a thread within this process. Until a distributed lock
    is configured, run one API worker per process. ``owner_reference`` must come
    from authenticated middleware, never from an untrusted request body.
    """

    def __init__(
        self,
        repository: ConversationStore,
        workflow: ConversationWorkflow,
        *,
        workflow_timeout_seconds: float = DEFAULT_WORKFLOW_TIMEOUT_SECONDS,
        max_message_characters: int = DEFAULT_MAX_MESSAGE_CHARACTERS,
        approval_pending_response: str = _APPROVAL_PENDING_RESPONSE,
    ) -> None:
        if not math.isfinite(workflow_timeout_seconds) or workflow_timeout_seconds <= 0:
            raise ValueError("workflow_timeout_seconds must be finite and greater than zero.")
        if max_message_characters < 1:
            raise ValueError("max_message_characters must be greater than zero.")
        if not approval_pending_response.strip():
            raise ValueError("approval_pending_response cannot be empty.")

        self._repository = repository
        self._workflow = workflow
        self._workflow_timeout_seconds = workflow_timeout_seconds
        self._max_message_characters = max_message_characters
        self._approval_pending_response = approval_pending_response.strip()
        self._repository_lock = asyncio.Lock()
        self._conversation_locks: dict[UUID, _LockEntry] = {}
        self._conversation_locks_guard = asyncio.Lock()

    async def _repository_call(
        self,
        operation: Callable[..., _Result],
        *args: Any,
    ) -> _Result:
        """Run synchronous psycopg work off-loop and serialize the shared connection."""
        async with self._repository_lock:
            return await asyncio.to_thread(operation, *args)

    @asynccontextmanager
    async def _serialize_conversation(
        self,
        conversation_id: UUID,
    ) -> AsyncIterator[None]:
        """Serialize turns for one thread without retaining idle locks forever."""
        async with self._conversation_locks_guard:
            entry = self._conversation_locks.get(conversation_id)
            if entry is None:
                entry = _LockEntry(asyncio.Lock())
                self._conversation_locks[conversation_id] = entry
            entry.users += 1

        try:
            async with entry.lock:
                yield
        finally:
            async with self._conversation_locks_guard:
                entry.users -= 1
                if entry.users == 0:
                    self._conversation_locks.pop(conversation_id, None)

    @staticmethod
    def _validate_owner_reference(owner_reference: str) -> str:
        if not isinstance(owner_reference, str) or not owner_reference.strip():
            raise ValueError("owner_reference cannot be empty.")
        return owner_reference.strip()

    async def create_conversation(self, owner_reference: str) -> Conversation:
        """Create conversation metadata for an authenticated opaque owner ID."""
        owner = self._validate_owner_reference(owner_reference)
        return await self._repository_call(self._repository.create, owner)

    async def get_conversation(
        self,
        conversation_id: UUID,
        *,
        owner_reference: str,
    ) -> Conversation:
        """Return a conversation only when it belongs to the authenticated owner."""
        owner = self._validate_owner_reference(owner_reference)
        conversation = await self._repository_call(
            self._repository.get_for_owner,
            conversation_id,
            owner,
        )
        if conversation is None:
            # Do not reveal whether a conversation owned by someone else exists.
            raise ConversationNotFound("Conversation was not found.")
        return conversation

    @staticmethod
    def _has_request(messages: Sequence[Any], request_id: UUID) -> bool:
        expected = str(request_id)
        return any(str(getattr(item, "id", "")) == expected for item in messages)

    def _response_from_state(self, state: Mapping[str, Any]) -> str:
        response = state.get("response")
        if isinstance(response, str) and response.strip():
            return response
        if state.get("approval_status") == "pending":
            return self._approval_pending_response
        return _UNAVAILABLE_RESPONSE

    def _response_for_replay(
        self,
        state: Mapping[str, Any],
        request_id: UUID,
    ) -> str:
        if state.get("request_id") == str(request_id):
            return self._response_from_state(state)
        return _ALREADY_PROCESSED_RESPONSE

    async def process_message(
        self,
        conversation_id: UUID,
        message: str,
        *,
        owner_reference: str,
        request_id: UUID,
    ) -> str:
        """Process one authorized, idempotent turn with bounded message history.

        The API must derive ``owner_reference`` from authentication and reuse
        ``request_id`` for retries of the same client message.
        """
        if not isinstance(message, str):
            raise TypeError("Conversation messages must be text.")
        if not message.strip():
            raise ValueError("Conversation messages cannot be empty.")
        if len(message) > self._max_message_characters:
            raise ValueError(
                f"Conversation messages cannot exceed {self._max_message_characters} characters."
            )
        owner = self._validate_owner_reference(owner_reference)

        async with self._serialize_conversation(conversation_id):
            conversation = await self.get_conversation(
                conversation_id,
                owner_reference=owner,
            )
            config = {"configurable": {"thread_id": conversation.thread_id}}
            snapshot = await self._workflow.aget_state(config)
            values = snapshot.values
            history = values.get("messages", ())
            if not isinstance(history, Sequence) or isinstance(history, (str, bytes)):
                history = ()

            # LangGraph's add_messages reducer uses message IDs to reconcile turns.
            # Inspect the checkpoint first so retries do not invoke tools twice.
            if self._has_request(history, request_id):
                return self._response_for_replay(values, request_id)

            user_message = HumanMessage(content=message, id=str(request_id))
            combined_history = [*history, user_message]
            trim_state = cast(
                ConversationState,
                {
                    "conversation_id": str(conversation.id),
                    "messages": cast(list[AnyMessage], combined_history),
                },
            )
            trim_updates = trim_messages_node(trim_state)
            removals = trim_updates.get("messages", [])
            excess_count = max(0, len(combined_history) - MAX_MESSAGES)
            if len(removals) < excess_count:
                raise ConversationProcessingError(
                    "Conversation history contains messages that cannot be safely trimmed."
                )

            state: ConversationState = {
                "conversation_id": str(conversation.id),
                "request_id": str(request_id),
                "messages": cast(list[AnyMessage], [*removals, user_message]),
                # Clear previous turn values before this checkpoint is resumed.
                "response": "",
                "approval_status": "not_required",
            }

            # Touch before invocation: if this DB operation fails, no graph side
            # effect has started and a completed checkpoint cannot be misreported.
            touched = await self._repository_call(
                self._repository.touch_for_owner,
                conversation.id,
                owner,
            )
            if touched is None:
                raise ConversationNotFound("Conversation was not found.")

            try:
                result = await asyncio.wait_for(
                    self._workflow.ainvoke(state, config),
                    timeout=self._workflow_timeout_seconds,
                )
            except TimeoutError as exc:
                # The graph may have checkpointed before timing out. Recover the
                # result if possible; otherwise tell the client to reuse request_id.
                latest = await self._workflow.aget_state(config)
                latest_values = latest.values
                latest_messages = latest_values.get("messages", ())
                if isinstance(latest_messages, Sequence) and self._has_request(
                    latest_messages,
                    request_id,
                ):
                    return self._response_for_replay(latest_values, request_id)
                raise ConversationProcessingError(
                    "The workflow timed out. Retry with the same request_id."
                ) from exc

            # An approval interrupt is a normal workflow outcome, not a missing reply.
            return self._response_from_state(result)
