"""HTTP endpoints for creating conversations and exchanging messages."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Protocol, cast
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from pydantic import BaseModel, Field, field_validator

from app.repositories.errors import ConversationNotFound
from app.repositories.models import Conversation

router = APIRouter(prefix="/conversations", tags=["conversations"])


class ConversationServiceProtocol(Protocol):
    """Service operations required by the conversation API."""

    async def create_conversation(self, owner_reference: str) -> Conversation: ...

    async def get_conversation(
        self,
        conversation_id: UUID,
        *,
        owner_reference: str,
    ) -> Conversation: ...

    async def process_message(
        self,
        conversation_id: UUID,
        message: str,
        *,
        owner_reference: str,
        request_id: UUID,
    ) -> str: ...


class ConversationResponse(BaseModel):
    """Conversation metadata returned to the chat client."""

    id: UUID
    thread_id: str
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_conversation(cls, conversation: Conversation) -> ConversationResponse:
        return cls(
            id=conversation.id,
            thread_id=conversation.thread_id,
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
        )


class MessageRequest(BaseModel):
    """A patient message sent to an existing conversation."""

    message: str = Field(min_length=1, max_length=4000)

    @field_validator("message")
    @classmethod
    def reject_blank_message(cls, value: str) -> str:
        message = value.strip()
        if not message:
            raise ValueError("Message cannot be blank.")
        return message


class MessageResponse(BaseModel):
    """The assistant response returned to the chat client."""

    response: str


def get_conversation_service(request: Request) -> ConversationServiceProtocol:
    """Return the configured conversation service from application state."""
    service = getattr(request.app.state, "conversation_service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Conversation service is not configured.",
        )
    return cast(ConversationServiceProtocol, service)


ConversationServiceDependency = Annotated[
    ConversationServiceProtocol,
    Depends(get_conversation_service),
]

OwnerReference = Annotated[
    str,
    Header(alias="X-Owner-Reference", min_length=1, max_length=200),
]

IdempotencyKey = Annotated[UUID, Header(alias="Idempotency-Key")]


@router.post(
    "",
    response_model=ConversationResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_conversation(
    service: ConversationServiceDependency,
    owner_reference: OwnerReference,
) -> ConversationResponse:
    """Create a conversation whose ID stays stable as the LangGraph thread ID."""
    conversation = await service.create_conversation(owner_reference)
    return ConversationResponse.from_conversation(conversation)


@router.get("/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(
    conversation_id: UUID,
    service: ConversationServiceDependency,
    owner_reference: OwnerReference,
) -> ConversationResponse:
    """Return metadata for one conversation."""
    try:
        conversation = await service.get_conversation(
            conversation_id,
            owner_reference=owner_reference,
        )
    except ConversationNotFound as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found.",
        ) from exc
    return ConversationResponse.from_conversation(conversation)


@router.post("/{conversation_id}/messages", response_model=MessageResponse)
async def send_message(
    conversation_id: UUID,
    payload: MessageRequest,
    service: ConversationServiceDependency,
    owner_reference: OwnerReference,
    request_id: IdempotencyKey,
) -> MessageResponse:
    """Send one patient message through the conversation workflow."""
    try:
        response = await service.process_message(
            conversation_id,
            payload.message,
            owner_reference=owner_reference,
            request_id=request_id,
        )
    except ConversationNotFound as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found.",
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The assistant could not process the message.",
        ) from exc
    return MessageResponse(response=response)
