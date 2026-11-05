"""FastAPI application and local runtime composition."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from fastapi import APIRouter, FastAPI, Request, Response
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import ConnectionPool

from app.api.appointments import router as appointments_router
from app.api.approvals import router as approvals_router
from app.api.conversations import router as conversations_router
from app.config import get_settings
from app.db.initialize import initialize_database
from app.graphs.states import ConversationState
from app.graphs.workflows import build_workflow
from app.llm import LLMSettings, create_llm
from app.rag import ClinicDocumentRetriever
from app.repositories.appointments import AppointmentRepository
from app.repositories.approvals import ApprovalDecision, ApprovalRepository
from app.repositories.conversations import ConversationRepository
from app.repositories.errors import (
    ApprovalAlreadyPending,
    ApprovalStateConflict,
    ApprovalWorkflowRequired,
    BookingConflict,
    ConversationNotFound,
    SlotUnavailable,
)
from app.repositories.models import Approval, AvailableAppointmentSlot, Conversation
from app.services.appointments import (
    AppointmentService,
    BookingConversationMismatch,
    BookingNotFound,
)
from app.services.approvals import ApprovalService
from app.services.clinic_knowledge import ClinicKnowledgeService
from app.services.conversations import ConversationService

api_router = APIRouter()


@api_router.get("/health", tags=["health"])
def health_check() -> dict[str, str]:
    """Report that the API process is ready to serve requests."""
    return {"status": "ok"}


class _PooledConversationStore:
    """Borrow a short-lived connection for each conversation repository operation."""

    def __init__(self, pool: ConnectionPool) -> None:
        self._pool = pool

    def create(self, owner_reference: str, request_id: UUID) -> Conversation:
        with self._pool.connection() as connection:
            return ConversationRepository(connection).create(owner_reference, request_id)

    def get(self, conversation_id: UUID) -> Conversation | None:
        with self._pool.connection() as connection:
            return ConversationRepository(connection).get(conversation_id)

    def get_for_owner(
        self,
        conversation_id: UUID,
        owner_reference: str,
    ) -> Conversation | None:
        with self._pool.connection() as connection:
            return ConversationRepository(connection).get_for_owner(
                conversation_id,
                owner_reference,
            )

    def touch_for_owner(
        self,
        conversation_id: UUID,
        owner_reference: str,
    ) -> Conversation | None:
        with self._pool.connection() as connection:
            return ConversationRepository(connection).touch_for_owner(
                conversation_id,
                owner_reference,
            )


class _PooledAppointmentService:
    """Run read-only slot searches using a connection borrowed per request."""

    def __init__(self, pool: ConnectionPool) -> None:
        self._pool = pool

    def search_available_slots(
        self,
        starts_at: datetime,
        ends_at: datetime,
        *,
        doctor_id: UUID | None = None,
        limit: int = 25,
    ) -> list[AvailableAppointmentSlot]:
        with self._pool.connection() as connection:
            return AppointmentRepository(connection).search_open_slots(
                starts_at,
                ends_at,
                doctor_id=doctor_id,
                limit=limit,
            )


class _PooledApprovalStore:
    """Run staff approval operations on short-lived pooled connections."""

    def __init__(self, pool: ConnectionPool) -> None:
        self._pool = pool

    def list_pending(self, *, limit: int = 50) -> list[Approval]:
        with self._pool.connection() as connection:
            return ApprovalRepository(connection).list_pending(limit=limit)

    def decide(
        self,
        approval_id: UUID,
        decision: ApprovalDecision,
        *,
        decided_by: str,
        decision_note: str | None = None,
    ) -> Approval:
        with self._pool.connection() as connection:
            return ApprovalRepository(connection).decide(
                approval_id,
                decision,
                decided_by=decided_by,
                decision_note=decision_note,
            )


def _appointment_workflow_node(
    pool: ConnectionPool,
    state: ConversationState,
) -> Mapping[str, Any]:
    """Execute an appointment action inside the durable graph workflow."""
    owner_reference = state.get("patient_id")
    if not isinstance(owner_reference, str) or not owner_reference.strip():
        return {"workflow_error": "conversation_not_found", "response": "Conversation not found."}

    try:
        conversation_id = UUID(str(state["conversation_id"]))
        raw_request_id = state.get("request_id")
        request_id = UUID(raw_request_id) if isinstance(raw_request_id, str) else None
        with pool.connection() as connection:
            service = AppointmentService.from_connection(connection)
            if state.get("intent") == "book_appointment":
                requested_slot = state.get("requested_slot")
                if not isinstance(requested_slot, str) or not requested_slot:
                    return {
                        "response": "Please select an available appointment slot before booking."
                    }
                slot_id = UUID(requested_slot)
                booking = service.book_appointment(
                    slot_id,
                    owner_reference,
                    conversation_id,
                    approval_workflow=True,
                    request_id=request_id,
                )
                pending = booking.status == "pending_approval"
                return {
                    "booking_id": str(booking.id),
                    "booking_status": booking.status,
                    "approval_id": str(booking.id) if pending else None,
                    "approval_status": "pending" if pending else "not_required",
                    "response": (
                        "Your booking request is waiting for staff approval."
                        if pending
                        else "Your appointment is booked."
                    ),
                }

            if state.get("intent") == "cancel_appointment":
                requested_booking_id = state.get("requested_booking_id")
                if not isinstance(requested_booking_id, str) or not requested_booking_id:
                    return {"response": "Please select the booking you want to cancel."}
                booking_id = UUID(requested_booking_id)
                approval = service.request_cancellation(
                    booking_id,
                    conversation_id,
                    owner_reference,
                    request_id=request_id,
                )
                return {
                    "booking_id": str(approval.booking_id),
                    "approval_id": str(approval.id),
                    "approval_status": approval.status,
                    "response": "Your cancellation request is waiting for staff approval.",
                }

            return {"response": "Please choose an appointment action to continue."}
    except ConversationNotFound:
        return {"workflow_error": "conversation_not_found", "response": "Conversation not found."}
    except BookingNotFound:
        return {"workflow_error": "booking_not_found", "response": "Booking not found."}
    except BookingConversationMismatch:
        return {
            "workflow_error": "booking_conversation_mismatch",
            "response": "The booking does not belong to this conversation.",
        }
    except BookingConflict:
        return {"workflow_error": "booking_conflict", "response": "The slot is already booked."}
    except SlotUnavailable:
        return {"workflow_error": "slot_unavailable", "response": "The slot is unavailable."}
    except ApprovalWorkflowRequired:
        return {
            "workflow_error": "approval_workflow_required",
            "response": "This request requires staff approval.",
        }
    except ApprovalAlreadyPending:
        return {
            "workflow_error": "approval_already_pending",
            "response": "A staff request is already pending.",
        }
    except ApprovalStateConflict:
        return {
            "workflow_error": "approval_state_conflict",
            "response": "The booking is not in a state that can be changed.",
        }
    except (TypeError, ValueError):
        return {"workflow_error": "slot_unavailable", "response": "The request is invalid."}


async def _appointment_handler(
    pool: ConnectionPool,
    state: ConversationState,
) -> Mapping[str, Any]:
    """Keep synchronous psycopg operations off the graph event loop."""
    return await asyncio.to_thread(_appointment_workflow_node, pool, state)


def create_app(
    *,
    appointment_service: Any | None = None,
    conversation_service: Any | None = None,
    approval_service: Any | None = None,
    owner_reference_resolver: Callable[[Request], str | None] | None = None,
    staff_reference_resolver: Callable[[Request], str | None] | None = None,
    configure_services: bool = True,
) -> FastAPI:
    """Build the API and inject services/auth for tests or an embedding host."""
    services_injected = appointment_service is not None and conversation_service is not None

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        if appointment_service is not None:
            application.state.appointment_service = appointment_service
        if conversation_service is not None:
            application.state.conversation_service = conversation_service
        if approval_service is not None:
            application.state.approval_service = approval_service
        if services_injected or not configure_services:
            yield
            return

        env_file = Path(__file__).resolve().parents[1] / ".env"
        if not os.getenv("DATABASE_URL") and not env_file.exists():
            # A bare checkout can still serve health. Domain endpoints fail closed
            # until a DB/checkpointer workflow is configured.
            yield
            return

        settings = get_settings()
        database_url = settings.database_url.get_secret_value()
        await asyncio.to_thread(initialize_database, database_url)
        pool = ConnectionPool(conninfo=database_url, min_size=1, max_size=10, open=False)
        await asyncio.to_thread(pool.open, wait=True)

        try:
            async with AsyncPostgresSaver.from_conn_string(database_url) as checkpointer:
                llm_settings = LLMSettings(
                    mode=settings.llm_mode,
                    provider=settings.llm_provider,
                    model=settings.llm_model,
                    base_url=settings.llm_base_url,
                    api_key=(
                        settings.llm_api_key.get_secret_value()
                        if settings.llm_api_key is not None
                        else None
                    ),
                    timeout_seconds=settings.llm_timeout_seconds,
                )
                knowledge_service = ClinicKnowledgeService(
                    ClinicDocumentRetriever(settings.clinic_documents_path),
                    llm=create_llm(llm_settings),
                )
                workflow = build_workflow(
                    knowledge_service=knowledge_service,
                    appointment_handler=lambda state: _appointment_handler(pool, state),
                    checkpointer=checkpointer,
                )
                application.state.database_pool = pool
                application.state.appointment_service = _PooledAppointmentService(pool)
                application.state.conversation_service = ConversationService(
                    _PooledConversationStore(pool),
                    workflow,
                )
                application.state.approval_service = ApprovalService(
                    _PooledApprovalStore(pool),
                    application.state.conversation_service,
                )
                yield
        finally:
            await asyncio.to_thread(pool.close)

    application = FastAPI(
        title="Clinic Front-Desk Assistant API",
        version="0.1.0",
        lifespan=lifespan,
    )
    if appointment_service is not None:
        application.state.appointment_service = appointment_service
    if conversation_service is not None:
        application.state.conversation_service = conversation_service
    if approval_service is not None:
        application.state.approval_service = approval_service
    application.include_router(api_router)
    application.include_router(appointments_router)
    application.include_router(conversations_router)
    application.include_router(approvals_router)

    if owner_reference_resolver is not None or staff_reference_resolver is not None:

        @application.middleware("http")
        async def attach_authenticated_owner(
            request: Request,
            call_next: Callable[[Request], Awaitable[Response]],
        ) -> Response:
            if owner_reference_resolver is not None:
                owner_reference = owner_reference_resolver(request)
                if owner_reference:
                    request.state.owner_reference = owner_reference
            if staff_reference_resolver is not None:
                staff_reference = staff_reference_resolver(request)
                if staff_reference:
                    request.state.staff_reference = staff_reference
            return await call_next(request)

    return application


app = create_app()
