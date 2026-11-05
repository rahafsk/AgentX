"""FastAPI application and PostgreSQL-backed runtime composition."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from fastapi import APIRouter, FastAPI, HTTPException, Request, Response, status
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import ConnectionPool

from app.api.appointments import router as appointments_router
from app.api.approvals import router as approvals_router
from app.api.conversations import router as conversations_router
from app.config import get_settings
from app.graphs.workflows import build_workflow, make_appointment_handler
from app.llm import LLMSettings, create_llm
from app.rag import ClinicDocumentRetriever
from app.repositories.appointments import AppointmentRepository
from app.repositories.approvals import ApprovalDecision, ApprovalRepository
from app.repositories.bookings import BookingRepository
from app.repositories.conversations import (
    ConversationRepository,
    ConversationRequestClaim,
)
from app.repositories.models import (
    AppointmentSlot,
    Approval,
    AvailableAppointmentSlot,
    Booking,
    Conversation,
)
from app.services.appointments import AppointmentService
from app.services.approvals import ApprovalService
from app.services.clinic_knowledge import ClinicKnowledgeService
from app.services.conversations import ConversationService

api_router = APIRouter()


@api_router.get("/health", tags=["health"])
def health_check(request: Request) -> dict[str, str]:
    """Expose readiness only after required application services are configured."""
    if not getattr(request.app.state, "services_ready", False):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Application services are not ready.",
        )
    return {"status": "ok"}


class _PooledConversationStore:
    """Borrow an independent pooled connection for each repository operation."""

    def __init__(self, pool: ConnectionPool) -> None:
        self._pool = pool

    def create(self, owner_reference: str, request_id: UUID) -> Conversation:
        with self._pool.connection() as connection:
            return ConversationRepository(connection).create(owner_reference, request_id)

    def create_idempotent(
        self,
        owner_reference: str,
        idempotency_key: str,
        request_hash: str,
    ) -> Conversation:
        with self._pool.connection() as connection:
            return ConversationRepository(connection).create_idempotent(
                owner_reference,
                idempotency_key,
                request_hash,
            )

    def claim_request(
        self,
        conversation_id: UUID,
        request_id: UUID,
        request_hash: str,
    ) -> ConversationRequestClaim:
        with self._pool.connection() as connection:
            return ConversationRepository(connection).claim_request(
                conversation_id,
                request_id,
                request_hash,
            )

    def complete_request(
        self,
        conversation_id: UUID,
        request_id: UUID,
        result: dict[str, Any],
    ) -> None:
        with self._pool.connection() as connection:
            ConversationRepository(connection).complete_request(
                conversation_id,
                request_id,
                result,
            )

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
    """Run appointment repository operations with per-operation connections."""

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

    def get_slot(self, slot_id: UUID) -> AppointmentSlot | None:
        with self._pool.connection() as connection:
            return AppointmentRepository(connection).get_slot(slot_id)

    def book_appointment(
        self,
        slot_id: UUID,
        owner_reference: str,
        conversation_id: UUID,
        *,
        approval_workflow: bool,
        request_id: UUID | None,
    ) -> Booking:
        with self._pool.connection() as connection:
            return AppointmentService.from_connection(connection).book_appointment(
                slot_id,
                owner_reference,
                conversation_id,
                approval_workflow=approval_workflow,
                request_id=request_id,
            )

    def request_cancellation(
        self,
        booking_id: UUID,
        conversation_id: UUID,
        owner_reference: str,
        *,
        request_id: UUID | None,
    ) -> Approval:
        with self._pool.connection() as connection:
            return AppointmentService.from_connection(connection).request_cancellation(
                booking_id,
                conversation_id,
                owner_reference,
                request_id=request_id,
            )

    def list_for_conversation(self, conversation_id: UUID) -> list[Booking]:
        with self._pool.connection() as connection:
            return BookingRepository(connection).list_for_conversation(conversation_id)


class _PooledBookingStore:
    """Read booking records using independent pooled connections."""

    def __init__(self, pool: ConnectionPool) -> None:
        self._pool = pool

    def get(self, booking_id: UUID) -> Booking | None:
        with self._pool.connection() as connection:
            return BookingRepository(connection).get(booking_id)

    def list_for_conversation(self, conversation_id: UUID) -> list[Booking]:
        with self._pool.connection() as connection:
            return BookingRepository(connection).list_for_conversation(conversation_id)


class _PooledApprovalStore:
    """Run staff approval operations on independent pooled connections."""

    def __init__(self, pool: ConnectionPool) -> None:
        self._pool = pool

    def get(self, approval_id: UUID) -> Approval | None:
        with self._pool.connection() as connection:
            return ApprovalRepository(connection).get(approval_id)

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


def _uses_external_database_config() -> bool:
    env_file = Path(__file__).resolve().parents[1] / ".env"
    return bool(os.getenv("DATABASE_URL")) or env_file.exists()


def _verify_application_schema(pool: ConnectionPool) -> None:
    """Fail startup if deployment migrations did not create core application tables."""
    with pool.connection() as connection:
        row = connection.execute(
            """
            SELECT to_regclass('doctors'),
                   to_regclass('conversations'),
                   to_regclass('conversation_idempotency'),
                   to_regclass('appointment_slots'),
                   to_regclass('bookings'),
                   to_regclass('approvals'),
                   to_regclass('conversation_request_results')
            """
        ).fetchone()
    if row is None or any(table is None for table in row):
        raise RuntimeError("PostgreSQL schema is not initialized; run the migration command.")


def create_app(
    *,
    appointment_service: Any | None = None,
    conversation_service: Any | None = None,
    approval_service: Any | None = None,
    owner_reference_resolver: Callable[[Request], str | None] | None = None,
    staff_reference_resolver: Callable[[Request], str | None] | None = None,
    configure_services: bool = True,
) -> FastAPI:
    """Build the API with optional injected services and trusted host auth resolvers."""
    injected_services = (
        appointment_service is not None
        and conversation_service is not None
        and approval_service is not None
        and owner_reference_resolver is not None
        and staff_reference_resolver is not None
    )

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.services_ready = False
        if appointment_service is not None:
            application.state.appointment_service = appointment_service
        if conversation_service is not None:
            application.state.conversation_service = conversation_service
        if approval_service is not None:
            application.state.approval_service = approval_service

        if injected_services:
            application.state.services_ready = True
            yield
            application.state.services_ready = False
            return
        if not configure_services or not _uses_external_database_config():
            yield
            return

        settings = get_settings()
        database_url = settings.database_url.get_secret_value()
        pool = ConnectionPool(conninfo=database_url, min_size=1, max_size=10, open=False)
        await asyncio.to_thread(pool.open, wait=True)
        application.state.database_pool = pool

        try:
            await asyncio.to_thread(_verify_application_schema, pool)
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
                appointment_service_runtime = _PooledAppointmentService(pool)
                conversation_store = _PooledConversationStore(pool)
                workflow = build_workflow(
                    knowledge_service=knowledge_service,
                    appointment_handler=make_appointment_handler(
                        appointment_service_runtime,
                        _PooledBookingStore(pool),
                    ),
                    checkpointer=checkpointer,
                )
                await workflow.aget_state(
                    {"configurable": {"thread_id": f"readiness-{UUID(int=0)}"}}
                )
                runtime_conversation_service = ConversationService(
                    conversation_store,
                    workflow,
                )
                application.state.appointment_service = appointment_service_runtime
                application.state.conversation_service = runtime_conversation_service
                application.state.approval_service = ApprovalService(
                    _PooledApprovalStore(pool),
                    _PooledBookingStore(pool),
                    conversation_store,
                    runtime_conversation_service,
                    resume_timeout_seconds=_resume_timeout_seconds(),
                )
                application.state.services_ready = True
                yield
        finally:
            application.state.services_ready = False
            await asyncio.to_thread(pool.close)

    application = FastAPI(
        title="Clinic Front-Desk Assistant API",
        version="0.1.0",
        lifespan=lifespan,
    )
    application.state.services_ready = False
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
        async def attach_authenticated_context(
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


def _resume_timeout_seconds() -> float:
    raw = os.getenv("APPROVAL_RESUME_TIMEOUT_SECONDS", "30")
    try:
        timeout = float(raw)
    except ValueError as exc:
        raise ValueError("APPROVAL_RESUME_TIMEOUT_SECONDS must be a positive number.") from exc
    if timeout <= 0:
        raise ValueError("APPROVAL_RESUME_TIMEOUT_SECONDS must be a positive number.")
    return timeout


app = create_app()
