"""FastAPI application entry point for the clinic front-desk assistant."""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any, cast

import psycopg
from fastapi import APIRouter, FastAPI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.api.approvals import router as approvals_router
from app.api.conversations import router as conversations_router
from app.config import get_settings
from app.db.initialize import initialize_database
from app.graphs.workflows import build_workflow, make_appointment_handler
from app.llm import LLMSettings, create_llm
from app.rag import ClinicDocumentRetriever
from app.repositories.appointments import AppointmentRepository
from app.repositories.approvals import ApprovalRepository
from app.repositories.bookings import BookingRepository
from app.repositories.conversations import ConversationRepository
from app.services.appointments import AppointmentService
from app.services.approvals import ApprovalService, ApprovalWorkflow
from app.services.clinic_knowledge import ClinicKnowledgeService
from app.services.conversations import ConversationService, ConversationWorkflow

logger = logging.getLogger(__name__)

api_router = APIRouter()


@api_router.get("/health", tags=["health"])
def health_check() -> dict[str, str]:
    """Report that the API process is ready to serve requests."""
    return {"status": "ok"}


async def _configure_services(
    app: FastAPI,
    stack: AsyncExitStack,
) -> None:
    """Build PostgreSQL-backed services and connect the workflow branches."""
    settings = get_settings()
    dsn = settings.database_url.get_secret_value()

    await asyncio.to_thread(initialize_database, dsn)

    connection = await asyncio.to_thread(_connect, dsn)
    stack.callback(connection.close)

    checkpointer = await stack.enter_async_context(AsyncPostgresSaver.from_conn_string(dsn))

    approval_repository = ApprovalRepository(connection)
    conversation_repository = ConversationRepository(connection)
    booking_repository = BookingRepository(connection)
    appointment_service = AppointmentService(
        AppointmentRepository(connection),
        booking_repository,
        approval_repository,
        conversation_repository,
    )

    knowledge_service = ClinicKnowledgeService(
        ClinicDocumentRetriever(settings.clinic_documents_path),
        llm=create_llm(LLMSettings.from_env()),
    )
    workflow = build_workflow(
        knowledge_service=knowledge_service,
        appointment_handler=make_appointment_handler(
            appointment_service,
            booking_repository,
        ),
        checkpointer=checkpointer,
    )

    app.state.approval_repository = approval_repository
    app.state.conversation_repository = conversation_repository
    app.state.appointment_service = appointment_service
    app.state.workflow = workflow

    conversation_service = ConversationService(
        conversation_repository,
        cast(ConversationWorkflow, workflow),
    )
    app.state.conversation_service = conversation_service
    app.state.approval_service = ApprovalService(
        approval_repository,
        conversation_repository,
        cast(ApprovalWorkflow, workflow),
        conversation_service,
        resume_timeout_seconds=_resume_timeout_seconds(),
    )


def _resume_timeout_seconds() -> float:
    """Read the configurable workflow resume timeout."""
    value = os.getenv("APPROVAL_RESUME_TIMEOUT_SECONDS", "30")
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError("APPROVAL_RESUME_TIMEOUT_SECONDS must be a positive number.") from exc


def _connect(dsn: str) -> psycopg.Connection[Any]:
    """Open one PostgreSQL connection outside the event loop."""
    return psycopg.connect(dsn)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Configure services when PostgreSQL is available."""
    stack = AsyncExitStack()
    try:
        await _configure_services(app, stack)
    except Exception:
        logger.warning(
            "Application services could not be configured; API endpoints will return 503."
        )
        await stack.aclose()
        yield
        return

    try:
        yield
    finally:
        await stack.aclose()


app = FastAPI(
    title="Clinic Front-Desk Assistant API",
    version="0.1.0",
    lifespan=lifespan,
)
app.include_router(api_router)
app.include_router(conversations_router)
app.include_router(approvals_router)
