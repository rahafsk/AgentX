"""Database-backed appointment approval pause, restart, and resume coverage."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo

import pytest
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import ConnectionPool

from app.graphs.states import ConversationState
from app.graphs.workflows import build_workflow
from app.rag import RetrievedChunk
from app.repositories.appointments import AppointmentRepository
from app.repositories.approvals import ApprovalRepository
from app.repositories.conversations import ConversationRepository
from app.repositories.models import Approval, ApprovalDecision, Conversation
from app.services.appointments import AppointmentService
from app.services.approvals import ApprovalService
from app.services.clinic_knowledge import ClinicKnowledgeService
from app.services.conversations import ConversationService


class _NeverCalledRetriever:
    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 3,
    ) -> Sequence[RetrievedChunk]:
        raise AssertionError("Appointment workflow must not query clinic documents.")


class _PooledConversationStore:
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


class _PooledApprovalStore:
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


def _workflow(pool: ConnectionPool, checkpointer: AsyncPostgresSaver) -> Any:
    async def appointment_handler(state: ConversationState) -> Mapping[str, Any]:
        def process_action() -> Mapping[str, Any]:
            conversation_id = UUID(state["conversation_id"])
            owner_reference = state["patient_id"]
            assert isinstance(owner_reference, str)
            request_id = UUID(str(state["request_id"]))
            with pool.connection() as connection:
                service = AppointmentService.from_connection(connection)
                if state.get("intent") == "book_appointment":
                    booking = service.book_appointment(
                        UUID(str(state["requested_slot"])),
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
                        "response": "Waiting for staff approval." if pending else "Booked.",
                    }

                approval = service.request_cancellation(
                    UUID(str(state["requested_booking_id"])),
                    conversation_id,
                    owner_reference,
                    request_id=request_id,
                )
                return {
                    "booking_id": str(approval.booking_id),
                    "approval_id": str(approval.id),
                    "approval_status": approval.status,
                    "response": "Waiting for cancellation approval.",
                }

        return await asyncio.to_thread(process_action)

    return build_workflow(
        knowledge_service=ClinicKnowledgeService(_NeverCalledRetriever()),
        appointment_handler=appointment_handler,
        checkpointer=checkpointer,
    )


async def test_same_day_booking_resumes_to_final_booking_status(database_url: str) -> None:
    clinic_timezone = ZoneInfo("Asia/Muscat")
    now = datetime.now(clinic_timezone)
    starts_at = now + timedelta(minutes=15)
    if starts_at.date() != now.date():
        pytest.skip("A same-day future slot is not available at this time.")

    doctor_id = uuid4()
    conversation_id = uuid4()
    slot_id: UUID | None = None
    request_id = uuid4()
    owner_reference = f"opaque-test-owner-{uuid4()}"
    thread_id = str(conversation_id)
    inserted = False

    with ConnectionPool(conninfo=database_url, min_size=1, max_size=5) as pool:
        with pool.connection() as connection:
            connection.execute(
                "INSERT INTO doctors (id, full_name, specialty) VALUES (%s, %s, %s)",
                (doctor_id, "Fictional Approval Workflow Doctor", "General"),
            )
            connection.execute(
                "INSERT INTO conversations (id, thread_id, owner_reference) VALUES (%s, %s, %s)",
                (conversation_id, thread_id, owner_reference),
            )
            slot = AppointmentRepository(connection).create_slot(
                doctor_id,
                starts_at,
                starts_at + timedelta(minutes=30),
            )
            slot_id = slot.id
            inserted = True

        try:
            async with AsyncPostgresSaver.from_conn_string(database_url) as checkpointer:
                workflow = _workflow(pool, checkpointer)
                conversations = ConversationService(
                    _PooledConversationStore(pool),
                    workflow,
                )
                assert slot_id is not None
                pending_state = await conversations.process_appointment_action(
                    conversation_id,
                    owner_reference=owner_reference,
                    request_id=request_id,
                    action="book",
                    slot_id=slot_id,
                )

                assert pending_state["booking_status"] == "pending_approval"
                assert pending_state["approval_status"] == "pending"
                assert "__interrupt__" in pending_state
                booking_id = UUID(str(pending_state["booking_id"]))
                approval_id = UUID(str(pending_state["approval_id"]))
                pending_approval = await asyncio.to_thread(lambda: _approval(pool, approval_id))
                assert pending_approval is not None
                assert pending_approval.status == "pending"

            # Simulate an API restart: use a new saver and new services against
            # the persisted thread before recording the final database status.
            async with AsyncPostgresSaver.from_conn_string(database_url) as checkpointer:
                workflow = _workflow(pool, checkpointer)
                conversations = ConversationService(
                    _PooledConversationStore(pool),
                    workflow,
                )
                approvals = ApprovalService(_PooledApprovalStore(pool), conversations)
                decided, final_state = await approvals.decide(
                    approval_id,
                    "approved",
                    decided_by="fictional-test-staff",
                    decision_note="Approved in an isolated test database.",
                )

                assert decided.status == "approved"
                assert final_state["approval_status"] == "approved"
                assert final_state["booking_status"] == "confirmed"
                assert "__interrupt__" not in final_state
                booking = await asyncio.to_thread(lambda: _booking(pool, booking_id))
                assert booking == "confirmed"
                saved_approval = await asyncio.to_thread(lambda: _approval(pool, approval_id))
                assert saved_approval is not None
                assert saved_approval.status == "approved"

                cancel_slot_starts = datetime.now(clinic_timezone) + timedelta(days=2)
                with pool.connection() as connection:
                    future_slot = AppointmentRepository(connection).create_slot(
                        doctor_id,
                        cancel_slot_starts,
                        cancel_slot_starts + timedelta(minutes=30),
                    )
                    future_booking = AppointmentService.from_connection(
                        connection
                    ).book_appointment(
                        future_slot.id,
                        owner_reference,
                        conversation_id,
                        approval_workflow=True,
                        request_id=uuid4(),
                    )

                cancel_request_id = uuid4()
                cancellation_state = await conversations.process_appointment_action(
                    conversation_id,
                    owner_reference=owner_reference,
                    request_id=cancel_request_id,
                    action="cancel",
                    booking_id=future_booking.id,
                )
                assert cancellation_state["approval_status"] == "pending"
                assert "__interrupt__" in cancellation_state
                cancel_approval_id = UUID(str(cancellation_state["approval_id"]))
                cancel_decision, cancelled_state = await approvals.decide(
                    cancel_approval_id,
                    "approved",
                    decided_by="fictional-test-staff",
                    decision_note="Cancellation approved in an isolated test database.",
                )
                assert cancel_decision.status == "approved"
                assert cancelled_state["approval_status"] == "approved"
                assert cancelled_state["booking_status"] == "cancelled"
                assert (
                    await asyncio.to_thread(lambda: _booking(pool, future_booking.id))
                    == "cancelled"
                )
        finally:
            async with AsyncPostgresSaver.from_conn_string(database_url) as checkpointer:
                await checkpointer.adelete_thread(thread_id)
            if inserted:
                with pool.connection() as connection:
                    connection.execute(
                        "DELETE FROM approvals WHERE conversation_id = %s",
                        (conversation_id,),
                    )
                    connection.execute(
                        "DELETE FROM bookings WHERE conversation_id = %s",
                        (conversation_id,),
                    )
                    connection.execute(
                        "DELETE FROM appointment_slots WHERE doctor_id = %s", (doctor_id,)
                    )
                    connection.execute("DELETE FROM doctors WHERE id = %s", (doctor_id,))
                    connection.execute(
                        "DELETE FROM conversations WHERE id = %s", (conversation_id,)
                    )


def _approval(pool: ConnectionPool, approval_id: UUID) -> Approval | None:
    with pool.connection() as connection:
        return ApprovalRepository(connection).get(approval_id)


def _booking(pool: ConnectionPool, booking_id: UUID) -> str | None:
    with pool.connection() as connection:
        booking = connection.execute(
            "SELECT status FROM bookings WHERE id = %s",
            (booking_id,),
        ).fetchone()
        return booking[0] if booking is not None else None
