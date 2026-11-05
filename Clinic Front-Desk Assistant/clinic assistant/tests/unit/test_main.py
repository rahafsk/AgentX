"""Unit coverage for application composition and pooled repository adapters."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any, cast
from uuid import UUID, uuid4

import pytest

from app import main as main_module
from app.repositories.models import (
    Approval,
    AvailableAppointmentSlot,
    Booking,
    Conversation,
)

_CONVERSATION_ID = UUID("11111111-1111-4111-8111-111111111111")
_BOOKING_ID = UUID("22222222-2222-4222-8222-222222222222")
_SLOT_ID = UUID("33333333-3333-4333-8333-333333333333")
_OWNER = "opaque-test-owner"
_NOW = datetime(2026, 1, 1, tzinfo=UTC)


class _FakePool:
    def __init__(self, connection: Any) -> None:
        self._connection = connection

    @contextmanager
    def connection(self) -> Iterator[Any]:
        yield self._connection


class _SchemaCursor:
    def __init__(self, row: tuple[str | None, ...]) -> None:
        self._row = row

    def fetchone(self) -> tuple[str | None, ...]:
        return self._row


class _SchemaConnection:
    def __init__(self, row: tuple[str | None, ...]) -> None:
        self.row = row
        self.statement = ""

    def execute(self, statement: str) -> _SchemaCursor:
        self.statement = statement
        return _SchemaCursor(self.row)


class _ConversationRepository:
    def __init__(self, connection: Any) -> None:
        self.calls: list[tuple[Any, ...]] = []

    def create(self, owner_reference: str, request_id: UUID) -> Conversation:
        self.calls.append(("create", owner_reference, request_id))
        return _conversation()

    def create_idempotent(
        self,
        owner_reference: str,
        idempotency_key: str,
        request_hash: str,
    ) -> Conversation:
        self.calls.append(("create_idempotent", owner_reference, idempotency_key, request_hash))
        return _conversation()

    def claim_request(
        self,
        conversation_id: UUID,
        request_id: UUID,
        request_hash: str,
    ) -> Any:
        self.calls.append(("claim", conversation_id, request_id, request_hash))
        return SimpleNamespace(created=True, result=None)

    def complete_request(
        self,
        conversation_id: UUID,
        request_id: UUID,
        result: dict[str, Any],
    ) -> None:
        self.calls.append(("complete", conversation_id, request_id, result))

    def get(self, conversation_id: UUID) -> Conversation | None:
        self.calls.append(("get", conversation_id))
        return _conversation()

    def get_for_owner(self, conversation_id: UUID, owner_reference: str) -> Conversation | None:
        self.calls.append(("get_for_owner", conversation_id, owner_reference))
        return _conversation()

    def touch_for_owner(self, conversation_id: UUID, owner_reference: str) -> Conversation | None:
        self.calls.append(("touch_for_owner", conversation_id, owner_reference))
        return _conversation()


class _AppointmentRepository:
    def __init__(self, connection: Any) -> None:
        pass

    def search_open_slots(self, *args: Any, **kwargs: Any) -> list[AvailableAppointmentSlot]:
        return []

    def get_slot(self, slot_id: UUID) -> Any:
        return SimpleNamespace(id=slot_id)


class _BookingRepository:
    def __init__(self, connection: Any) -> None:
        pass

    def get(self, booking_id: UUID) -> Booking | None:
        return None

    def list_for_conversation(self, conversation_id: UUID) -> list[Booking]:
        return []


class _ApprovalRepository:
    def __init__(self, connection: Any) -> None:
        pass

    def get(self, approval_id: UUID) -> Approval | None:
        return None

    def list_pending(self, *, limit: int = 50) -> list[Approval]:
        return []

    def decide(self, *args: Any, **kwargs: Any) -> Approval:
        return _approval()


class _DomainAppointmentService:
    def __init__(self, *, booking_error: Exception | None = None) -> None:
        self.booking_error = booking_error
        self.booking_calls: list[tuple[UUID, str, UUID, UUID | None]] = []
        self.cancellation_calls: list[tuple[UUID, UUID, str, UUID | None]] = []

    @classmethod
    def from_connection(cls, connection: Any) -> _DomainAppointmentService:
        return cls()

    def book_appointment(
        self,
        slot_id: UUID,
        owner_reference: str,
        conversation_id: UUID,
        *,
        approval_workflow: bool,
        request_id: UUID | None,
    ) -> Booking:
        self.booking_calls.append((slot_id, owner_reference, conversation_id, request_id))
        if self.booking_error is not None:
            raise self.booking_error
        return cast(Booking, SimpleNamespace(id=_BOOKING_ID, status="pending_approval"))

    def request_cancellation(
        self,
        booking_id: UUID,
        conversation_id: UUID,
        owner_reference: str,
        *,
        request_id: UUID | None,
    ) -> Approval:
        self.cancellation_calls.append((booking_id, conversation_id, owner_reference, request_id))
        return cast(
            Approval,
            SimpleNamespace(
                id=request_id or uuid4(),
                booking_id=booking_id,
                status="pending",
            ),
        )


def _conversation() -> Conversation:
    return Conversation(
        id=_CONVERSATION_ID,
        thread_id=str(_CONVERSATION_ID),
        created_at=_NOW,
        updated_at=_NOW,
        owner_reference=_OWNER,
    )


def _approval() -> Approval:
    return Approval(
        id=_BOOKING_ID,
        booking_id=_BOOKING_ID,
        conversation_id=_CONVERSATION_ID,
        action="same_day_booking",
        status="pending",
        requested_at=_NOW,
        decided_at=None,
        decided_by=None,
        decision_note=None,
    )


def test_schema_readiness_verification_requires_core_tables() -> None:
    complete_pool = _FakePool(
        _SchemaConnection(
            ("doctors", "conversations", "idempotency", "slots", "bookings", "approvals", "results")
        )
    )
    missing_pool = _FakePool(
        _SchemaConnection(
            ("doctors", "conversations", "idempotency", "slots", None, "approvals", "results")
        )
    )

    main_module._verify_application_schema(cast(Any, complete_pool))
    with pytest.raises(RuntimeError, match="schema is not initialized"):
        main_module._verify_application_schema(cast(Any, missing_pool))


def test_pooled_conversation_store_uses_repository_per_operation(monkeypatch) -> None:
    monkeypatch.setattr(main_module, "ConversationRepository", _ConversationRepository)
    store = main_module._PooledConversationStore(cast(Any, _FakePool(object())))
    request_id = uuid4()

    assert store.create(_OWNER, request_id) == _conversation()
    assert store.create_idempotent(_OWNER, str(request_id), "hash") == _conversation()
    assert store.claim_request(_CONVERSATION_ID, request_id, "hash").created is True
    store.complete_request(_CONVERSATION_ID, request_id, {"response": "done"})
    assert store.get(_CONVERSATION_ID) == _conversation()
    assert store.get_for_owner(_CONVERSATION_ID, _OWNER) == _conversation()
    assert store.touch_for_owner(_CONVERSATION_ID, _OWNER) == _conversation()


def test_pooled_appointment_booking_and_approval_adapters(monkeypatch) -> None:
    monkeypatch.setattr(main_module, "AppointmentRepository", _AppointmentRepository)
    monkeypatch.setattr(main_module, "BookingRepository", _BookingRepository)
    monkeypatch.setattr(main_module, "ApprovalRepository", _ApprovalRepository)
    monkeypatch.setattr(main_module, "AppointmentService", _DomainAppointmentService)
    pool = cast(Any, _FakePool(object()))

    appointments = main_module._PooledAppointmentService(pool)
    start = datetime(2030, 1, 1, tzinfo=UTC)
    assert appointments.search_available_slots(start, start + timedelta(days=1)) == []
    assert appointments.get_slot(_SLOT_ID).id == _SLOT_ID
    assert (
        appointments.book_appointment(
            _SLOT_ID,
            _OWNER,
            _CONVERSATION_ID,
            approval_workflow=True,
            request_id=_BOOKING_ID,
        ).status
        == "pending_approval"
    )
    assert (
        appointments.request_cancellation(
            _BOOKING_ID,
            _CONVERSATION_ID,
            _OWNER,
            request_id=_BOOKING_ID,
        ).status
        == "pending"
    )
    assert appointments.list_for_conversation(_CONVERSATION_ID) == []

    bookings = main_module._PooledBookingStore(pool)
    assert bookings.get(_BOOKING_ID) is None
    assert bookings.list_for_conversation(_CONVERSATION_ID) == []

    approvals = main_module._PooledApprovalStore(pool)
    assert approvals.get(_BOOKING_ID) is None
    assert approvals.list_pending() == []
    assert (
        approvals.decide(
            _BOOKING_ID,
            "approved",
            decided_by="test-staff",
            decision_note="Reviewed.",
        )
        == _approval()
    )
