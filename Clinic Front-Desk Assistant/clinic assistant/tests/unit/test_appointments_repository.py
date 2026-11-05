"""Unit tests for AppointmentRepository."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import psycopg
import pytest

from app.repositories.appointments import MAX_SEARCH_LIMIT, AppointmentRepository
from app.repositories.errors import (
    AppointmentSlotConflict,
    DoctorNotFound,
    InvalidAppointmentSlot,
)
from app.repositories.models import AppointmentSlot, AvailableAppointmentSlot, SlotStatus


class _FakeTransaction:
    """A transaction context that never suppresses exceptions."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: object) -> bool:
        return False


class _FakeCursor:
    """A cursor that records the executed query and returns configured rows."""

    def __init__(self) -> None:
        self.statement = ""
        self.parameters: object | None = None
        self.rows: list[dict[str, object]] = []
        self.row: dict[str, object] | None = None
        self.error: Exception | None = None

    def __enter__(self) -> _FakeCursor:
        return self

    def __exit__(self, *args: object) -> bool:
        return False

    def execute(self, statement: str, parameters: object | None = None) -> None:
        self.statement = statement
        self.parameters = parameters
        if self.error is not None:
            raise self.error

    def fetchall(self) -> list[dict[str, object]]:
        return self.rows

    def fetchone(self) -> dict[str, object] | None:
        return self.row


class _FakeConnection:
    """A connection that exposes a fake cursor and transaction."""

    def __init__(self) -> None:
        self.cursor_instance = _FakeCursor()
        self.transaction_count = 0

    def cursor(self, *, row_factory: object | None = None) -> _FakeCursor:
        return self.cursor_instance

    def transaction(self) -> _FakeTransaction:
        self.transaction_count += 1
        return _FakeTransaction()


@pytest.fixture
def connection() -> _FakeConnection:
    return _FakeConnection()


@pytest.fixture
def cursor(connection: _FakeConnection) -> _FakeCursor:
    return connection.cursor_instance


@pytest.fixture
def repository(connection: _FakeConnection) -> AppointmentRepository:
    return AppointmentRepository(connection)  # type: ignore[arg-type]


def _slot_row(
    slot_id: UUID,
    doctor_id: UUID,
    starts_at: datetime,
    *,
    ends_at: datetime | None = None,
    status: str = "open",
) -> dict[str, object]:
    return {
        "id": slot_id,
        "doctor_id": doctor_id,
        "starts_at": starts_at,
        "ends_at": ends_at or starts_at + timedelta(minutes=30),
        "status": status,
        "created_at": starts_at - timedelta(days=1),
    }


def _available_row(
    slot_id: UUID,
    doctor_id: UUID,
    starts_at: datetime,
) -> dict[str, object]:
    return {
        "id": slot_id,
        "doctor_id": doctor_id,
        "doctor_name": "Dr. Sara",
        "doctor_specialty": "Pediatrics",
        "starts_at": starts_at,
        "ends_at": starts_at + timedelta(minutes=30),
    }


def test_max_search_limit_is_100() -> None:
    assert MAX_SEARCH_LIMIT == 100


def test_search_open_slots_returns_available_slots(
    repository: AppointmentRepository,
    cursor: _FakeCursor,
) -> None:
    starts_at = datetime.now(UTC)
    ends_at = starts_at + timedelta(hours=2)
    first_row = _available_row(uuid4(), uuid4(), starts_at)
    second_row = _available_row(
        uuid4(),
        uuid4(),
        starts_at + timedelta(hours=1),
    )
    cursor.rows = [first_row, second_row]

    results = repository.search_open_slots(starts_at, ends_at)

    assert results == [
        AvailableAppointmentSlot(**first_row),
        AvailableAppointmentSlot(**second_row),
    ]
    assert "FROM appointment_slots AS slot" in cursor.statement
    assert "JOIN doctors AS doctor" in cursor.statement
    assert "slot.status = 'open'" in cursor.statement
    assert "doctor.is_active = TRUE" in cursor.statement
    assert "slot.starts_at >= %s" in cursor.statement
    assert "slot.starts_at < %s" in cursor.statement
    assert "NOT EXISTS" in cursor.statement
    assert "booking.status IN ('pending_approval', 'confirmed')" in cursor.statement
    assert "slot.doctor_id = %s" not in cursor.statement
    assert "ORDER BY slot.starts_at, doctor.full_name LIMIT %s" in cursor.statement
    assert cursor.parameters == [starts_at, ends_at, 25]


def test_search_open_slots_filters_by_doctor_and_accepts_maximum_limit(
    repository: AppointmentRepository,
    cursor: _FakeCursor,
) -> None:
    starts_at = datetime.now(UTC)
    ends_at = starts_at + timedelta(hours=1)
    doctor_id = uuid4()

    repository.search_open_slots(
        starts_at,
        ends_at,
        doctor_id=doctor_id,
        limit=MAX_SEARCH_LIMIT,
    )

    assert "slot.doctor_id = %s" in cursor.statement
    assert cursor.parameters == [
        starts_at,
        ends_at,
        doctor_id,
        MAX_SEARCH_LIMIT,
    ]


@pytest.mark.parametrize("limit", [0, -1, MAX_SEARCH_LIMIT + 1])
def test_search_open_slots_rejects_invalid_limit(
    repository: AppointmentRepository,
    cursor: _FakeCursor,
    limit: int,
) -> None:
    starts_at = datetime.now(UTC)
    ends_at = starts_at + timedelta(hours=1)

    with pytest.raises(ValueError, match="limit"):
        repository.search_open_slots(starts_at, ends_at, limit=limit)

    assert cursor.statement == ""


@pytest.mark.parametrize("field_name", ["starts_at", "ends_at"])
def test_search_open_slots_rejects_naive_datetimes(
    repository: AppointmentRepository,
    cursor: _FakeCursor,
    field_name: str,
) -> None:
    starts_at = datetime.now(UTC)
    ends_at = starts_at + timedelta(hours=1)
    if field_name == "starts_at":
        starts_at = starts_at.replace(tzinfo=None)
    else:
        ends_at = ends_at.replace(tzinfo=None)

    with pytest.raises(ValueError, match=field_name):
        repository.search_open_slots(starts_at, ends_at)

    assert cursor.statement == ""


@pytest.mark.parametrize("offset", [timedelta(0), timedelta(minutes=-1)])
def test_search_open_slots_rejects_non_increasing_range(
    repository: AppointmentRepository,
    cursor: _FakeCursor,
    offset: timedelta,
) -> None:
    starts_at = datetime.now(UTC)

    with pytest.raises(ValueError, match="ends_at"):
        repository.search_open_slots(starts_at, starts_at + offset)

    assert cursor.statement == ""


def test_get_slot_returns_slot(
    repository: AppointmentRepository,
    cursor: _FakeCursor,
) -> None:
    slot_id = uuid4()
    doctor_id = uuid4()
    starts_at = datetime.now(UTC)
    cursor.row = _slot_row(slot_id, doctor_id, starts_at)

    result = repository.get_slot(slot_id)

    assert result == AppointmentSlot(
        id=slot_id,
        doctor_id=doctor_id,
        starts_at=starts_at,
        ends_at=starts_at + timedelta(minutes=30),
        status="open",
        created_at=starts_at - timedelta(days=1),
    )
    assert "WHERE id = %s" in cursor.statement
    assert cursor.parameters == (slot_id,)


def test_get_slot_returns_none_when_missing(
    repository: AppointmentRepository,
    cursor: _FakeCursor,
) -> None:
    cursor.row = None

    assert repository.get_slot(uuid4()) is None


@pytest.mark.parametrize("status", ["open", "blocked"])
def test_create_slot_returns_created_slot(
    repository: AppointmentRepository,
    connection: _FakeConnection,
    cursor: _FakeCursor,
    status: SlotStatus,
) -> None:
    slot_id = uuid4()
    doctor_id = uuid4()
    starts_at = datetime.now(UTC)
    ends_at = starts_at + timedelta(minutes=30)
    cursor.row = _slot_row(
        slot_id,
        doctor_id,
        starts_at,
        ends_at=ends_at,
        status=status,
    )

    result = repository.create_slot(
        doctor_id,
        starts_at,
        ends_at,
        status=status,
    )

    assert result == AppointmentSlot(
        id=slot_id,
        doctor_id=doctor_id,
        starts_at=starts_at,
        ends_at=ends_at,
        status=status,
        created_at=starts_at - timedelta(days=1),
    )
    assert connection.transaction_count == 1
    assert "INSERT INTO appointment_slots" in cursor.statement
    assert "RETURNING" in cursor.statement
    assert cursor.parameters == (doctor_id, starts_at, ends_at, status)


@pytest.mark.parametrize("field_name", ["starts_at", "ends_at"])
def test_create_slot_rejects_naive_datetimes(
    repository: AppointmentRepository,
    connection: _FakeConnection,
    cursor: _FakeCursor,
    field_name: str,
) -> None:
    starts_at = datetime.now(UTC)
    ends_at = starts_at + timedelta(minutes=30)
    if field_name == "starts_at":
        starts_at = starts_at.replace(tzinfo=None)
    else:
        ends_at = ends_at.replace(tzinfo=None)

    with pytest.raises(ValueError, match=field_name):
        repository.create_slot(uuid4(), starts_at, ends_at)

    assert connection.transaction_count == 0
    assert cursor.statement == ""


@pytest.mark.parametrize("offset", [timedelta(0), timedelta(minutes=-1)])
def test_create_slot_rejects_non_increasing_range(
    repository: AppointmentRepository,
    connection: _FakeConnection,
    cursor: _FakeCursor,
    offset: timedelta,
) -> None:
    starts_at = datetime.now(UTC)

    with pytest.raises(ValueError, match="ends_at"):
        repository.create_slot(uuid4(), starts_at, starts_at + offset)

    assert connection.transaction_count == 0
    assert cursor.statement == ""


def test_create_slot_rejects_invalid_status(
    repository: AppointmentRepository,
    connection: _FakeConnection,
    cursor: _FakeCursor,
) -> None:
    starts_at = datetime.now(UTC)

    with pytest.raises(ValueError, match="status"):
        repository.create_slot(
            uuid4(),
            starts_at,
            starts_at + timedelta(minutes=30),
            status="cancelled",  # type: ignore[arg-type]
        )

    assert connection.transaction_count == 0
    assert cursor.statement == ""


@pytest.mark.parametrize(
    ("database_error", "expected_error"),
    [
        (psycopg.errors.ExclusionViolation("overlapping slot"), AppointmentSlotConflict),
        (psycopg.errors.ForeignKeyViolation("missing doctor"), DoctorNotFound),
        (psycopg.errors.CheckViolation("invalid slot"), InvalidAppointmentSlot),
    ],
)
def test_create_slot_maps_database_errors(
    repository: AppointmentRepository,
    cursor: _FakeCursor,
    database_error: Exception,
    expected_error: type[Exception],
) -> None:
    cursor.error = database_error
    starts_at = datetime.now(UTC)

    with pytest.raises(expected_error) as exc_info:
        repository.create_slot(
            uuid4(),
            starts_at,
            starts_at + timedelta(minutes=30),
        )

    assert exc_info.value.__cause__ is database_error


def test_create_slot_raises_when_database_returns_no_row(
    repository: AppointmentRepository,
    connection: _FakeConnection,
    cursor: _FakeCursor,
) -> None:
    cursor.row = None
    starts_at = datetime.now(UTC)

    with pytest.raises(RuntimeError, match="did not return"):
        repository.create_slot(
            uuid4(),
            starts_at,
            starts_at + timedelta(minutes=30),
        )

    assert connection.transaction_count == 1
