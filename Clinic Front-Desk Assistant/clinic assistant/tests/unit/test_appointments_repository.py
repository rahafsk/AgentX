
"""Unit tests for AppointmentRepository."""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest

from app.repositories.appointments import AppointmentRepository


@pytest.fixture
def repository() -> AppointmentRepository:
    connection = MagicMock()
    return AppointmentRepository(connection)


def test_search_open_slots_rejects_zero_limit(
    repository: AppointmentRepository,
) -> None:
    starts_at = datetime.now(UTC)
    ends_at = starts_at + timedelta(hours=1)

    with pytest.raises(ValueError, match="limit"):
        repository.search_open_slots(starts_at, ends_at, limit=0)


def test_search_open_slots_rejects_negative_limit(
    repository: AppointmentRepository,
) -> None:
    starts_at = datetime.now(UTC)
    ends_at = starts_at + timedelta(hours=1)

    with pytest.raises(ValueError, match="limit"):
        repository.search_open_slots(starts_at, ends_at, limit=-1)


def test_search_open_slots_rejects_limit_above_maximum(
    repository: AppointmentRepository,
) -> None:
    starts_at = datetime.now(UTC)
    ends_at = starts_at + timedelta(hours=1)

    with pytest.raises(ValueError, match="limit"):
        repository.search_open_slots(starts_at, ends_at, limit=101)

    repository._connection.cursor.assert_not_called()
