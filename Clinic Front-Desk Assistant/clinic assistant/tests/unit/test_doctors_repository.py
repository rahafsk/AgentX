from datetime import UTC, datetime
from unittest.mock import MagicMock
from uuid import uuid4

from app.repositories.doctors import DoctorRepository
from app.repositories.models import Doctor


def test_get_returns_doctor() -> None:
    doctor_id = uuid4()
    created_at = datetime.now(UTC)

    connection = MagicMock()
    cursor = connection.cursor.return_value.__enter__.return_value
    cursor.fetchone.return_value = {
        "id": doctor_id,
        "full_name": "Dr. Ahmed",
        "specialty": "Cardiology",
        "is_active": True,
        "created_at": created_at,
    }

    repository = DoctorRepository(connection)
    result = repository.get(doctor_id)

    assert result == Doctor(
        id=doctor_id,
        full_name="Dr. Ahmed",
        specialty="Cardiology",
        is_active=True,
        created_at=created_at,
    )

    cursor.execute.assert_called_once()
    query = cursor.execute.call_args.args[0]
    assert "FROM doctors" in query
    assert "WHERE id = %s" in query
    assert cursor.execute.call_args.args[1] == (doctor_id,)


def test_get_returns_none_when_missing() -> None:
    connection = MagicMock()
    cursor = connection.cursor.return_value.__enter__.return_value
    cursor.fetchone.return_value = None

    repository = DoctorRepository(connection)
    result = repository.get(uuid4())

    assert result is None


def test_list_active_returns_doctors() -> None:
    first_id = uuid4()
    second_id = uuid4()
    created_at = datetime.now(UTC)

    connection = MagicMock()
    cursor = connection.cursor.return_value.__enter__.return_value
    cursor.fetchall.return_value = [
        {
            "id": first_id,
            "full_name": "Dr. Ahmed",
            "specialty": "Cardiology",
            "is_active": True,
            "created_at": created_at,
        },
        {
            "id": second_id,
            "full_name": "Dr. Sara",
            "specialty": "Pediatrics",
            "is_active": True,
            "created_at": created_at,
        },
    ]

    repository = DoctorRepository(connection)
    result = repository.list_active()

    assert len(result) == 2
    assert all(doctor.is_active for doctor in result)
    assert result[0].full_name == "Dr. Ahmed"
    assert result[1].full_name == "Dr. Sara"

    query = cursor.execute.call_args.args[0]
    assert "WHERE is_active = TRUE" in query
    assert "ORDER BY full_name ASC, id ASC" in query
    assert "FROM doctors" in query
