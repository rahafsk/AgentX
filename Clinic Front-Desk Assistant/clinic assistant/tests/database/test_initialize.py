"""Unit tests for database bootstrap script handling."""

from __future__ import annotations

from pathlib import Path
from types import ModuleType

import pytest

from app.db import initialize as db_initialize


class _FakeCursor:
    def __init__(self) -> None:
        self.statements: list[str] = []

    def __enter__(self) -> _FakeCursor:
        return self

    def __exit__(self, *args: object) -> bool:
        return False

    def execute(self, statement: str, *, prepare: bool) -> None:
        assert prepare is False
        self.statements.append(statement)


class _FakeConnection:
    def __init__(self) -> None:
        self.cursor_instance = _FakeCursor()

    def __enter__(self) -> _FakeConnection:
        return self

    def __exit__(self, *args: object) -> bool:
        return False

    def cursor(self) -> _FakeCursor:
        return self.cursor_instance


class _FakePsycopg(ModuleType):
    class Error(Exception):
        pass

    def __init__(self) -> None:
        super().__init__("psycopg")
        self.connection = _FakeConnection()

    def connect(self, dsn: str) -> _FakeConnection:
        assert dsn == "postgresql://test/clinic_test"
        return self.connection


def test_initialize_skips_missing_optional_seed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schema_path = tmp_path / "schema.sql"
    schema_path.write_text("CREATE TABLE example (id INTEGER);", encoding="utf-8")
    seed_path = tmp_path / "seed.sql"
    fake_psycopg = _FakePsycopg()

    monkeypatch.setattr(db_initialize, "_SCHEMA_PATH", schema_path)
    monkeypatch.setattr(db_initialize, "_SEED_PATH", seed_path)
    monkeypatch.setattr(db_initialize, "import_module", lambda _: fake_psycopg)

    db_initialize.initialize_database("postgresql://test/clinic_test")

    assert fake_psycopg.connection.cursor_instance.statements == [
        "CREATE TABLE example (id INTEGER);"
    ]


def test_initialize_rejects_missing_required_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing_schema = tmp_path / "schema.sql"
    monkeypatch.setattr(db_initialize, "_SCHEMA_PATH", missing_schema)

    with pytest.raises(RuntimeError, match="Required database script not found"):
        db_initialize.initialize_database("postgresql://test/clinic_test")
