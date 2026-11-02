"""Unit tests for PostgreSQL initialization and SQL script loading."""

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

    def __init__(self, *, fail_connect: bool = False) -> None:
        super().__init__("psycopg")
        self.connection = _FakeConnection()
        self.fail_connect = fail_connect

    def connect(self, dsn: str) -> _FakeConnection:
        assert dsn == "postgresql://test/clinic_test"
        if self.fail_connect:
            raise self.Error("connection failed")
        return self.connection


def _write_sql_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    schema: str = "CREATE TABLE example (id INTEGER);",
    seed: str = "INSERT INTO example VALUES (1);",
) -> tuple[Path, Path]:
    schema_path = tmp_path / "schema.sql"
    seed_path = tmp_path / "seed.sql"
    schema_path.write_text(schema, encoding="utf-8")
    seed_path.write_text(seed, encoding="utf-8")
    monkeypatch.setattr(db_initialize, "_SCHEMA_PATH", schema_path)
    monkeypatch.setattr(db_initialize, "_SEED_PATH", seed_path)
    return schema_path, seed_path


def test_initializer_executes_schema_and_nonempty_seed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_sql_files(tmp_path, monkeypatch)
    fake_psycopg = _FakePsycopg()
    monkeypatch.setattr(db_initialize, "import_module", lambda _: fake_psycopg)

    db_initialize.initialize_database("postgresql://test/clinic_test")

    assert fake_psycopg.connection.cursor_instance.statements == [
        "CREATE TABLE example (id INTEGER);",
        "INSERT INTO example VALUES (1);",
    ]


def test_initializer_skips_empty_seed_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_sql_files(tmp_path, monkeypatch, seed="  \n")
    fake_psycopg = _FakePsycopg()
    monkeypatch.setattr(db_initialize, "import_module", lambda _: fake_psycopg)

    db_initialize.initialize_database("postgresql://test/clinic_test")

    assert fake_psycopg.connection.cursor_instance.statements == [
        "CREATE TABLE example (id INTEGER);"
    ]


def test_initializer_requires_nonempty_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_sql_files(tmp_path, monkeypatch, schema="  \n")

    with pytest.raises(RuntimeError, match="schema.sql is empty"):
        db_initialize.initialize_database("postgresql://test/clinic_test")


def test_initializer_requires_database_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)

    with pytest.raises(RuntimeError, match="DATABASE_URL must be set"):
        db_initialize.initialize_database()


def test_initializer_wraps_postgres_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_sql_files(tmp_path, monkeypatch)
    fake_psycopg = _FakePsycopg(fail_connect=True)
    monkeypatch.setattr(db_initialize, "import_module", lambda _: fake_psycopg)

    with pytest.raises(RuntimeError, match="PostgreSQL initialization failed"):
        db_initialize.initialize_database("postgresql://test/clinic_test")
