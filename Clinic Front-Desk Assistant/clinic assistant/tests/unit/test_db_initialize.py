"""Unit tests for PostgreSQL initialization and SQL script loading."""

from __future__ import annotations

from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from app.db import initialize as db_initialize


class _FakeCursor:
    def __init__(self) -> None:
        self.statements: list[str] = []
        self.parameters: list[object | None] = []

    def __enter__(self) -> _FakeCursor:
        return self

    def __exit__(self, *args: object) -> bool:
        return False

    def execute(
        self,
        statement: str,
        parameters: object | None = None,
        *,
        prepare: bool | None = None,
    ) -> None:
        if prepare is not None:
            assert prepare is False
        self.statements.append(statement)
        self.parameters.append(parameters)


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

    def __init__(
        self,
        *,
        fail_connect: bool = False,
        error_message: str = "connection failed",
        expected_dsn: str = "postgresql://test/clinic_test",
    ) -> None:
        super().__init__("psycopg")
        self.connection = _FakeConnection()
        self.fail_connect = fail_connect
        self.error_message = error_message
        self.expected_dsn = expected_dsn

    def connect(self, dsn: str) -> _FakeConnection:
        assert dsn == self.expected_dsn
        if self.fail_connect:
            raise self.Error(self.error_message)
        return self.connection


class _FakePostgresSaver:
    setup_calls = 0

    @classmethod
    def from_conn_string(cls, dsn: str) -> _FakePostgresSaver:
        assert dsn == "postgresql://test/clinic_test"
        return cls()

    def __enter__(self) -> _FakePostgresSaver:
        return self

    def __exit__(self, *args: object) -> bool:
        return False

    def setup(self) -> None:
        type(self).setup_calls += 1


def _fake_import_module(fake_psycopg: _FakePsycopg):
    def import_module(module_name: str) -> object:
        if module_name == "psycopg":
            return fake_psycopg
        if module_name == "langgraph.checkpoint.postgres":
            return SimpleNamespace(PostgresSaver=_FakePostgresSaver)
        raise ImportError(module_name)

    return import_module


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
    _FakePostgresSaver.setup_calls = 0
    monkeypatch.setattr(db_initialize, "import_module", _fake_import_module(fake_psycopg))
    monkeypatch.setenv("APP_ENV", "development")

    db_initialize.initialize_database("postgresql://test/clinic_test")

    assert fake_psycopg.connection.cursor_instance.statements == [
        "CREATE TABLE example (id INTEGER);",
        "SELECT set_config('app.environment', %s, true)",
        "INSERT INTO example VALUES (1);",
    ]
    assert fake_psycopg.connection.cursor_instance.parameters == [
        None,
        ("development",),
        None,
    ]
    assert _FakePostgresSaver.setup_calls == 1


def test_initializer_skips_empty_seed_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_sql_files(tmp_path, monkeypatch, seed="  \n")
    fake_psycopg = _FakePsycopg()
    _FakePostgresSaver.setup_calls = 0
    monkeypatch.setattr(db_initialize, "import_module", _fake_import_module(fake_psycopg))
    monkeypatch.setenv("APP_ENV", "test")

    db_initialize.initialize_database("postgresql://test/clinic_test")

    assert fake_psycopg.connection.cursor_instance.statements == [
        "CREATE TABLE example (id INTEGER);"
    ]
    assert _FakePostgresSaver.setup_calls == 1


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
    monkeypatch.setattr(db_initialize, "load_dotenv", lambda *args, **kwargs: False)

    with pytest.raises(RuntimeError, match="DATABASE_URL must be set"):
        db_initialize.initialize_database()


def test_initializer_wraps_postgres_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_sql_files(tmp_path, monkeypatch)
    fake_psycopg = _FakePsycopg(fail_connect=True)
    monkeypatch.setattr(db_initialize, "import_module", _fake_import_module(fake_psycopg))

    with pytest.raises(RuntimeError, match="PostgreSQL initialization failed"):
        db_initialize.initialize_database("postgresql://test/clinic_test")


def test_initializer_explains_compose_hostname_from_host(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_sql_files(tmp_path, monkeypatch)
    fake_psycopg = _FakePsycopg(
        fail_connect=True,
        error_message="failed to resolve host 'db'",
        expected_dsn="postgresql://test@db:5432/clinic_test",
    )
    monkeypatch.setattr(db_initialize, "import_module", _fake_import_module(fake_psycopg))

    with pytest.raises(RuntimeError, match="available only inside Docker Compose"):
        db_initialize.initialize_database("postgresql://test@db:5432/clinic_test")
