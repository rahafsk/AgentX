"""Initialize PostgreSQL tables and reference data for the application."""

from __future__ import annotations

import os
from importlib import import_module
from pathlib import Path

_DB_DIRECTORY = Path(__file__).resolve().parent
_SCHEMA_PATH = _DB_DIRECTORY / "schema.sql"
_SEED_PATH = _DB_DIRECTORY / "seed.sql"


def _read_sql(path: Path, *, required: bool = False) -> str:
    """Read an initialization script, optionally rejecting an empty file."""
    try:
        statement = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        if not required:
            return ""
        raise RuntimeError(f"Required database script not found: {path.name}") from exc
    except OSError as exc:
        raise RuntimeError(f"Could not read database script: {path.name}") from exc

    if required and not statement:
        raise RuntimeError(
            f"Database script {path.name} is empty; add the required SQL before initializing."
        )

    return statement


def initialize_database(database_url: str | None = None) -> None:
    """Apply the schema and seed scripts in one PostgreSQL transaction.

    ``schema.sql`` must contain idempotent DDL. ``seed.sql`` may be empty; if
    it contains data, make its statements safe to run more than once.
    """
    dsn = (database_url or os.getenv("DATABASE_URL") or "").strip()
    if not dsn:
        raise RuntimeError("DATABASE_URL must be set before initializing PostgreSQL.")

    schema_sql = _read_sql(_SCHEMA_PATH, required=True)
    seed_sql = _read_sql(_SEED_PATH)

    try:
        psycopg = import_module("psycopg")
    except ImportError as exc:
        raise RuntimeError(
            "psycopg is required to initialize the database; install the project dependencies."
        ) from exc

    try:
        with psycopg.connect(dsn) as connection:
            with connection.cursor() as cursor:
                # prepare=False allows the SQL files to contain multiple statements.
                cursor.execute(schema_sql, prepare=False)
                if seed_sql:
                    cursor.execute(seed_sql, prepare=False)
    except psycopg.Error as exc:
        raise RuntimeError(
            "PostgreSQL initialization failed; verify DATABASE_URL and the SQL scripts."
        ) from exc


def main() -> None:
    """Run database initialization as a command-line entry point."""
    initialize_database()
    print("Database schema and seed data initialized successfully.")


if __name__ == "__main__":
    main()
