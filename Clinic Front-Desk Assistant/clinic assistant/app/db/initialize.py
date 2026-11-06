"""Initialize application tables, LangGraph checkpoints, and dev fixtures."""

from __future__ import annotations

import os
from importlib import import_module
from pathlib import Path
from urllib.parse import urlsplit

try:
    load_dotenv = import_module("dotenv").load_dotenv
except ImportError:

    def load_dotenv(*args: object, **kwargs: object) -> bool:
        """Allow initialization to use environment variables without python-dotenv."""
        return False


_DB_DIRECTORY = Path(__file__).resolve().parent
_PROJECT_DIRECTORY = _DB_DIRECTORY.parent.parent
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
    """Apply application schema, checkpoint migrations, and dev-only seed data.

    ``schema.sql`` must contain idempotent DDL. LangGraph owns its checkpoint
    schema and migrations. Demo seed data is guarded here and in ``seed.sql``.
    """
    load_dotenv(_PROJECT_DIRECTORY / ".env", override=False)
    dsn = (database_url or os.getenv("DATABASE_URL") or "").strip()
    if not dsn:
        raise RuntimeError("DATABASE_URL must be set before initializing PostgreSQL.")

    app_environment = (os.getenv("APP_ENV") or "production").strip().lower()
    schema_sql = _read_sql(_SCHEMA_PATH, required=True)
    seed_sql = _read_sql(_SEED_PATH)

    try:
        psycopg = import_module("psycopg")
    except ImportError as exc:
        raise RuntimeError(
            "psycopg is required to initialize the database; install the project dependencies."
        ) from exc

    try:
        postgres_saver = import_module("langgraph.checkpoint.postgres").PostgresSaver
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "langgraph-checkpoint-postgres is required to initialize checkpoint storage."
        ) from exc

    try:
        with psycopg.connect(dsn) as connection:
            with connection.cursor() as cursor:
                # prepare=False allows the SQL files to contain multiple statements.
                cursor.execute(schema_sql, prepare=False)
                if seed_sql and app_environment == "development":
                    cursor.execute(
                        "SELECT set_config('app.environment', %s, true)",
                        (app_environment,),
                    )
                    cursor.execute(seed_sql, prepare=False)

        # Use LangGraph's supported migrations instead of duplicating its
        # internal checkpoint table definitions in schema.sql.
        with postgres_saver.from_conn_string(dsn) as checkpointer:
            checkpointer.setup()
    except psycopg.Error as exc:
        message = "PostgreSQL initialization failed; verify DATABASE_URL and the SQL scripts."
        try:
            database_host = urlsplit(dsn).hostname
        except ValueError:
            database_host = None
        if database_host == "db" and "resolve host" in str(exc).lower():
            message = (
                "PostgreSQL initialization could not resolve host 'db'. That hostname is "
                "available only inside Docker Compose. Run `docker compose up --build` to "
                "initialize there, or use a host-reachable PostgreSQL URL (usually "
                "127.0.0.1) when running this command directly."
            )
        raise RuntimeError(message) from exc


def main() -> None:
    """Run database initialization as a command-line entry point."""
    initialize_database()
    print("Database schema and LangGraph checkpoint storage initialized successfully.")


if __name__ == "__main__":
    main()
