"""Fixtures for PostgreSQL-backed database tests."""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

import psycopg  # type: ignore[import-not-found]
import pytest  # type: ignore[import-not-found]
from psycopg import Connection  # type: ignore[import-not-found]

from app.db.initialize import initialize_database


@pytest.fixture(scope="session")
def database_url() -> str:
    """Use only an explicitly marked test database; initialize its schema once."""
    app_environment = (os.getenv("APP_ENV") or "").strip().lower()
    if app_environment != "test":
        pytest.skip("Set APP_ENV=test to enable PostgreSQL integration tests.")

    dsn = (os.getenv("TEST_DATABASE_URL") or os.getenv("DATABASE_URL") or "").strip()
    if not dsn:
        pytest.skip("Set TEST_DATABASE_URL (or DATABASE_URL in test mode) to run database tests.")

    initialize_database(dsn)
    return dsn


@pytest.fixture
def database_connection(database_url: str) -> Iterator[Connection[Any]]:
    """Roll back each test's rows so tests remain isolated and repeatable."""
    connection = psycopg.connect(database_url)
    try:
        yield connection
    finally:
        connection.rollback()
        connection.close()
