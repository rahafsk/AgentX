"""PostgreSQL tests for idempotent conversation creation."""

from __future__ import annotations

import threading
from typing import Any
from uuid import uuid4

import psycopg
import pytest
from psycopg import Connection

from app.repositories.conversations import (
    ConversationRepository,
    IdempotencyKeyConflict,
)
from app.repositories.models import Conversation


def test_matching_retry_returns_original_conversation(
    database_connection: Connection[Any],
) -> None:
    repository = ConversationRepository(database_connection)
    owner = f"owner-{uuid4()}"
    key = f"key-{uuid4()}"

    first = repository.create_idempotent(owner, key, "hash-1")
    second = repository.create_idempotent(owner, key, "hash-1")

    assert second == first


def test_key_reuse_with_different_content_is_rejected(
    database_connection: Connection[Any],
) -> None:
    repository = ConversationRepository(database_connection)
    owner = f"owner-{uuid4()}"
    key = f"key-{uuid4()}"

    repository.create_idempotent(owner, key, "hash-1")

    with pytest.raises(IdempotencyKeyConflict):
        repository.create_idempotent(owner, key, "hash-2")


def test_same_key_is_isolated_by_owner(
    database_connection: Connection[Any],
) -> None:
    repository = ConversationRepository(database_connection)
    key = f"key-{uuid4()}"

    first = repository.create_idempotent("owner-a", key, "hash-1")
    second = repository.create_idempotent("owner-b", key, "hash-1")

    assert first.id != second.id


def test_concurrent_retries_create_one_conversation(database_url: str) -> None:
    owner = f"owner-{uuid4()}"
    key = f"key-{uuid4()}"
    barrier = threading.Barrier(2)
    results: list[Conversation] = []
    errors: list[BaseException] = []

    def create() -> None:
        try:
            with psycopg.connect(database_url) as connection:
                repository = ConversationRepository(connection)
                barrier.wait(timeout=5)
                results.append(repository.create_idempotent(owner, key, "hash-1"))
        except BaseException as exc:  # pragma: no cover - surfaced by assertions
            errors.append(exc)

    threads = [threading.Thread(target=create) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert errors == []
    assert len(results) == 2
    assert results[0] == results[1]

    with psycopg.connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT COUNT(*) FROM conversations WHERE owner_reference = %s",
                (owner,),
            )
            count = cursor.fetchone()
            assert count == (1,)
            cursor.execute(
                "DELETE FROM conversations WHERE owner_reference = %s",
                (owner,),
            )
