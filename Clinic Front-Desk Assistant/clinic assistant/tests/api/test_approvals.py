"""API tests for staff approval decisions."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langgraph.types import Command

from app.api.approvals import router
from app.repositories.errors import (
    ApprovalAlreadyDecided,
    ApprovalNotFound,
    ApprovalStateConflict,
)
from app.repositories.models import Approval, Conversation

_APPROVAL_ID = UUID("11111111-1111-1111-1111-111111111111")
_BOOKING_ID = UUID("22222222-2222-2222-2222-222222222222")
_CONVERSATION_ID = UUID("33333333-3333-3333-3333-333333333333")
_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _approval(*, status: str = "pending") -> Approval:
    return Approval(
        id=_APPROVAL_ID,
        booking_id=_BOOKING_ID,
        conversation_id=_CONVERSATION_ID,
        action="cancel_booking",
        status=status,  # type: ignore[arg-type]
        requested_at=_NOW,
        decided_at=None,
        decided_by=None,
        decision_note=None,
    )


def _conversation() -> Conversation:
    return Conversation(
        id=_CONVERSATION_ID,
        thread_id="thread-123",
        created_at=_NOW,
        updated_at=_NOW,
        owner_reference="owner-1",
    )


class _ApprovalRepository:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[tuple[UUID, str, str, str]] = []

    def list_pending(self, *, limit: int = 50) -> list[Approval]:
        return [_approval()]

    def decide(
        self,
        approval_id: UUID,
        decision: str,
        *,
        decided_by: str,
        decision_note: str | None = None,
    ) -> Approval:
        self.calls.append((approval_id, decision, decided_by, decision_note or ""))
        if self.error is not None:
            raise self.error
        return _approval(status=decision)


class _ConversationRepository:
    def get(self, conversation_id: UUID) -> Conversation | None:
        return _conversation()


class _Workflow:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[tuple[Command[Any], Mapping[str, Any]]] = []

    async def ainvoke(
        self,
        input: Command[Any],
        config: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        self.calls.append((input, config))
        if self.error is not None:
            raise self.error
        return {"response": "The decision was applied."}


def _client(
    *,
    approvals: _ApprovalRepository | None = None,
    conversations: _ConversationRepository | None = None,
    workflow: _Workflow | None = None,
) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    if approvals is not None:
        app.state.approval_repository = approvals
    if conversations is not None:
        app.state.conversation_repository = conversations
    if workflow is not None:
        app.state.workflow = workflow
    return TestClient(app)


def test_list_pending_approvals() -> None:
    client = _client(
        approvals=_ApprovalRepository(),
        conversations=_ConversationRepository(),
        workflow=_Workflow(),
    )

    response = client.get("/approvals")

    assert response.status_code == 200
    assert response.json()[0]["id"] == str(_APPROVAL_ID)
    assert response.json()[0]["status"] == "pending"


@pytest.mark.parametrize("decision", ["approved", "rejected"])
def test_decision_resumes_matching_thread(decision: str) -> None:
    workflow = _Workflow()
    client = _client(
        approvals=_ApprovalRepository(),
        conversations=_ConversationRepository(),
        workflow=workflow,
    )

    response = client.post(
        f"/approvals/{_APPROVAL_ID}/decision",
        json={
            "decision": decision,
            "decided_by": "staff-1",
            "decision_note": "Reviewed",
        },
    )

    assert response.status_code == 200
    assert response.json()["response"] == "The decision was applied."
    command, config = workflow.calls[0]
    assert command.resume == {"decision": decision}
    assert config == {"configurable": {"thread_id": "thread-123"}}


@pytest.mark.parametrize(
    ("error", "status_code"),
    [
        (ApprovalNotFound(), 404),
        (ApprovalAlreadyDecided(), 409),
        (ApprovalStateConflict(), 409),
    ],
)
def test_repository_errors_are_safe(
    error: Exception,
    status_code: int,
) -> None:
    client = _client(
        approvals=_ApprovalRepository(error=error),
        conversations=_ConversationRepository(),
        workflow=_Workflow(),
    )

    response = client.post(
        f"/approvals/{_APPROVAL_ID}/decision",
        json={
            "decision": "approved",
            "decided_by": "staff-1",
            "decision_note": "Reviewed",
        },
    )

    assert response.status_code == status_code


def test_workflow_failure_returns_500() -> None:
    client = _client(
        approvals=_ApprovalRepository(),
        conversations=_ConversationRepository(),
        workflow=_Workflow(error=RuntimeError("workflow failed")),
    )

    response = client.post(
        f"/approvals/{_APPROVAL_ID}/decision",
        json={
            "decision": "approved",
            "decided_by": "staff-1",
            "decision_note": "Reviewed",
        },
    )

    assert response.status_code == 500


def test_invalid_decision_and_blank_note_are_rejected() -> None:
    client = _client(
        approvals=_ApprovalRepository(),
        conversations=_ConversationRepository(),
        workflow=_Workflow(),
    )

    invalid_decision = client.post(
        f"/approvals/{_APPROVAL_ID}/decision",
        json={
            "decision": "maybe",
            "decided_by": "staff-1",
            "decision_note": "Reviewed",
        },
    )
    blank_note = client.post(
        f"/approvals/{_APPROVAL_ID}/decision",
        json={
            "decision": "approved",
            "decided_by": "staff-1",
            "decision_note": "   ",
        },
    )

    assert invalid_decision.status_code == 422
    assert blank_note.status_code == 422


def test_missing_service_state_returns_503() -> None:
    response = _client().get("/approvals")

    assert response.status_code == 503
