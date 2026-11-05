"""PostgreSQL checkpoint restart test for staff approval pauses."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast
from uuid import uuid4

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.types import Command

from app.graphs.states import ConversationState
from app.graphs.workflows import APPROVAL_HANDLER, build_workflow
from app.rag import RetrievedChunk
from app.services.clinic_knowledge import ClinicKnowledgeService


class _NeverCalledRetriever:
    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 3,
    ) -> Sequence[RetrievedChunk]:
        raise AssertionError("Retriever must not be called.")


async def _pending_appointment(state: ConversationState) -> Mapping[str, Any]:
    return {
        "response": "Waiting for staff approval.",
        "approval_status": "pending",
    }


def _state(text: str, conversation_id: str) -> ConversationState:
    return cast(
        ConversationState,
        {
            "conversation_id": conversation_id,
            "messages": [HumanMessage(content=text)],
        },
    )


async def test_pending_approval_survives_new_checkpointer(database_url: str) -> None:
    thread_id = f"workflow-restart-{uuid4()}"
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    knowledge_service = ClinicKnowledgeService(_NeverCalledRetriever())

    async with AsyncPostgresSaver.from_conn_string(database_url) as saver:
        workflow = build_workflow(
            knowledge_service=knowledge_service,
            appointment_handler=_pending_appointment,
            checkpointer=saver,
        )

        first = await workflow.ainvoke(
            _state("I need to cancel my appointment.", thread_id),
            config,
        )
        assert first["approval_status"] == "pending"
        assert "__interrupt__" in first

        pending = await workflow.aget_state(config)
        assert pending.values["approval_status"] == "pending"
        assert pending.next == (APPROVAL_HANDLER,)

    # Simulate a restart with a new checkpointer connection.
    async with AsyncPostgresSaver.from_conn_string(database_url) as saver:
        workflow = build_workflow(
            knowledge_service=knowledge_service,
            appointment_handler=_pending_appointment,
            checkpointer=saver,
        )

        restored = await workflow.aget_state(config)
        assert restored.values["approval_status"] == "pending"
        assert restored.next == (APPROVAL_HANDLER,)

        final = await workflow.ainvoke(
            Command(resume={"decision": "approved"}),
            config,
        )
        assert final["approval_status"] == "approved"

        await saver.adelete_thread(thread_id)
