# ADR-0002: LangGraph workflows with PostgreSQL checkpoints

- **Status:** Accepted by the project blueprint
- **Date:** 2026-11-03

## Context

Patient requests can branch into appointment, clinic-information, emergency, or staff-approval flows. Cancellations and same-day bookings must pause for a staff decision, and the workflow must resume correctly even after an application or container restart.

## Decision

- Use LangGraph to represent conversation state, intent routing, and workflow branches.
- Use the LangGraph PostgreSQL checkpointer and its supported setup/migrations; do not duplicate its internal checkpoint schema in application SQL.
- Keep one stable conversation identifier mapped to the LangGraph `thread_id`.
- Persist the approval record and workflow state needed to resume the correct conversation after a staff decision.

## Alternatives considered

- Process-local state was rejected because it is lost on restart and cannot support durable approval pauses.
- Hand-written copies of LangGraph's internal checkpoint tables were rejected because they can diverge from the library's migrations.

## Consequences

- Initialization must set up both application tables and LangGraph checkpoint storage.
- Workflow tests must cover pause, restart, and resume behavior using PostgreSQL.
- Conversation and checkpoint identifiers are a shared contract between the conversation API, graph state, and workflow services.
