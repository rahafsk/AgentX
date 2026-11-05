# Changelog

Notable changes to the Clinic Front-Desk Assistant are recorded here. This project follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and Semantic Versioning.

## [Unreleased]

### Added

- Add owner-scoped conversation and appointment APIs with UUID idempotency keys, trusted caller identity, and checkpointed booking/cancellation actions.
- Add staff approval queue and decision endpoints that update booking records and resume paused LangGraph threads.
- Reject elapsed appointment slots and replay booking/approval actions safely after a database commit.
- Return real selectable slot/booking options in conversation workflows, require explicit appointment selection, and provide actionable rescheduling guidance.
- Persist conversation turn idempotency results outside bounded message history and recover approved actions when a checkpointed pause needs replay.
- Require trusted staff identity for approval access, use pooled per-operation database connections, expose readiness only after service configuration, and move schema migrations out of API startup.
- Recover stale approval pauses by replaying their idempotent appointment action, serialize patient and staff runs through the same conversation lock, and report appointment failures explicitly.
- Offer actual available slots and upcoming bookings for explicit selection, and return actionable rescheduling guidance instead of silently choosing a booking.
- Add owner-scoped conversation processing with request IDs, bounded history, approval-pending replies, and workflow timeouts.
- Add a local Docker Compose setup for the FastAPI service and PostgreSQL, with health checks, initialization, and persistent database storage.
- Scaffold the backend package structure for the API, LangGraph workflows, services, repositories, RAG, database, tests, and documentation.
- Add the FastAPI application entry point and `/health` endpoint.
- Add a PostgreSQL initialization command that applies the schema and seed SQL files through psycopg.
- Add parameterized PostgreSQL repositories for doctors, appointment slots, bookings, conversations, and approvals, with atomic booking and approval operations.
- Add deterministic fake and OpenAI-compatible LLM adapters with request timeouts and error handling.
- Add BM25-style lexical retrieval for approved Markdown and text documents, including source names, no-evidence results, and guarded development-only demo retrieval.
- Document the target backend architecture, request flow, data contracts, and safety requirements.
- Add typed repository domain exceptions for doctor, conversation,
  appointment, booking, and approval operations.
- Add DoctorRepository to retrieve doctors by ID and list active doctors.
- Add `ClinicKnowledgeService` for grounded clinic-information answers from approved documents, including source citations, an honest no-evidence fallback, and refusal/screening of diagnosis or treatment advice.

### In progress

- The embedding host must provide trusted owner and staff authentication resolvers; authentication-provider integration is not included here.
- Full deployment integration remains in progress.
- No versioned release has been published yet.
