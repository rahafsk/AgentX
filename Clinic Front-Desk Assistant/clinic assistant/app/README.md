# Application package

This directory contains the Python backend for the Clinic Front-Desk Assistant. The project provides a REST API for the clinic's existing chat client; it does not provide a website or frontend.

**Implementation status:** This is the target package structure. Some modules are still under development. Check the current implementation before relying on a planned endpoint or workflow.

## Entry point and local run

Run commands from the project root (the directory containing `pyproject.toml`):

```sh
uv sync --locked --extra dev
uv run uvicorn app.main:app --reload
```

The current application exposes `GET /health`. FastAPI's interactive API documentation is available at `/docs` while the service is running.

## Package layout

| Path | Responsibility |
| --- | --- |
| `main.py` | FastAPI application, router registration, and health endpoint. |
| `config.py` | Validated settings loaded from environment variables and local `.env`. |
| `api/` | HTTP endpoints for conversations, appointments, and staff approvals. |
| `graphs/` | LangGraph conversation state, intent routing, and workflow branches. |
| `services/` | Conversation, appointment, and clinic-knowledge behavior. |
| `repositories/` | PostgreSQL reads and writes using parameterized psycopg queries. |
| `db/` | PostgreSQL schema, development seed data, and database initialization. LangGraph checkpoint tables are managed by its supported PostgreSQL checkpointer setup. |
| `rag/` | Retrieval from clinic-approved documents, including source metadata. |
| `llm.py` | Deterministic fake model and configurable real-model adapter. |

The overall request flow and shared contracts are described in [`../docs/architecture.md`](../docs/architecture.md).

## Safety and data rules

- Route emergency messages before ordinary intents; return immediate emergency instructions and flag them for staff.
- Never diagnose or recommend treatment.
- Answer clinic questions only from approved documents, name the source, and fall back honestly when evidence is missing.
- Use PostgreSQL only. Do not add an ORM; parameterize SQL.
- Keep conversation history bounded and use a stable conversation ID as the LangGraph thread ID.
- Same-day bookings and cancellations pause for staff approval and must resume safely after a restart.
- Do not put patient data or real credentials in this package, tests, logs, or documentation.

## Configuration and models

Settings are documented in [`../.env.example`](../.env.example). Copy it to a local `.env`, replace development placeholders as needed, and never commit `.env` or real secrets.

The fake LLM is the default for local tests and CI. Real model requests use a configured OpenAI-compatible endpoint and timeout. Tests should use the fake adapter or `httpx.MockTransport`; do not call a real provider from CI.

## Tests

Tests live outside this package under `../tests/`:

- `unit/` — isolated settings, service, and adapter behavior.
- `api/` — HTTP request and response behavior.
- `database/` — PostgreSQL constraints and initialization; use a disposable test database only.
- `integration/` — end-to-end workflows, persistence, restart/resume, and safety routing.

Run the suite from the project root with `uv run pytest`. The project targets at least 80% coverage.
