# Clinic Front-Desk Assistant

A **backend REST API** for the clinic's existing chat client. The planned system uses FastAPI, LangGraph, and PostgreSQL to support conversations, appointment requests, staff approvals, and answers grounded in clinic-approved information. No website or frontend is included.

> **Status: Work in progress.** At present, the implemented API endpoint is `GET /health`. Conversation, appointment, approval, graph, repository, RAG, and database-schema modules are still under development. See [`docs/architecture.md`](docs/architecture.md) for the target design.

## Requirements

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/)

## Set up

From the cloned repository root:

```sh
cd "Clinic Front-Desk Assistant/clinic assistant"
uv sync --locked --extra dev
```

## Run the current API

From the project directory (the one containing `pyproject.toml`):

```sh
uv run uvicorn app.main:app --reload
```

- Health check: <http://127.0.0.1:8000/health>
- Interactive API documentation: <http://127.0.0.1:8000/docs>

The current health endpoint returns:

```json
{"status":"ok"}
```

## Configuration

`.env.example` lists the planned environment variables. Copy it to `.env` for local development and replace placeholder values as needed. Never commit `.env`, real credentials, or API keys. The fake LLM mode is intended for local tests and CI; real LLM requests require a configured OpenAI-compatible endpoint and must use a timeout.

Database initialization is not ready yet: `app/db/schema.sql` and `app/db/seed.sql` still need their clinic database definitions. Do not point development or test commands at a production database.

## Safety requirements

- Never diagnose or recommend treatment.
- Detect emergencies before normal request routing and return immediate emergency instructions while flagging the case for staff.
- Use only clinic-approved documents for clinic-information answers. Name sources and fall back honestly when evidence is missing.
- Keep patient data out of documentation, fixtures, logs, and `data/clinic_documents/`.
- Use PostgreSQL only, parameterized SQL through psycopg, and no ORM.

## Tests and quality checks

Tests belong under `tests/unit/`, `tests/api/`, `tests/database/`, and `tests/integration/`. Use a disposable PostgreSQL database for database or integration tests, and use the fake LLM or a mocked HTTP transport rather than a real model provider.

Once test modules are available, run from the project directory:

```sh
uv run ruff check .
uv run ruff format --check .
uv run mypy app
uv run pytest
```

The project targets at least 80% test coverage. See [`app/README.md`](app/README.md) for package responsibilities and [`AGENTS.md`](AGENTS.md) for ownership and contributor guidance.
