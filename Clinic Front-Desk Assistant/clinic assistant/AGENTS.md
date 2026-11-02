# Repository Guidance

## Project scope

- Build the clinic front-desk **backend REST API** only. The clinic already has a chat client; do not add a website or frontend.
- Use FastAPI for HTTP, LangGraph for conversation workflows, and PostgreSQL as the only database.
- Keep this document aligned with `docs/architecture.md` and the current implementation. The architecture document describes the target design; do not assume every planned module is complete.

## Safety and privacy

- Never diagnose a patient or recommend treatment.
- Check for emergencies before ordinary intent routing. Return immediate emergency instructions and flag the case for staff; do not route emergencies through booking or RAG.
- Answer clinic-information questions only from clinic-approved documents. Cite the source and give an honest fallback when evidence is missing.
- Do not add patient data, real credentials, API keys, or secrets to the repository, tests, logs, or documentation.
- Store only the minimum patient reference needed for front-desk operations. Keep local `.env` files and database credentials uncommitted.
- Use fictional fixtures only in an explicitly isolated test/development database. Never seed them into production.

## Architecture and shared contracts

- PostgreSQL is the only database. Use psycopg with parameterized SQL; do not add an ORM.
- Appointment booking must be atomic and protected by database constraints against duplicate or overlapping reservations.
- A conversation identifier must consistently map to the LangGraph thread ID. Keep message history bounded.
- Persist LangGraph checkpoints in PostgreSQL using the package's supported checkpointer setup/migrations; do not duplicate its internal table definitions.
- Same-day bookings and cancellations require a durable staff-approval pause/resume flow.
- Use the deterministic fake LLM for tests and CI. Real model requests must have a timeout; never log API keys or full patient messages.
- Agree on shared state, repository, and routing contracts before changing another member's owned files.

## File ownership

| Owner | Assigned files / areas |
| --- | --- |
| Rahaf — team leader | `app/main.py`, `app/config.py`, `app/llm.py`, `app/rag/`, `app/repositories/`, `app/db/`, cross-module integration tests, and root tooling/configuration. |
| Salwa — conversations | `app/api/conversations.py`, `app/graphs/states.py`, `app/services/conversations.py`. |
| Alkulaikha — appointments and routing | `app/api/appointments.py`, `app/graphs/routing.py`, `app/services/appointments.py`. |
| Asila — approvals and workflows | `app/api/approvals.py`, `app/graphs/workflows.py`, `app/services/clinic_knowledge.py`. |

Do not edit another owner's assigned files without coordinating with that owner first. Shared contracts and cross-module changes should be discussed before implementation.

## Code and test expectations

- Keep code typed, focused, and consistent with nearby patterns. Avoid adding dependencies unless the feature requires them.
- Add tests with each behavior change:
  - `tests/unit/` for isolated logic; use `httpx.MockTransport` for real-LLM adapter tests.
  - `tests/api/` for HTTP request/response behavior.
  - `tests/database/` for PostgreSQL constraints, triggers, and initialization.
  - `tests/integration/` for complete workflows, persistence, restart/resume, and safety routing.
- Use a disposable PostgreSQL test database only. Never run integration tests against production data.
- Keep the configured 80%+ coverage target; tests should verify behavior rather than implementation details.
- Run these checks from the project directory before opening a PR:

  ```sh
  uv sync --locked --extra dev
  uv run ruff check .
  uv run ruff format --check .
  uv run mypy app
  uv run pytest
  ```

## Changes and reviews

- Work on a feature branch and open a pull request to `main`; do not push directly to `main`.
- Keep PRs focused, describe behavior and safety impact, and include relevant tests.
- Follow the repository policy of two approvals and passing CI before merging.
- Update `CHANGELOG.md` for notable changes. Record research in `docs/research/` and durable design decisions in `docs/adr/`.
