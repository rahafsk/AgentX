# Changelog

Notable changes to the Clinic Front-Desk Assistant are recorded here. This project follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and Semantic Versioning.

## [Unreleased]

### Added

- Scaffold the backend package structure for the API, LangGraph workflows, services, repositories, RAG, database, tests, and documentation.
- Add the FastAPI application entry point and `/health` endpoint.
- Add a PostgreSQL initialization command that applies the schema and seed SQL files through psycopg.
- Add deterministic fake and OpenAI-compatible LLM adapters with request timeouts and error handling.
- Document the target backend architecture, request flow, data contracts, and safety requirements.

### In progress

- Conversation, appointment, approval, routing, workflow, service, repository, and RAG modules are scaffolded but not yet implemented.
- The database schema and seed SQL are not yet populated on this branch.
- No versioned release has been published yet.
