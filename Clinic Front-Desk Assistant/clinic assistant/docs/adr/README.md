# Architecture Decision Records

This directory records durable design decisions for the Clinic Front-Desk Assistant. ADRs describe why a decision was made, not just how a module works. The records below capture decisions specified by the project blueprint; implementation status may differ while the system is under development.

## Records

- [ADR-0001: PostgreSQL and psycopg for persistence](0001-postgresql-and-psycopg.md)
- [ADR-0002: LangGraph workflows with PostgreSQL checkpoints](0002-langgraph-durable-workflows.md)
- [ADR-0003: Emergency-first safety and grounded clinic answers](0003-safety-and-clinic-knowledge.md)
- [ADR-0004: Deterministic fake and timed real LLM adapters](0004-llm-adapters.md)

## Adding a decision

Create the next numbered Markdown file. Include its status, date, context, decision, alternatives considered, and consequences. If an accepted decision changes, add a new ADR that supersedes the old one rather than silently rewriting the history.
