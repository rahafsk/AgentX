# ADR-0001: PostgreSQL and psycopg for persistence

- **Status:** Accepted by the project blueprint
- **Date:** 2026-11-03

## Context

The backend must persist doctors, appointment slots, bookings, approvals, and conversation metadata. Booking requests may arrive concurrently, so correctness cannot depend only on application-level checks. The project also requires a single database technology and explicitly rules out an ORM.

## Decision

- Use PostgreSQL as the application's only database.
- Use psycopg and parameterized SQL through repository functions; do not add an ORM.
- Enforce booking integrity in PostgreSQL, including database-backed protection against duplicate reservations and overlapping doctor slots.
- Keep patient references minimal and do not store clinical details in documents or logs.

## Alternatives considered

- An ORM was rejected because the project requires direct, parameterized SQL and explicit transaction behavior.
- An in-memory or file-based database was rejected because it cannot provide the required PostgreSQL persistence and concurrency guarantees.

## Consequences

- Local development, database tests, and deployment require PostgreSQL.
- Database tests must use a disposable PostgreSQL database and verify constraints, not mock them.
- Schema changes belong in `app/db/schema.sql`; SQL calls remain parameterized in repositories.
