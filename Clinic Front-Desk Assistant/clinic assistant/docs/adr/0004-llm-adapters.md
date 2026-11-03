# ADR-0004: Deterministic fake and timed real LLM adapters

- **Status:** Accepted by the project blueprint
- **Date:** 2026-11-03

## Context

The application needs model responses for conversational flows, while continuous integration must be deterministic and must not depend on network access or external credentials. Real provider calls can fail or stall.

## Decision

- Provide a deterministic fake adapter for local tests and CI.
- Support a configurable real OpenAI-compatible chat endpoint for development/deployment.
- Apply a finite request timeout and convert provider failures into safe application errors.
- Keep API keys in environment variables; do not log secrets or patient messages.

## Alternatives considered

- Calling a real model in CI was rejected because responses, latency, availability, and credentials are not deterministic.
- An unbounded provider request was rejected because it could leave API requests waiting indefinitely.

## Consequences

- Tests should use the fake adapter or a mocked HTTP transport.
- Real-model use requires an explicit model and base URL, plus credentials when the provider requires them.
- Workflows remain responsible for emergency-first routing and the prohibition on diagnosis or treatment recommendations.
