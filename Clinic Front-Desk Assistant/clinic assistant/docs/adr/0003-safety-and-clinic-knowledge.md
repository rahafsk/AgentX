# ADR-0003: Emergency-first safety and grounded clinic answers

- **Status:** Accepted by the project blueprint
- **Date:** 2026-11-03

## Context

The assistant operates in a healthcare setting. A routine booking or information workflow must never delay an emergency response, and model-generated answers must not invent clinic policies or provide medical advice.

## Decision

- Detect emergencies before ordinary intent routing and return immediate emergency instructions while flagging the case for staff.
- Never diagnose or recommend treatment.
- Answer clinic-information questions only from clinic-approved documents, name the source, and use an honest fallback when evidence is missing.
- Keep fictional RAG examples in a separate development-only directory; never mix them with approved production documents.

## Alternatives considered

- Sending every message through ordinary intent routing was rejected because emergency messages could be delayed or sent to an inappropriate workflow.
- Allowing the model to answer clinic-policy questions from general knowledge was rejected because it can invent clinic-specific facts.

## Consequences

- Emergency routing must bypass appointment and RAG branches.
- Retrieval tests must verify source attribution and no-evidence fallback behavior.
- Only verified, clinic-approved, non-patient reference information belongs in `data/clinic_documents/`.
