# R1 — PostgreSQL Checkpoint Persistence

- **Research owner:** Salwa (Member 2)
- **Documentation owner:** Rahaf (Member 1, issue [#51](https://github.com/rahafsk/AgentX/issues/51))
- **Status:** Pending — no experiment results recorded yet
- **Issue:** [#3](https://github.com/rahafsk/AgentX/issues/3)

## Research question

Does LangGraph conversation state stored through the PostgreSQL checkpointer survive an application/container restart and resume on the same conversation thread?

## Experiment plan

1. Use a disposable PostgreSQL test database and the supported LangGraph PostgreSQL checkpointer setup.
2. Start a conversation with a stable conversation ID/thread ID and save identifiable, non-sensitive test state.
3. Stop and restart the application process; reconnect to the same database.
4. Load the graph using the same thread ID and verify the saved state is restored and can continue.

## Evidence to record

- PostgreSQL, LangGraph, and checkpointer package versions.
- Exact test command or automated test name.
- Observed state before restart and after resume, using synthetic test values only.
- Any setup requirements, failures, or limitations.

## Findings and recommendation

Pending experiment. Do not record the expected result as a finding until the restart/resume test has run and evidence has been captured.
