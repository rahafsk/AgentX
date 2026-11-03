# R2 — Durable Human-in-the-Loop Approval

- **Research owner:** Alkulaikha (Member 3)
- **Documentation owner:** Rahaf (Member 1, issue [#51](https://github.com/rahafsk/AgentX/issues/51))
- **Status:** Pending — no experiment results recorded yet
- **Issue:** [#4](https://github.com/rahafsk/AgentX/issues/4)
- **Coordination:** Share the research decision with Asila (Member 4), who owns the approvals API and workflow implementation.

## Research question

Can a cancellation or same-day booking pause for front-desk approval, remain pending across an application/container restart, and resume the correct conversation after staff approves or rejects it?

## Experiment plan

1. Use a disposable PostgreSQL database and synthetic conversation/booking fixtures.
2. Run a cancellation or same-day booking workflow until it is interrupted for staff review.
3. Confirm the approval and checkpoint are persisted, restart the application, then load the same conversation thread.
4. Submit both an approval and a rejection in separate runs and verify the matching workflow resumes with the correct result.

## Evidence to record

- Pending approval state before and after restart.
- Conversation/thread ID used to resume the graph.
- Test commands, observed outcomes, and any failure/recovery behavior.
- Limitations or contract decisions shared with Asila.

## Findings and recommendation

Pending experiment. Record only behavior verified against the implementation; do not infer restart safety from an in-process test.
