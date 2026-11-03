# R3 — Grounded Clinic Answers and Fallback

- **Research owner:** Asila (Member 4)
- **Documentation owner:** Rahaf (Member 1, issue [#51](https://github.com/rahafsk/AgentX/issues/51))
- **Status:** Pending — no experiment results recorded yet
- **Issue:** [#5](https://github.com/rahafsk/AgentX/issues/5)
- **Coordination:** Agree on the retrieval interface with Rahaf, who owns `app/rag/`.

## Research question

Do clinic-information answers use only approved documents, identify the source document, and fall back honestly when the documents do not contain the answer?

## Experiment plan

1. Use a small, explicitly approved or clearly fictional test corpus outside production data; do not use patient information.
2. Ask a question whose answer is present in one document and verify the response cites that document's source name.
3. Ask a question not covered by the corpus and verify the assistant states that it does not have enough information instead of inventing a clinic fact.
4. Verify that retrieved content does not produce diagnoses or treatment recommendations.

## Evidence to record

- Test document filenames/source labels and how the corpus was isolated.
- Queries, relevant retrieved sources, and observed answers.
- Test command or automated test name.
- Retrieval thresholds, fallback behavior, and remaining limitations.

## Findings and recommendation

Pending experiment. Do not add fabricated clinic facts to `data/clinic_documents/`; demo material must remain isolated from production retrieval.
