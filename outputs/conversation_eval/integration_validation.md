# Targeted production conversation integration validation

**Status:** 5 targeted mocked integration tests passed.

## What ran

The tests used complete supplied conversation sequences from:

- `PC002` — high-value damaged item, verification success, ticket creation, Priority Resolution workflow.
- `PC003` — successful verification, return eligibility and refund follow-up.
- `PC005` — human request, verification, ticket success/failure and no-handoff behavior.
- `PC013` — policy-only coupon question followed by a return to the verified order topic.
- `PC015` — compensation request and refusal.
- `PC021` — failed verification attempts and privacy boundary.

Runtime messages were loaded from `practice/conversations.csv`; expected labels and notes were not sent to the responder.

## Real and mocked components

- **Real:** `run_conversations`, its per-conversation state and verification controller, `RAGConversationResponder`, `answer_with_evidence`, citation validation, deterministic RAG safeguards, current Markdown/CSV evidence records, and `EvidenceRetriever` ranking/required-section logic.
- **Mocked:** Ollama chat output, the query encoder vectors, order lookup responses, and ticket creation responses. The encoder uses deterministic category vectors to exercise retrieval code without downloading or invoking an embedding model. Tool mocks produce no ticket files or external side effects.
- **Not tested:** live Ollama answers, live embedding quality, real tool writes, Gnani, or voice.

These results are mocked integration behavior checks, not live answer-quality measurements.

## Findings and fix

The high-value damaged-item branch returned the correct `escalate` decision, required evidence, and ticket action. After the ticket tool reported success, however, the runner replaced the RAG response with a generic ticket confirmation and omitted the policy-required **Priority Resolution** destination. `conversation_runner.py` now preserves that review target in the ticket confirmation while explicitly stating that no human transfer has taken place. Ticket failure still reports no successful ticket and no handoff.

No verification-digit or private-order leak was detected in tested prompts, histories, or run results. Failed verification exhausted the allowed attempts without invoking ticket creation. A model-generated ticket claim did not become a success claim when the injected ticket tool returned `created=False`.

## Test coverage

`tests/test_conversation_integration.py` covers successful and failed verification, proof redaction, verified order plus policy answers, policy-only topic switching and follow-up, refusal, high-value escalation, ticket creation success/failure, and unexecuted handoff behavior. The ticket creator and lookup functions are injected mocks.

Run with:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_conversation_integration -v
```

## Limitations

The encoder and model responses are deterministic fixtures. Retrieval results verify the integration and guard paths, not the quality of learned semantic similarity. The synthetic fixture does not establish behavior under live model variation or production authentication. No handoff API exists, so `handoff_performed` remains false.
