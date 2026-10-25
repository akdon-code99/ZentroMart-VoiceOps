# Capstone practice dataset: ZentroMart customer support

Everything here is synthetic. ZentroMart is a fictional Indian online store. No real customers, orders, phone numbers or recordings were used. Phone numbers use an unallocated prefix (+91 5…) and every email is on example.com.

**Today's date for this dataset is 20 October 2026.** Use it for every deadline and delay calculation, not your computer's clock.

## What you're building with it

A support agent for ZentroMart that answers customers who write in by chat, email or support ticket. In Step 4 you add voice as a feature: customers can send a voice message, which your agent transcribes with Gnani STT over WebSocket, and your agent can reply with audio streamed from Gnani TTS. No prerecorded audio files are supplied. Record your own voice to test Gnani STT/TTS.

## What's in the folder

Every data file is a CSV: 9 in total. The company documents are Markdown files, because they are written documents meant to be read and searched, and the tools are one Python file.

| File | Contents | Used in |
| --- | --- | --- |
| `kb/*.md` | 21 company documents: policies, agent guides, help articles, FAQs (one in Hindi) and a marketing page | Steps 2 to 4 |
| `data/products.csv` | 70 products with return category, price, seller and warranty | Steps 2 and 3 |
| `data/customers.csv` | 300 fictional customers | Step 3, through the tools |
| `data/orders.csv` | 1,000 orders, one row each: payment, delivery, returns, refunds and a timeline | Step 3 |
| `data/order_items.csv` | What was bought in each order, one row per item, linked by `order_id` | Step 3 |
| `data/pincodes.csv` | Where ZentroMart delivers, and where cash on delivery and Express are available | Steps 2 and 3 |
| `data/holidays.csv` | The holidays that business-day counts skip | Steps 2 and 3 |
| `data/support_logs.csv` | 244 past support chats and calls handled by human agents | Optional |
| `practice/questions.csv` | 52 single questions with expected decisions, key facts, the document sections that answer them, and a speech-to-text version of each question | Steps 1, 2 and 4 |
| `practice/conversations.csv` | 23 multi-turn support conversations (96 customer-message rows), one row per customer message, with the expected behaviour at every turn | Step 3 |
| `tools/tools.py` | Two offline mock tools for identity-checked order lookup and ticket creation | Step 3 |

Field-by-field descriptions are in `DATA_DICTIONARY.md`.

## Working with the CSV files

- They open in Google Sheets or Excel, and in Python with `pandas.read_csv`. Use `dtype=str` so order IDs, pincodes and SKUs stay as text.
- Some cells hold a short list. List items are separated by ` ; ` (space, semicolon, space).
- In the `must_include` column, ` / ` separates alternatives: any one of them is enough.
- Empty cells mean "not applicable", for example no return on that order.

## The mock tools (identity-checked)

`tools/tools.py` runs locally and reads the CSV files in `data/`. No real accounts or personal information are needed. **Order details are withheld until verification succeeds.** The tool cannot securely authenticate real customers: it only simulates the capstone's identity-checking flow, with synthetic data.

```python
from tools import lookup_order, create_ticket

pending = lookup_order("8978 2837")
# -> {"verification_required": True, ...}; no private order details

# Ask the caller for the last four digits of their registered phone
# OR their delivery pincode. Never say the expected value aloud.
order = lookup_order("8978 2837", verification_value="4533")
assert order["found"] and order["verified"]
# Only NOW use order["status"], order["items"], and order["total_inr"].

ticket = create_ticket(
    summary="Verified caller reports a damaged TV on order 89782837; "
            "human review is required under the damage policy.",
    order_id="89782837", reason="high_value_damage",
)
```

- `lookup_order(order_id)` returns a verification prompt only. It does **not** return order status, items, pincode, phone details, amount, or customer information.
- `lookup_order(order_id, verification_value=...)` checks a **4-digit registered phone suffix** or **6-digit delivery pincode**, returning order details only on a match. Failed and unknown lookups do not disclose private data.
- Ask for only one verification value at a time, allow at most one retry, and do not share order details if both attempts fail. This is a rule for your conversation agent; it does not impose a persistent rate limit inside the mock tool.
- Keep verified order state in the current conversation only. Do not expose or repeat verification secrets in your response, logs, or handover summary.
- `create_ticket` records a handover in `tools/tickets_created.csv` and returns a ticket ID. Only say a ticket was created if the tool returned `created=True`. This file is generated at runtime and is not supplied in the ZIP.
- Practice conversations specify an unverified lookup when an order ID first appears and a verification lookup when the caller provides the phone suffix/pincode. See `practice/conversations.csv`.

## Things that make this realistic

Real support data is messy, and so is this. Build for it:

- **Documents disagree.** Policies, help articles, FAQs and a marketing page sometimes say different things. `kb/00_about_this_knowledge_base.md` says which one wins.
- **Policies change over time.** Every document has a `status` and an `effective_from` date in its header. Some answers depend on when an order was placed or delivered.
- **Customers don't use the documents' words.** People write "send it back" and "nobody came to collect it", while the documents say "reverse pickup". Keyword search alone will miss a lot.
- **Some answers need two sources.** A policy plus a catalog row, or a policy plus an order record.
- **Some questions have no answer** in the documents. The right reply says so and offers a human colleague.
- **Customers write in Hindi or Hinglish**, and a few in other Indian languages, while most documents are in English.
- **Timelines use business days**: Monday to Saturday, skipping the holidays in `data/holidays.csv`.
- **Verify before you share.** The agent guide `agent_identity_verification.md` explains how.
- **The support logs are not a source of truth.** They were written by human agents, and a few contain mistakes. Use them to learn how customers talk; answer from the documents.

## Section IDs

Every section heading carries an ID in square brackets, such as `## [RET-2] Return windows by category`. Keep this ID with each chunk you index, and report the IDs your agent used. That's how retrieval is scored, so you're free to chunk however you like.

## ZIP schema — folder structure, table keys and usage

The archive extracts directly into the following files and folders (no duplicate PDF or hidden macOS files):

```text
Capstone_Project_Dataset_FTTT_2026/
├── README.md                # Read first; project rules and usage
├── DATA_DICTIONARY.md       # Every CSV field, type and meaning
├── ZIP_SCHEMA.md            # Structure and relational schema overview
├── CHANGELOG.md             # Changes from the supplied draft archive
├── validate_dataset.py     # Offline, standard-library validation
├── kb/                     # 21 Markdown knowledge documents
│   ├── 00_about_this_knowledge_base.md
│   ├── agent_*.md           # Verification, conversation, escalation
│   ├── pol_*.md             # Current + superseded policies
│   ├── help_*.md, faq_*.md, mkt_*.md
├── data/                   # 7 CSV data tables
│   ├── customers.csv        # 300   PK: customer_id
│   ├── orders.csv           # 1,000 PK: order_id; FK: customer_id
│   ├── order_items.csv      # 1,783 PK: (order_id, line_no); FK: sku
│   ├── products.csv         # 70    PK: sku
│   ├── pincodes.csv         # 36    PK: pincode
│   ├── holidays.csv         # 4     PK: date
│   └── support_logs.csv     # 244   PK: log_id; references order/customer
├── practice/               # Text-only practice; no supplied audio
│   ├── questions.csv        # 52 individual questions
│   └── conversations.csv    # 96 turns across 23 conversations
└── tools/
    ├── __init__.py
    └── tools.py             # lookup_order and create_ticket
```

**Relational joins:** `customers.customer_id → orders.customer_id`, `orders.order_id → order_items.order_id`, `order_items.sku → products.sku`, and `orders.delivery_pincode → pincodes.pincode`. Customer pincodes also reference `pincodes.pincode`. The composite key for each order item is `(order_id, line_no)`. Join the logs by `order_id` and `customer_id` for optional analysis; do not treat them as policy authority.

**How to use:** Index `kb/*.md` for retrieval, retaining `[SECTION-ID]`, `status`, and `effective_from`. Use `data/*.csv` for factual lookups, `tools/` for simulated verified orders and escalations, and `practice/*.csv` for evaluation. Interpret time-dependent scenarios using **20 October 2026**, the fixed dataset reference date; exclude Sundays and listed holidays from business-day calculations. Run `python validate_dataset.py` from the extracted folder before starting.

See `ZIP_SCHEMA.md` for a concise table of join keys and `DATA_DICTIONARY.md` for every field.

## Step 1: LLM-only baseline

The baseline sends each standalone practice question to the configured Ollama Cloud model without retrieving knowledge-base documents or calling order/ticket tools. It uses the question's requested reply language, but keeps expected answers and other evaluation fields out of the model prompt.

From the project root with `.venv` activated, run:

```powershell
python .\run_baseline.py --limit 3
```

Each question makes one live API request. Omit `--limit 3` to run all practice questions. Results are written to `outputs/baseline_results.csv`, preserving the practice evaluation columns and adding `generated_answer` and `error`. Failed requests are recorded in `error`; no answer is fabricated. The result file is ignored by Git.

## Step 2: policy-grounded retrieval

Build the local NumPy index from section-aware KB content and selected product, pincode and holiday records:

```powershell
python .\build_rag_index.py
```

The first build downloads `intfloat/multilingual-e5-small` into the Hugging Face model cache. The index files are generated under `.cache/rag_index/`; delete that directory and rerun the build command to rebuild from the supplied source files. The retriever uses E5 query/passage prefixes and excludes superseded Markdown by default. Current-policy precedence follows `KB-00.1`; a similarity score never makes a superseded rule authoritative.

Measure retrieval against the practice evidence without making LLM requests:

```powershell
python .\evaluate_retrieval.py
```

This writes to `outputs/retrieval_evaluation_latest.csv` so the earlier `outputs/retrieval_evaluation.csv` remains available as a comparison point. To test the optional pool-16 relevance-aware ranking separately, use:

```powershell
python .\evaluate_retrieval.py --candidate-pool 16 --authority-adjustment 0.03 --relevance-window 0.002 --max-product-rows 2 --output .\outputs\retrieval_evaluation_experiment.csv
```

The RAG answer path uses this pool-16 ranking configuration. The authority adjustment is bounded and only applied to candidates close to the top-k similarity cutoff; generic product rows are capped. Superseded Markdown remains excluded unless a dated event, uniquely matched catalog product, current successor policy, and older policy version all establish that the successor explicitly scopes the event to the historical rule. In that case, the current successor is labeled governing evidence and the older section is labeled historical reference only.

To smoke-test the RAG answer path against the first three questions, run:

```powershell
python .\run_rag.py --limit 3 --output .\outputs\rag_results_smoke.csv
```

Each question makes one Ollama Cloud request. The runner validates local configuration first, writes errors and run status to the result CSV, and stops after an authentication rejection rather than sending more requests. It records evidence IDs, evidence keys and source metadata. Evaluation references are saved for review but are not sent to the answer model.

Only after the three-question run succeeds, generate answers for all 52 practice questions:

```powershell
python .\run_rag.py --output .\outputs\rag_results.csv
```

Create a separate baseline/RAG comparison and manual-review file with:

```powershell
python .\compare_rag.py --baseline .\outputs\baseline_results.csv --rag .\outputs\rag_results.csv --retrieval .\outputs\retrieval_evaluation_historical.csv --output .\outputs\baseline_vs_rag.csv
```

The comparison joins by question ID, adds retrieved text from the local index for evidence review, flags zero/partial/no-gold retrieval groups, and performs only a structural citation-ID check automatically. Human review is required for evidence support, factual correctness, completeness, policy-version correctness, abstention, and unsupported claims. No-gold evidence is not treated as proof that a question is unanswerable. These generated outputs are separate from the baseline and retrieval-evaluation reports.

### Step 3B: evidence routing and answer safeguards

The RAG retriever adds current policy sections relevant to a question's intent (for example, refund method, warranty, return condition, or damaged-item handling) while retaining a strict top-k limit. Markdown preambles are excluded from candidate ranking only when that same document has identified sections; preambles from unsectioned documents remain eligible. This prevents generic introductions from crowding out section-level evidence. For a dated refund question, the answer path reads the matching method's range from `REF-2` and Sunday/holiday rules from `KB-00.3`; it excludes the initiation date and counts forward from the next eligible business day, matching the practice references. The date answer is deterministic and cites both sections.

### Step 3C: conversation state skeleton

`conversation_runner.py` loads only the conversation ID, numeric turn, language, and customer message from `practice/conversations.csv`, groups rows by conversation, and orders each conversation by turn. Its callback-based runner keeps separate in-memory state for each conversation, including prior exchanges, customer-provided issue context, the latest topic, an explicitly provided order ID, verification status/attempt count, and ticket-creation status. The responder receives only the current customer input and runtime context; expected decisions, rubric fields, tool-call expectations, gold evidence, and evaluator notes are not included.

The verification controller calls an injected `lookup_order` function only for an order-specific workflow with a known order ID. It asks for either the registered phone’s last four digits or the delivery pincode, and treats the caller as verified only when the tool returns both `found=True` and `verified=True`. Two proof attempts are allowed per order; changing topic or wording does not reset the count. Failed/exhausted verification never releases order details, and ticket state is not verification. Verification-proof messages are redacted before storage in history, issue context, responder input, or run results. The runner does not implicitly invoke tools or the LLM; they run only when the caller injects the corresponding tool/responder, and tests use mocks.

### Step 3D: policy and verified-order answers

`RAGConversationResponder` adapts conversation turns to the existing `answer_with_evidence` pipeline. It passes recent prior customer messages with the current question to resolve follow-ups, and passes verified order facts separately. The runner stores only an allowlisted operational snapshot from a successful lookup, excluding phone, email, address, customer ID, and delivery pincode; each responder call receives only fields relevant to the current question. RAG retains its current-policy retrieval, citation validation, deterministic safeguards, and uncertainty behavior.

Structured RAG decisions (`answer`, `clarify`, `escalate`, or `refuse`) are retained in each turn result; refusal and question-form clarification use explicit response cues, while missing-evidence and required-action cases use RAG grounding metadata. When RAG requires escalation, the runner calls an injected ticket creator first and deduplicates by order, required action, and issue intent. It reports ticket success only on `created=True`. No handoff function is supplied, so the runner always reports that a direct human transfer has not occurred; ticket creation is not represented as a completed handoff.

For a damaged item priced above ₹10,000, the answer path requires `DWM-3`, `ESC-1`, `ESC-2`, and the matching product row. A reporting-only question is answered deterministically only when current `DWM-1` unambiguously specifies one numeric window anchored to delivery and explicitly requires photos of both the item and packaging. Conflicting or incomplete policy wording falls back to a safe uncertainty response; questions about an already-filed report do not use the reporting-only answer. Remedy, pickup-fee, and high-value workflow questions remain separate. Unrecognized-order reports also follow the ticket-before-handoff workflow. Both workflows mark the required action `not_executed`; this single-turn RAG runner does not invoke the supplied mock ticket tool, so it never claims that a ticket or handoff occurred. COD refund options are rendered from the current `REF-2` and `REF-3` evidence, including bank-detail verification. Citation requirements follow the question and answer claims: damage-reporting claims cite `DWM-1`, remedy claims cite `DWM-2`, and pickup-fee claims cite `RET-6`; return-condition and return-window questions use their respective sections. Model-generated answers are rejected to a cautious response if they cite evidence that was not retrieved or omit required policy citations. These checks validate evidence availability and citation identity, not semantic support; manual factual review is still required.

To preserve the original RAG and comparison files, use new filenames for any later live answer run and comparison, for example:

```powershell
python .\run_rag.py --output .\outputs\rag_results_v2.csv
python .\compare_rag.py --baseline .\outputs\baseline_results.csv --rag .\outputs\rag_results_v2.csv --retrieval .\outputs\retrieval_evaluation_step3b_v2.csv --output .\outputs\baseline_vs_rag_v2.csv
```

The runner refuses to overwrite an existing result file. A live rerun should be performed only after review and approval; unit tests and the offline retrieval evaluation do not make Ollama requests.

### Step 3E: offline conversation evaluator

Run the conversation evaluator without Ollama or real ticket writes:

```powershell
python .\conversation_evaluator.py
```

It reads the supplied rubric separately from runtime messages, then sends only ordered customer-facing messages through `run_conversations`. By default it uses a limited deterministic responder, a local read-only fixture for order verification, and a no-write ticket fixture. New timestamped CSV, JSON, and Markdown reports are written under `outputs/conversation_eval/`; existing reports are not overwritten.

The results are an offline harness/fixture check, not a live-agent quality score. Required and forbidden facts are checked using normalized alternatives and common date/digit normalization. Evidence checks measure gold-section ID overlap only, not whether the evidence semantically supports a claim. The report defines metric denominators and marks empty rubric categories as not applicable. Actual tool logs include safe order IDs but never verification digits; expected verification values are redacted. The handoff status remains false because no handoff action is available.

### Step 4 Phase 1 and Phase 2: voice adapters and local audio

`voice_adapter.py` provides separate Gnani STT and TTS adapters and a
`VoiceConversationSession` that preserves the runner's in-memory conversation
state between audio turns. STT expects mono, signed 16-bit little-endian PCM
frames of exactly 1,024 bytes at the configured sample rate; the documented
16 kHz default sends one frame every 32 ms. `LocalAudioInterface` captures one
short utterance at 16 kHz mono int16 and prepares the required frames; it also
plays the streamed TTS response when it is a valid 16-bit PCM WAV. Audio capture
and playback use `sounddevice`, which is declared but not installed here.
The `websocket-client` package is also declared for live STT and is not
installed by this implementation task.

Set `GNANI_API_KEY` in the local `.env` file. `.env` is ignored by Git; never
commit or print the key. TTS uses Gnani Timbre v2.5 and the documented
language-specific voice defaults in `voice_adapter.py`. The adapter groups
order IDs and renders ISO dates with month names; Gnani's documented
language-aware text normalization handles rupee amounts during synthesis. Run
the mocked audio and adapter tests without contacting Gnani or Ollama:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_voice_adapter -v
```

The STT WebSocket protocol, TTS Server-Sent Events, and local audio backend are
exercised with mocks. Passing these tests does not verify live credentials,
service availability, real-device permissions, audio capture quality,
transcription accuracy, or speaker output.

### Step 4 Phase 3: batch audio runner

Run a folder of customer audio files through Gnani STT, the existing
RAG-backed conversation agent and supplied order/ticket tools, then Gnani TTS:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_voice_batch.py `
  --input-dir .\audio\incoming `
  --output-dir .\outputs\voice_batch `
  --language en
```

The runner currently accepts uncompressed PCM WAV files (`.wav`), including
8-, 16-, 24-, or 32-bit mono/stereo audio. It converts them to 16 kHz mono
signed 16-bit PCM for STT; non-16-kHz sample rates use SciPy's resampler, and
SciPy is not installed automatically by the runner. Unsupported files receive
an error status record rather than being silently ignored. Files are scanned
recursively and sorted by relative path.

Every audio file starts a **fresh, independent conversation**. The runner does
not infer multi-turn grouping from names, directories, or file order; this
version has no metadata-based session grouping.

Each invocation creates a new timestamped batch directory beneath
`--output-dir`; existing run output is not overwritten. Each input gets its
own uniquely named subdirectory containing:

```text
<batch-run>/
  <input-name>_<relative-path-hash>/
    transcript.txt
    response.txt
    reply.wav
    retrieved_chunks.json
    status.json
```

`retrieved_chunks.json` contains the actual text, score, and provenance metadata
returned by the RAG retrieval used for that answer; it is empty when the turn
did not retrieve evidence. `status.json` records success or the error stage and
message, plus decision, evidence IDs, ticket status, and handoff status on
success. A failed input is recorded and later files are still attempted.

This is a **live** command: each valid audio file may make one Gnani STT
request, one or more RAG/LLM requests as needed, and a Gnani TTS request.
There are no automatic retries. The existing synthetic ticket tool can write
to `tools/tickets_created.csv` when the agent's workflow requires it. Review
the inputs and API costs before running; the mocked offline tests do not call
Gnani, Ollama, or the supplied ticket tool:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_voice_batch.py" -v
```
