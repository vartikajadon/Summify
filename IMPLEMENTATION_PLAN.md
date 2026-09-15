. # Lucidoc — Implementation Plan

**Purpose of this file:** a step-by-step build plan you can hand to an agentic coding tool (Antigravity, Codex, Claude Code, etc.) one step at a time. Each step is scoped to be buildable and testable in a single agent session — small enough to review, big enough to be useful.

**How to use this document:**
1. Work through the steps **in order** — each one depends on the ones before it.
2. Start a fresh agent session per step (or per sub-step if a step feels large).
3. Copy the **"Agent Prompt"** block for that step into your tool of choice.
4. After the agent finishes, check the **"Definition of Done"** list before moving on.
5. Commit to git after each step. Small commits = easy rollback if a step goes sideways.

This plan builds **Phase 1 (MVP)** in full (Steps 0–12), then gives condensed plans for **Phase 2** (code-aware ingestion) and **Phase 3** (OCR) at the end, matching the phased roadmap in the PRD.

---

## 0. Before You Start

### 0.1 Decide your environment
- **Recommended starting point: Groq API first.** Validate that the product idea actually works — retrieval quality, document structuring, whether a prompt-based faithfulness check catches bad answers — before installing anything local. See **§0.5** below for exactly how to do this without burning your free-tier limits.
- **Local (bring in once Groq has proven the concept):** Python 3.11+, [Ollama](https://ollama.com) installed, a machine with 8GB+ VRAM (or CPU-only for early steps — just slower). Required regardless of Groq, once you reach Steps 4b/8b — see §0.5.

### 0.2 Pull the base models you'll need locally (once you get here — not needed for Groq-first testing)
```bash
ollama pull qwen3:8b        # primary generation model (or llama3.3:8b)
ollama pull qwen3:1.7b      # base for both fine-tuned classifiers
```

### 0.3 Repo skeleton
Have the agent create this structure in Step 1 — listed here so you know what to expect:

```
lucidoc/
├── app/                      # Streamlit UI (Step 10)
├── pipeline/
│   ├── ingest.py             # Step 2
│   ├── chunk.py              # Step 3
│   ├── classify.py           # Step 4
│   ├── structure.py          # Step 5
│   ├── index.py              # Step 6
│   ├── chat.py                # Step 7
│   └── faithfulness.py       # Step 8
├── graph/
│   └── build_graph.py        # LangGraph wiring, all stages
├── models/
│   ├── classifier/            # fine-tuning scripts + adapters (Step 4b)
│   └── faithfulness/          # fine-tuning scripts + adapters (Step 8b)
├── session/
│   └── manager.py             # Step 9
├── eval/
│   └── metrics.py             # Step 11
├── tests/
├── config.yaml
├── requirements.txt
└── README.md
```

### 0.4 Tech stack reference (from the PRD — keep these consistent across every step)
| Component | Choice |
|---|---|
| Orchestration | LangGraph + LangChain retrieval utilities |
| Primary model | Configurable via `config.yaml` (`provider: groq \| ollama`); confirm the actual current free/open-access model name at `console.groq.com/docs/models` before hardcoding one — see §0.5 |
| Chunk classifier base | Qwen3 1.7B, LoRA fine-tuned via Unsloth |
| Faithfulness checker base | Small encoder or Qwen3 1.7B classification head, LoRA via Unsloth |
| Vector store | Chroma or FAISS, local, session-scoped |
| Document export | `python-docx` / `docx` library, Markdown as canonical format, PDF via conversion |
| App layer | Streamlit (v1) |

---

### 0.5 Validate on Groq First — Without Burning Your Free-Tier Limit

**Why start here:** Groq lets you test whether the actual idea holds up — good retrieval, sensible document structure, a faithfulness check that catches bad answers — using nothing but an API key. No GPU, no model downloads, no local setup. If the core loop doesn't work well on a fast hosted model, it's not going to work better on a local one; find that out for free before installing anything.

**What Groq can and can't stand in for:**
| Can run on Groq | Cannot run on Groq |
|---|---|
| Step 7 — chat generation (primary model) | Step 4b — fine-tuned chunk classifier (needs local LoRA training via Unsloth) |
| Step 8a — prompt-based faithfulness check (primary model) | Step 8b — fine-tuned faithfulness model (same — needs local training) |
| Steps 1–6, 9–11 (no model calls, or use the primary model only) | Groq's LoRA *inference* exists but is enterprise-tier only, and only supports `llama-3.1-8b-instant` as a base — not useful for our two custom classifiers |

In short: **Steps 1–8a of this plan can be fully built and demoed on Groq alone.** Steps 4b/8b were always a separate, local-only track — treat them as a later quality upgrade, not a blocker (see the working-order note at the end of this document).

**Model name caveat — check before you hardcode anything:**
The PRD's tech stack section names "Qwen3-8B or Llama 3.3-8B." Two problems: Llama 3.3 was never released as an 8B model (only 70B), and Groq's currently-listed free/pay-as-you-go models don't match either name — availability shifts over time. **Before Step 1, check `https://console.groq.com/docs/models` for what's actually open access right now.** As of this writing, `openai/gpt-oss-20b` is fast, cheap, and open to standard accounts — a safe default. Put the model name in `config.yaml`, never hardcoded in a prompt to the agent, so a future change is a one-line edit.

**Free-tier limits you're working within:** no credit card required, every model available, but rate-limited — roughly 30 requests/minute and a low-thousands-per-minute token budget, with a daily request cap. That's enough for iterative development if you're deliberate about it, and not enough if you re-run large batches carelessly. The practices below keep you inside that budget:

1. **Build the LLM client wrapper once, in Step 1, with three things baked in from the start** (not bolted on later):
   - **Response caching** — hash the (prompt, model, params) tuple and cache the response to disk (`.cache/llm/`). Re-running a test or restarting a debug session should never re-call the API for a prompt you've already sent.
   - **A mock/dry-run mode** — an env var or config flag that returns canned/fake responses instead of calling Groq at all. Use this for graph-wiring tests (Step 1's smoke test), unit tests, and anything checking control flow rather than model output.
   - **Rate-limit-aware retry with exponential backoff** — catch 429s and back off, rather than hammering the API and burning your daily cap on failed retries.
2. **Keep test fixtures small on purpose.** 2–4 short sample documents is enough to validate ingestion, chunking, structuring, and retrieval logic. Save large batches (the ≤20-file, ~50k-word target from the PRD's performance metric) for a handful of deliberate full-scale runs, not every test iteration.
3. **Set conservative `max_tokens` on every generation call** during development (config.yaml, not per-call) — verbose test answers cost the same rate-limit budget as concise ones and add up fast.
4. **Batch where the pipeline allows it** — e.g. classify or evaluate several chunks per call with structured output, rather than one API call per chunk, when the step's design allows it (Step 4a already uses structured JSON output — extend the batch size instead of the call count where practical).
5. **Run the eval harness (Step 11) small-scale first.** Point it at a handful of fixture docs and a short question set to sanity-check it works, before pointing it at anything resembling the full metric-validation run.
6. **When you do bring in Ollama (Steps 4b/8b, and Step 12's portability pass), the same `config.yaml` `provider: ollama | groq` switch means nothing about the pipeline code changes** — only which provider your day-to-day testing calls should switch once local models are running, so you stop spending Groq budget on repeat runs you can now do locally for free.

---

## Phase 1 — MVP (Steps 1–12)

---

### Step 1 — Project Scaffolding

**Goal:** a running, empty pipeline skeleton with config, logging, a LangGraph stub that does nothing yet but proves the graph wiring works, and a shared LLM client wrapper that keeps every later step inside Groq's free-tier limits by default.

**Deliverables:**
- Repo structure from §0.3
- `config.yaml` (provider: groq | ollama; model name — confirm the actual current name at `console.groq.com/docs/models` per §0.5, don't assume one; paths; chunk size defaults; thresholds; `max_tokens` default)
- `llm/client.py`: a single shared LLM client wrapper used by every later step (classify, structure dedup, chat, faithfulness), with:
  - Disk-backed response caching keyed on a hash of (prompt, model, params) — never re-calls the API for a prompt already seen
  - A `LUCIDOC_MOCK_LLM=1` dry-run mode returning canned responses, no network call
  - Exponential backoff + retry on 429/rate-limit errors, with a hard cap on retries so a stuck loop can't silently exhaust the daily request budget
- `graph/build_graph.py` with 5 empty passthrough nodes wired in sequence (ingest → chunk_classify → structure_export → index → chat)
- Basic pytest setup with one smoke test that runs the empty graph end-to-end using the mock LLM mode (no real API calls in CI/local test runs)

**Definition of Done:**
- [ ] `pytest` passes on the smoke test with `LUCIDOC_MOCK_LLM=1` and makes zero network calls
- [ ] Running the graph on a dummy input returns a state object that passed through all 5 nodes untouched
- [ ] Calling `llm/client.py` twice with the identical prompt hits the cache on the second call (verify via a test that mocks/counts network calls)
- [ ] `requirements.txt` pins: `langgraph`, `langchain`, `chromadb`, `python-docx`, `streamlit`, `groq`, `ollama`, `pytest`

**Agent Prompt:**
```
Set up a new Python project called "lucidoc" implementing a document-structuring
and RAG-chat pipeline. Create this structure: [paste §0.3 tree]. Use LangGraph to
define a state graph with 5 nodes: ingest, chunk_classify, structure_export, index,
chat — each currently a passthrough function that just logs its name and returns
state unchanged. Define a shared state schema (TypedDict or Pydantic) with fields:
uploaded_files, raw_text, chunks, classified_chunks, structured_document,
vector_index_ref, chat_history. Add a config.yaml with a `provider` field
(groq | ollama, default groq), a `model` field (leave a placeholder value and a
comment telling the developer to confirm the current model name at
console.groq.com/docs/models before running anything), plus chunk size, thresholds,
and a `max_tokens` default for generation calls.

Also build llm/client.py: a single wrapper class other modules will use for every
LLM call in this project (no module should call the Groq/Ollama SDK directly).
It must support: (1) disk-backed response caching in .cache/llm/, keyed on a hash
of the exact prompt + model + parameters, so an identical call is never repeated
against the API; (2) a dry-run mode toggled by the LUCIDOC_MOCK_LLM=1 environment
variable that returns a deterministic canned response with no network call at all;
(3) automatic retry with exponential backoff specifically on rate-limit (429)
errors, with a hard maximum retry count so a bug can't silently burn through a
daily request quota. Read provider/model/max_tokens from config.yaml.

Add pytest tests: a smoke test that runs the graph on a dummy state with
LUCIDOC_MOCK_LLM=1 set and asserts it passes through all 5 nodes and makes zero
real network calls; a test that calls llm/client.py twice with an identical
prompt and asserts the underlying API call happens only once (mock/count the
network layer).

Pin dependencies in requirements.txt: langgraph, langchain, chromadb, python-docx,
streamlit, groq, ollama-python, pytest.
```

---

### Step 2 — Ingestion (FR-1.1, FR-1.2, FR-1.3)

**Goal:** accept a batch of files in `.md`, `.txt`, `.docx`, and text-extractable `.pdf`; normalize to plain text with source metadata; don't let one bad file kill the batch.

**Deliverables:**
- `pipeline/ingest.py`:
  - `ingest_files(file_paths: list[str]) -> list[IngestedDoc]`
  - Per-format parsers (`.md`/`.txt` = read directly, `.docx` via `python-docx`, `.pdf` via `pypdf` or `pdfplumber` — text layer only, no OCR yet)
  - Preserve: source filename, heading hierarchy (where present), file order
  - Corrupt/unsupported file → logged and skipped, batch continues
- Unit tests with sample fixtures (one good file per format, one intentionally corrupt file)

**Definition of Done:**
- [ ] A batch with a mix of valid files + one corrupt file processes successfully, corrupt file is flagged in the return value (not silently dropped, not fatal)
- [ ] Each `IngestedDoc` retains `source_filename` and any detected heading structure
- [ ] Wire this into the `ingest` node in the graph from Step 1

**Agent Prompt:**
```
Implement pipeline/ingest.py for the lucidoc project (see existing graph/build_graph.py
for the state schema). Requirements:
- Accept a list of file paths in .md, .txt, .docx, and text-extractable .pdf.
- Normalize each to plain text, preserving source filename and heading hierarchy
  (from markdown headers, docx heading styles, or PDF font-size heuristics) as metadata.
- If a file is corrupt or an unsupported format, log a warning, mark it as failed
  in the return value, and continue processing the rest of the batch — never raise
  and kill the whole batch.
- Return a list of IngestedDoc objects: {source_filename, text, headings: list, status}.
- Use python-docx for .docx and pdfplumber for .pdf (text layer only — no OCR).
- Wire this into the `ingest` node of the LangGraph pipeline, replacing the passthrough.
- Add pytest tests using small fixture files for each format, plus one deliberately
  corrupt file to verify graceful failure.
```

---

### Step 3 — Chunking (FR-2.1)

**Goal:** split normalized text into semantically coherent chunks, preferring natural boundaries over fixed windows.

**Deliverables:**
- `pipeline/chunk.py`:
  - `chunk_text(doc: IngestedDoc) -> list[Chunk]`
  - Boundary-aware splitting: split on headings/paragraphs first; only fall back to a fixed-length window (with overlap) when a section is too long
  - Each `Chunk` retains: source filename, position/order, and the heading path it fell under
- Unit tests: a doc with clear headings splits along them; a doc with one giant unstructured blob still produces reasonably sized chunks

**Definition of Done:**
- [ ] No chunk exceeds the configured max token/char length
- [ ] Chunks preserve traceability back to source file + heading
- [ ] Wired into the `chunk_classify` node (chunking half only — classification is Step 4)

**Agent Prompt:**
```
Implement pipeline/chunk.py for lucidoc. Input: IngestedDoc objects from ingest.py
(text + heading metadata). Output: a list of Chunk objects, each with
{text, source_filename, heading_path, chunk_index}. Splitting strategy:
prefer natural boundaries (markdown/docx headings, then paragraph breaks) over
fixed-length windows. Only apply a fixed-length sliding window with ~10-15% overlap
when a single section exceeds the configured max_chunk_tokens (read from config.yaml).
Add pytest tests: one fixture with clear heading structure (verify chunks align to
headings), one fixture that's one long unstructured paragraph (verify it still
produces bounded chunks). Wire the chunking half into the chunk_classify graph node,
storing results in state.chunks — leave classification for the next step.
```

---

### Step 4 — Chunk Classification (FR-2.2, FR-2.3, FR-2.4 deferred to Phase 2)

Split into two sub-steps so you get a working pipeline fast, then upgrade it.

#### Step 4a — Zero-shot classifier (fallback path, build first)

**Goal:** classify each chunk via a prompt to the primary LLM — this ships first and also becomes the permanent fallback (FR-2.3).

**Deliverables:**
- `pipeline/classify.py`:
  - `classify_chunk_zeroshot(chunk: Chunk, llm) -> ClassifiedChunk`
  - Label taxonomy: `concept, setup/install, api-reference, example, troubleshooting, config, faq, other`
  - Structured output (JSON mode / function-calling) so labels are always one of the taxonomy values
- Wired fully into the `chunk_classify` node

**Definition of Done:**
- [ ] Every chunk gets exactly one valid label
- [ ] Malformed/unexpected model output falls back to `other` rather than crashing
- [ ] Full pipeline (Steps 1–4a) runs end-to-end on a real sample doc set

**Agent Prompt:**
```
Implement the zero-shot path of pipeline/classify.py for lucidoc. Given a Chunk
(from chunk.py) and an LLM client (Ollama, model configurable via config.yaml,
default qwen3:8b), classify it into exactly one of: concept, setup/install,
api-reference, example, troubleshooting, config, faq, other. Use structured/JSON
output so the label is always constrained to this set — if the model returns
anything else or fails to parse, default to "other" and log a warning (never crash
the pipeline on one bad classification). Return ClassifiedChunk = Chunk + {label,
confidence: null for now, method: "zero-shot"}. Wire this into the chunk_classify
graph node so the full pipeline (ingest → chunk → classify) runs end-to-end.
Add a pytest test with a handful of example chunks with obvious expected labels
(e.g. a chunk starting "## Installation" should classify as setup/install) and
assert reasonable accuracy on that small set.
```

#### Step 4b — Fine-tuned classifier (upgrade path)

**Goal:** replace the zero-shot call as the primary path with a small fine-tuned model; keep 4a as the fallback (FR-2.3) when confidence is low or the fine-tuned model is unavailable.

**Deliverables:**
- `models/classifier/build_dataset.py`: uses the larger model (Qwen3-8B) to auto-label a batch of chunks from real/sample docs
- A manual review step (script or notebook) to correct a sample of a few hundred labels
- `models/classifier/train.py`: LoRA fine-tune of Qwen3 1.7B via Unsloth on the corrected set
- `models/classifier/eval.py`: held-out set, per-class precision/recall/F1, overall agreement %
- `pipeline/classify.py` updated: try fine-tuned model first, confidence threshold (config.yaml) triggers fallback to 4a's zero-shot path

**Definition of Done:**
- [ ] Eval script reports ≥ 90% overall agreement with human labels on the held-out set (PRD target — Section 3.2)
- [ ] Low-confidence predictions correctly route to the zero-shot fallback (test with a mocked low-confidence case)
- [ ] `classify.py`'s public interface is unchanged from 4a's, so nothing else in the pipeline needs to change

**Agent Prompt:**
```
Build the fine-tuning workflow for lucidoc's chunk classifier:
1. models/classifier/build_dataset.py: takes a directory of sample docs, runs them
   through the existing ingest -> chunk pipeline, then auto-labels each chunk using
   Qwen3-8B via Ollama with the same taxonomy as the zero-shot classifier
   (concept, setup/install, api-reference, example, troubleshooting, config, faq,
   other). Output a labeled dataset as JSONL.
2. Write a small review script/notebook that samples ~300 auto-labels for manual
   correction and saves a corrected JSONL.
3. models/classifier/train.py: LoRA fine-tune of Qwen3-1.7B on the corrected set
   using Unsloth. Save the adapter to models/classifier/adapter/.
4. models/classifier/eval.py: evaluate on a held-out split — report per-class
   precision/recall/F1 and overall agreement %. Target ≥90% overall agreement.
5. Update pipeline/classify.py: add classify_chunk_finetuned() using the trained
   adapter. The main classify_chunk() function should try the fine-tuned model
   first; if its confidence score is below a configurable threshold (config.yaml)
   or the adapter fails to load, fall back to the existing zero-shot function.
   Keep the public function signature identical so callers don't need to change.
Add tests mocking both the high-confidence and low-confidence-fallback paths.
```

---

### Step 5 — Document Structuring & Export (FR-3.1–FR-3.5)

**Goal:** assemble classified chunks into one document with headings grouped by type, a table of contents, deduplication, and export to Markdown/.docx/PDF — with every section traceable back to its source file.

**Deliverables:**
- `pipeline/structure.py`:
  - `assemble_document(classified_chunks) -> StructuredDocument`: group by label, build heading hierarchy, order groups sensibly (e.g. concept → setup → config → api-reference → example → troubleshooting → faq → other)
  - `generate_toc(doc) -> TOC`
  - `deduplicate(classified_chunks) -> (deduped_chunks, flagged_duplicates)`: embedding-similarity based near-duplicate detection (reuse the embedding model from Step 6, or a lightweight one here)
  - `export_markdown`, `export_docx`, `export_pdf`
  - Each rendered section carries a footer/marginal note listing its source file(s) (FR-3.5)

**Definition of Done:**
- [ ] Running on a sample multi-file input produces one document with clearly grouped sections and a working TOC
- [ ] A deliberately duplicated paragraph across two input files is detected and merged/flagged, not duplicated in the output
- [ ] All three export formats open correctly and match content
- [ ] Manually check: each section can be traced back to its source file(s)

**Agent Prompt:**
```
Implement pipeline/structure.py for lucidoc. Input: list of ClassifiedChunk from
classify.py. Steps:
1. deduplicate(): detect near-duplicate chunks across different source files using
   embedding cosine similarity (threshold in config.yaml); merge duplicates into one,
   keeping a list of all contributing source files, and log/flag what was merged.
2. assemble_document(): group deduplicated chunks by label into sections, using this
   default group order: concept, setup/install, config, api-reference, example,
   troubleshooting, faq, other. Build a heading hierarchy (H1 per group, H2 per
   distinct sub-topic or source file within a group as appropriate).
3. generate_toc(): produce a table of contents matching the assembled structure.
4. Each rendered section must include a visible trace back to its originating
   source file(s) (e.g. a small "Source: filename.md" note).
5. Export functions: export_markdown(doc) -> str, export_docx(doc) -> bytes (use
   python-docx, include a real Word TOC field and heading styles),
   export_pdf(doc) -> bytes (convert from the docx or markdown via an existing
   converter, e.g. weasyprint or a LibreOffice headless call).
Wire deduplicate + assemble_document + generate_toc into the structure_export graph
node. Add tests: verify grouping order, verify a duplicated paragraph across two
fixture files appears once in output with both sources listed, verify all three
export functions produce valid non-empty output.
```

---

### Step 6 — Indexing & Retrieval (FR-4.1–FR-4.3)

**Goal:** embed chunks into a session-scoped vector store with zero cross-session leakage, and support re-indexing when files change.

**Deliverables:**
- `pipeline/index.py`:
  - `build_index(session_id, chunks) -> VectorIndexRef` using Chroma (or FAISS) with a **collection per session_id**
  - `add_to_index` / `remove_from_index` for incremental updates
  - `query_index(session_id, query, k) -> list[Chunk]`
- A test that creates two sessions and asserts a query in session A never returns chunks from session B

**Definition of Done:**
- [ ] Cross-session isolation test passes
- [ ] Adding/removing a file and re-querying reflects the change without a full rebuild being required (or a full rebuild is fast enough to be a non-issue — document which approach was taken)
- [ ] Wired into the `index` graph node

**Agent Prompt:**
```
Implement pipeline/index.py for lucidoc using Chroma (local, persistent client)
as the vector store. Requirements:
- build_index(session_id, chunks): creates or updates a Chroma collection scoped
  to session_id (collection name derived from session_id, never shared across
  sessions). Embed using a local embedding model (e.g. nomic-embed-text via Ollama,
  configurable in config.yaml).
- add_to_index(session_id, new_chunks) and remove_from_index(session_id, chunk_ids)
  for incremental updates when a user adds/removes files from an existing session
  (FR-4.3), without requiring a full rebuild.
- query_index(session_id, query, k=5): returns top-k relevant chunks with scores.
- Write a test that builds indexes for two different session_ids with different
  content, then asserts that querying session A's index never returns any chunk
  whose session_id metadata belongs to session B — hard-fail if any leakage is found.
Wire build_index into the `index` graph node using session_id from the graph state.
```

---

### Step 7 — Conversational Q&A (FR-5.1, FR-5.2 — verification comes in Step 8)

**Goal:** retrieve relevant chunks, generate a grounded answer, cite sources. No faithfulness check yet — that's Step 8, deliberately separated so you can test retrieval and generation in isolation first.

**Deliverables:**
- `pipeline/chat.py`:
  - `answer_question(session_id, question, history) -> DraftAnswer` (retrieve → generate, with citations)
  - Prompt template that instructs the model to answer **only** from provided context and to cite source file/section per claim
- Wired into the `chat` graph node

**Definition of Done:**
- [ ] Asking a question covered by the sample docs returns an answer with correct citations
- [ ] Asking a question NOT covered by the docs still gets a response (unsupported responses are handled properly starting Step 8 — for now, just confirm the model attempts an answer and citations are present or explicitly absent)

**Agent Prompt:**
```
Implement pipeline/chat.py for lucidoc (RAG generation step, without the
faithfulness check yet — that's a separate step). Requirements:
- answer_question(session_id, question, chat_history): calls query_index() from
  index.py to retrieve the top-k relevant chunks for the question, then generates
  an answer using the primary LLM (Ollama, model from config.yaml).
- The prompt must instruct the model to answer ONLY using the provided retrieved
  context, and to cite the source file/section for each claim it makes.
- Return a DraftAnswer object: {answer_text, citations: list[{source_file, chunk_id}],
  retrieved_context}.
- Support multi-turn chat_history being included in the prompt for follow-up questions.
Wire into the `chat` graph node. Add tests using a fixture index with known content:
assert a question directly answerable from the fixture gets a correctly-cited answer.
```

---

### Step 8 — Faithfulness / Groundedness Checker (FR-5.3, FR-5.4)

This is the trust-critical piece of the product — split it the same way as Step 4.

#### Step 8a — Rule/prompt-based checker (build first)

**Goal:** a working "second opinion" check using the primary LLM in a separate prompt, wired into the response path before anything is fine-tuned.

**Deliverables:**
- `pipeline/faithfulness.py`:
  - `check_answer(answer, retrieved_context, llm) -> FaithfulnessResult` (`supported` / `unsupported` + reasoning)
  - On `unsupported`: trigger one regeneration attempt with a stricter "only state what's directly in the text" prompt; if still unsupported, return the explicit "not covered in the provided material" response
- Wired into the `chat` node, after `answer_question`

**Definition of Done:**
- [ ] A deliberately fabricated answer (test fixture: answer contradicts the source) is correctly flagged
- [ ] A well-grounded answer passes through unflagged
- [ ] The "not covered" fallback message is returned verbatim (not paraphrased differently each time) when both attempts fail

**Agent Prompt:**
```
Implement the prompt-based version of pipeline/faithfulness.py for lucidoc.
check_answer(answer_text, retrieved_context, llm) sends the (answer, context) pair
to the primary LLM with a prompt asking it to judge strictly whether every claim in
the answer is directly supported by the context, returning a structured
{verdict: "supported"|"unsupported", reasoning: str}.
Wire this into the chat graph node, after answer_question(): if verdict is
"unsupported", regenerate the answer once with a stricter prompt ("state only
facts explicitly present in the context, do not infer or add detail") and check
again. If still unsupported after the retry, return a fixed response: "This isn't
covered in the material you provided." — never fall through to showing an
unsupported answer.
Add tests using fixtures: one answer that clearly contradicts/invents beyond its
context (should be flagged unsupported), one answer that's clearly grounded
(should pass). Also test that two consecutive unsupported verdicts correctly
produce the fixed fallback message rather than a third regeneration attempt.
```

#### Step 8b — Fine-tuned faithfulness model (upgrade path)

**Goal:** replace the prompt-based judge with a small trained classifier for speed/consistency, following the PRD's synthetic-data approach.

**Deliverables:**
- `models/faithfulness/build_dataset.py`: positive examples = answers entailed by their context (reuse Step 7 outputs on fixture docs); negative examples via controlled perturbation — contradict a fact, swap in unrelated context, remove the supporting sentence
- `models/faithfulness/train.py`: LoRA fine-tune (Qwen3 1.7B classification head, or a small encoder) via Unsloth
- `models/faithfulness/eval.py`: precision/recall on held-out set, **recall on unsupported answers is the metric to optimize** (target ≥ 90% per PRD)
- `pipeline/faithfulness.py` updated to try the fine-tuned model first, same fallback pattern as Step 4b

**Definition of Done:**
- [ ] Eval script reports ≥ 90% recall on unsupported answers (PRD target — bias the decision threshold toward recall, per the PRD's risk mitigation)
- [ ] False-positive rate on correctly-grounded answers is ≤ 10% (PRD target)
- [ ] Public interface of `check_answer()` unchanged from 8a

**Agent Prompt:**
```
Build the fine-tuning workflow for lucidoc's faithfulness checker:
1. models/faithfulness/build_dataset.py: generate positive examples (answer,
   context) pairs where the answer is genuinely entailed by the context (reuse
   good outputs from the Step 7/8a pipeline on sample docs). Generate negative
   examples via three perturbation strategies applied to positive examples:
   (a) contradict a fact in the answer, (b) swap in an unrelated context chunk,
   (c) remove the specific sentence that supports the answer's key claim.
   Output a labeled JSONL dataset, balanced across strategies.
2. models/faithfulness/train.py: LoRA fine-tune of Qwen3-1.7B (or a small encoder
   model) via Unsloth as a binary classifier (supported/unsupported) on the dataset.
3. models/faithfulness/eval.py: evaluate on a held-out split. Report precision and
   recall for the "unsupported" class specifically — this is the target metric,
   aim for ≥90% recall on unsupported answers, and bias the decision threshold
   toward recall over precision (a missed hallucination is worse than a false
   alarm, per product requirements). Also report false-positive rate on correctly-
   grounded answers, target ≤10%.
4. Update pipeline/faithfulness.py: check_answer() should try the fine-tuned model
   first; on load failure or a low-confidence prediction, fall back to the existing
   prompt-based check_answer from Step 8a. Keep the function signature identical.
Add tests for both the fine-tuned and fallback paths.
```

---

### Step 9 — Session & Data Management (FR-6.1, FR-6.2)

**Goal:** users can delete everything tied to a session, and download all generated artifacts before doing so.

**Deliverables:**
- `session/manager.py`:
  - `create_session() -> session_id`
  - `delete_session(session_id)`: removes uploaded files, generated document, and vector index (calls into `index.py`'s per-session collection deletion)
  - `export_all(session_id) -> dict[str, bytes]`: bundles the structured document exports (md/docx/pdf) for download
- Storage layout: one directory per session under a configurable data root, so deletion is a straightforward directory removal + Chroma collection drop

**Definition of Done:**
- [ ] After `delete_session`, no files remain on disk for that session and its vector collection no longer exists
- [ ] `export_all` returns all three export formats correctly before deletion

**Agent Prompt:**
```
Implement session/manager.py for lucidoc. Requirements:
- create_session(): generates a UUID session_id, creates a per-session directory
  under the configured data root (config.yaml) for uploaded files and generated
  documents.
- delete_session(session_id): deletes the session's directory (uploaded files +
  generated documents) AND drops its Chroma collection via index.py. Must fully
  remove all traces — verify with a test that lists the data root and the Chroma
  client afterward and finds nothing for that session_id.
- export_all(session_id): returns the markdown, docx, and pdf exports (from
  structure.py) as a dict of {format: bytes}, for the user to download before
  deleting.
Add tests: full lifecycle test — create session, add fixture data, export_all
succeeds and returns non-empty content for all three formats, delete_session,
then assert no files or vector data remain.
```

---

### Step 10 — Streamlit UI

**Goal:** wire everything into a usable web UI — the first point where a human actually interacts with the whole system.

**Deliverables:**
- `app/main.py`:
  - File upload (multi-file)
  - "Generate document" button → runs the full graph through Step 5, shows the structured document + download buttons (md/docx/pdf)
  - Chat panel → runs Step 7+8, shows answer with citations, or the explicit "not covered" message
  - "Delete my session" button, wired to Step 9
  - Basic progress indicators (this pipeline can take up to 2 minutes per PRD's performance target — don't leave the user staring at a blank screen)

**Definition of Done:**
- [ ] Manual end-to-end walkthrough: upload sample files → get structured doc → ask a question → get a cited answer → ask an out-of-scope question → get the explicit refusal → delete session → confirm data is gone
- [ ] UI never crashes on a bad file upload (surfaces the Step 2 per-file failure gracefully)

**Agent Prompt:**
```
Build app/main.py, a Streamlit UI for lucidoc, wiring together the full pipeline.
Layout:
1. A multi-file uploader accepting .md, .txt, .docx, .pdf.
2. A "Generate structured document" button that runs the full LangGraph pipeline
   (ingest -> chunk_classify -> structure_export -> index) using a session_id
   created via session/manager.py. Show a progress spinner/status text per stage
   (this can take up to ~2 minutes for larger batches). On completion, render the
   structured document (at least its TOC and first section) and provide download
   buttons for the Markdown, .docx, and PDF exports.
3. A chat panel below: a text input for questions, calling pipeline/chat.py +
   pipeline/faithfulness.py, showing the answer with its citations, or the fixed
   "not covered in the material you provided" message when unsupported. Maintain
   chat history across turns within the session.
4. A "Delete my session" button in the sidebar that calls session/manager.py's
   delete_session() and resets the UI state, with a confirmation step.
5. Handle per-file ingestion failures gracefully — show which files failed and why,
   without blocking the rest of the batch or crashing the app.
Keep this as a single-file Streamlit app calling into the existing pipeline modules
— no business logic should live in app/main.py itself.
```

---

### Step 11 — Evaluation Harness (validates PRD Section 3.2 targets)

**Goal:** one script that reports all six success metrics from the PRD against a held-out test set, so you can tell if the MVP actually hits its targets.

**Deliverables:**
- `eval/metrics.py` reporting:
  1. Chunk-type classification accuracy (≥ 90% target)
  2. Faithfulness checker recall on unsupported answers (≥ 90% target)
  3. Faithfulness checker false-positive rate (≤ 10% target)
  4. End-to-end structuring quality (needs a manual-review rubric + sample; ≥ 80% target)
  5. Time-to-first-structured-document (< 2 min for ≤ 20 files)
  6. Chatbot answer relevance (≥ 85% target — needs manual/LLM-judged rubric)

**Definition of Done:**
- [ ] Running `python eval/metrics.py` against the test fixture set prints all six metrics in one report
- [ ] Report clearly flags which metrics are below target

**Agent Prompt:**
```
Build eval/metrics.py for lucidoc, a single script that runs the full pipeline
against a held-out test set (a fixtures directory of sample docs + a labeled
question set) and reports these six metrics, matching PRD targets:
1. Chunk-type classification accuracy vs human labels (target >=90%) — reuse
   models/classifier/eval.py logic if applicable.
2. Faithfulness checker recall on unsupported answers (target >=90%) — reuse
   models/faithfulness/eval.py logic.
3. Faithfulness checker false-positive rate on grounded answers (target <=10%).
4. End-to-end structuring quality: for a sample of generated documents, use an
   LLM-as-judge rubric prompt (or hooks for manual review) rating whether each
   section is "accurately organized" — report % rated accurate (target >=80%).
5. Time-to-first-structured-document: measure wall-clock time for the full
   pipeline on batches of <=20 files, report against the <2 minute target.
6. Chatbot answer relevance: for a labeled question set, use an LLM-as-judge
   rubric to rate whether each answer is relevant and correctly scoped
   (target >=85%).
Print a clean summary table at the end showing each metric, its target, its
measured value, and PASS/FAIL. Make this runnable via `python eval/metrics.py
--fixtures path/to/dir`.
```

---

### Step 12 — Packaging & Local/Hosted Portability (NFR: Cost, Portability)

**Goal:** you built and validated Steps 1–11 against Groq (per §0.5). This step adds the local Ollama backend to `llm/client.py` (only the mock and Groq paths existed before now), confirms the whole thing also runs with zero mandatory paid dependency, and makes switching between the two a config change, not a code change — so day-to-day testing can move off Groq's free-tier budget once local models are available, while Groq remains available as a fast option going forward.

**Deliverables:**
- `llm/client.py` extended with an Ollama backend alongside the existing Groq and mock paths from Step 1
- `config.yaml`'s `provider: groq | ollama` switch now fully functional both ways, with the caching/backoff/mock wrapper from Step 1 applying to both providers identically
- `README.md`: setup instructions, how to run locally, how to switch providers, hardware requirements (per PRD: consumer GPU with 8GB+ VRAM, or free-tier hosted endpoint), and a short note that the project was built and validated against Groq first, with Ollama added for local/offline use and heavier iterative testing
- A `Dockerfile` (optional but recommended) for reproducible local deployment

**Definition of Done:**
- [ ] Fresh clone + `pip install -r requirements.txt` + `ollama pull` + `provider: ollama` in `config.yaml` + `streamlit run app/main.py` works with no paid API key and no Groq account
- [ ] `provider: groq` continues to work unchanged (nothing regressed from Steps 1–11)
- [ ] Switching providers is confirmed to require zero code changes — config only
- [ ] README documents both paths and the caching/mock-mode behavior from Step 1, so budget-conscious testing habits carry forward for whoever works on this next

**Agent Prompt:**
```
Extend lucidoc's llm/client.py (built in Step 1 with Groq + mock support) to add
a local Ollama backend, so config.yaml's `provider: groq | ollama` field is now
fully functional both ways. The existing response caching, dry-run mock mode, and
rate-limit backoff behavior from Step 1 must apply identically regardless of
provider — don't duplicate that logic per-backend. Every place the primary LLM is
called (chat.py, classify.py fallback, faithfulness.py fallback) must already be
reading from this shared client, so no other module should need to change.

Write README.md covering: prerequisites, how to install dependencies, how to get
a Groq API key vs. how to pull required Ollama models locally, how to switch
providers via config.yaml, minimum hardware for local use (consumer GPU with
8GB+ VRAM, or a free-tier hosted inference endpoint as a no-GPU alternative), and
a short section explaining the project's dev workflow: built and validated against
Groq's free tier first (fast to start, rate-limited), with local Ollama available
for unlimited-iteration testing and for the fine-tuned models from Steps 4b/8b
(which only ever ran locally). Mention the LUCIDOC_MOCK_LLM=1 dry-run mode as the
right choice for anyone just testing pipeline wiring without burning either
provider's budget.

Write a Dockerfile that installs dependencies and runs the Streamlit app,
documenting that Ollama itself should run as a separate container/service when
provider=ollama.

Do a final pass confirming there is no code path that requires a paid API key to
function end-to-end, and that both provider values pass the existing test suite
(run the suite once per provider, or with LUCIDOC_MOCK_LLM=1 for a provider-agnostic
pass).
```

---

## Phase 2 — Code-Aware Ingestion (FR-1.4, FR-2.4)

Builds on Phase 1. Two focused steps:

**Step 13 — Code ingestion.** Extend `pipeline/ingest.py` to accept source repositories/files, extracting docstrings, comments, and signatures as a distinct content stream (separate from prose docs, but flowing through the same chunk → classify → structure pipeline).

**Step 14 — Code-example label.** Extend the classification taxonomy (both zero-shot prompt and fine-tuned model — requires re-running the Step 4b dataset-build + fine-tune with the new label added) with `code-example`, and update `structure.py`'s grouping order to place it sensibly alongside `example`.

```
Agent Prompt (Step 13):
Extend pipeline/ingest.py to accept a source code repository path. Walk the repo,
extract docstrings, comments, and function/class signatures per file, tagging each
extracted piece with its source file and a `content_stream: "code"` marker
(distinct from `content_stream: "prose"` for existing formats). These flow through
the same chunk.py -> classify.py -> structure.py pipeline as before. Add tests with
a small fixture repo (a couple of Python files with docstrings and comments).

Agent Prompt (Step 14):
Add a "code-example" label to the classification taxonomy across zero-shot
(classify.py) and the fine-tuned model (rebuild the dataset with models/classifier/
build_dataset.py including code chunks from Step 13, re-train via train.py,
re-evaluate via eval.py — same >=90% agreement target). Update structure.py's
default section ordering to place code-example alongside example. Add tests
confirming code chunks are correctly labeled and grouped.
```

---

## Phase 3 — Scanned Document Support (FR-1.5)

**Step 15 — OCR ingestion.** Extend `pipeline/ingest.py`'s PDF handler: detect scanned/image-based PDFs (no extractable text layer), run OCR (e.g. Tesseract) as the primary path, and fall back to a vision-capable model for low-quality scans where OCR confidence is low.

```
Agent Prompt (Step 15):
Extend pipeline/ingest.py's PDF handling to detect scanned/image-based PDFs
(pages with no extractable text layer). For these, run OCR (Tesseract via
pytesseract) page by page. When OCR confidence is low (configurable threshold)
or output looks garbled (e.g. high proportion of non-dictionary tokens), fall back
to a vision-capable model (e.g. a local vision-language model via Ollama) to
transcribe the page instead. Tag resulting IngestedDoc chunks with
`extraction_method: "ocr" | "vision-fallback"` for traceability. Add tests with a
fixture scanned-image PDF, asserting text is correctly extracted and the fallback
triggers on a deliberately low-quality fixture.
```

---

## Suggested Working Order Summary

| # | Step | Unlocks | Runs on |
|---|---|---|---|
| 1 | Scaffolding (incl. cached/mock-mode LLM client) | Everything else | No model calls in tests (mock mode) |
| 2 | Ingestion | Real content into the pipeline | No model calls |
| 3 | Chunking | Classification | No model calls |
| 4a | Zero-shot classify | First end-to-end run | Groq |
| 5 | Structure & export | First visible deliverable (the document) | Groq (dedup step only) |
| 6 | Indexing | Chat | Local embedding model (no Groq call) |
| 7 | Chat (no verification) | Faithfulness checking | Groq |
| 8a | Faithfulness (prompt-based) | Trustworthy chatbot, MVP-complete core loop | Groq |
| 9 | Session management | Safe multi-user-ready storage | No model calls |
| 10 | Streamlit UI | A demoable product | Groq |
| 11 | Eval harness | Confidence the MVP hits its targets | Groq — start small, see §0.5 |
| 4b, 8b | Fine-tuned models | Quality/speed upgrade, can happen anytime after 11 | Local only (LoRA training) |
| 12 | Packaging | Shippable v1, adds the local Ollama backend | Both |
| 13–14 | Phase 2 | Code-aware docs | Either |
| 15 | Phase 3 | Scanned document support | Either |

**Note on 4b/8b timing:** the zero-shot and prompt-based versions (4a, 8a) are fully functional on their own — they satisfy every functional requirement in the PRD except the specific "fine-tuned model" implementation detail. It's reasonable to ship a demo or even a v1 on 4a+8a alone and treat 4b/8b as a quality upgrade once you have real usage data to build a better training set from. Notably, **Steps 1–11 (the entire demoable MVP) can be built and validated on Groq's free tier alone** — Ollama only becomes necessary at 4b/8b (fine-tuning) or if you want unlimited local iteration in Step 12 onward.
