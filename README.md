# Manufacturing RAG — Built to Be Helpful, Not Just Built
*An end-to-end framework for measuring whether a RAG system actually produces correct, grounded, trustworthy answers — not just plausible-sounding ones.*

---

## The Problem With Most RAG Projects

Building a RAG system is easy. Wiring up a vector database, an embedding model, and an LLM takes an afternoon.

The hard question is: **does it actually help?**

Most RAG demos skip this entirely. They retrieve something, generate something, and call it done. There is no measurement of whether the answer was correct, whether it came from the documents or the model's own memory, or whether the retrieval even found the right content in the first place.

This project is built around that hard question. The pipeline exists to give the evaluation something to measure. The notebooks are the actual work.

---

## What This Project Measures

Quality is evaluated at four levels — each one constraining the next:

```
Is the corpus any good?
        ↓
Are documents chunked and represented correctly?
        ↓
Does retrieval find the right documents?
        ↓
Do the final answers actually contain correct, grounded information?
```

**A system can fail at any of these layers.** Most RAG failures are data problems disguised as model problems. This framework finds where the failure actually lives.

---

## Evaluation Framework (The Core of This Project)

### NB1 — Corpus Quality
Before retrieval is attempted, is the data worth indexing?

Metrics: signal yield, signal density, domain coverage, redundancy (pairwise Jaccard), information entropy. A composite quality score per document.

> Most retrieval failures are data problems in disguise. This notebook catches them before they propagate.

### NB2 — Chunking & Signal Quality
Are documents represented in a way that makes retrieval possible?

Metrics: BoundaryScore (does a chunk start and end at a natural sentence boundary?), TopicConsistency (is one chunk about one thing?), signal specificity and redundancy rates.

> A correct document retrieved as a badly-chunked fragment fails the user just as much as no retrieval at all.

### NB3 — Retrieval Quality
Does the system return the right documents when asked?

Metrics: Recall@k, Precision@k, MRR. Tested across dense, hybrid, reranked, signal-filtered, SQL-filtered, and HyDE configurations against a 27-query human-labeled test set.

| Method | MRR | Recall@10 |
|---|---|---|
| Dense only | 0.399 | 0.403 |
| Hybrid (RRF) | 0.442 | 0.453 |
| Hybrid + Reranker | 0.426 | 0.473 |
| HyDE + Hybrid *(current production config)* | 0.436 | 0.447 |

> Retrieval is measured against a labeled test set built by hand — not a proxy metric. These numbers mean something. Full config comparison (signal-filtered, SQL-filtered, reranker pool-size sweep) is versioned in `notebooks/eval_snapshots.json`.

### NB4 — Answer Quality
Does better retrieval actually produce better answers?

Metrics: FactRecall (did the answer contain the required facts?), Groundedness (is each claim traceable to a retrieved document?), Completeness, Hallucination rate. Evaluated using LLM-as-judge with structured prompts against 15 human-authored test cases with pre-defined key facts, run through the **real production agent graph** (router → retrieval → generation), not a notebook-only shortcut.

| Metric | Baseline | Current |
|---|---|---|
| Fact recall | 0.583 | **0.833** |
| Groundedness | 0.913 | 0.947 |
| Hallucination rate | 0% | 0% |

> This is the final accountability check. A high-retrieval system that generates ungrounded answers is not a helpful system. The jump from baseline to current isn't a tuning artifact — it's a real bug found via trace triage (see "Chat Agent System" below), fixed, and re-measured to confirm it worked.

---

## Chat Agent System

The evaluation framework above measures a conversational, multi-persona agent system (`agents/`, `chat_app.py`) — not a single retrieve-then-generate function.

**Four personas, routed automatically per question**, built on LangGraph:

| Persona | Handles | Grounded in |
|---|---|---|
| `technical_document_agent` | General manufacturing/industry questions | HyDE + hybrid retrieval over the document corpus, with a corrective-RAG web-search fallback |
| `plc_expert` | PLC / Structured Text code questions, best-practices checks | A reference code corpus + `iec-checker` (a real static analyzer, not an LLM opinion) |
| `analytics_agent` | Energy/production telemetry questions, including real-code-computed charts | `pandas`-computed statistics — never an LLM guessing at numbers |
| `direct_reply` | Greetings, meta-conversation, genuinely off-topic questions | Conversation history / general knowledge — skips retrieval entirely for latency |

A **router** (one LLM call per turn) both classifies intent and resolves follow-ups ("does that follow best practices too?") into standalone queries before retrieval ever runs. A `multi_intent` node handles compound questions that span more than one persona, running them concurrently.

**Design choices that matter more than the routing:**

- **Fail-open vs. fail-closed, deliberately different.** The structured-output marker that decides whether to show sources fails *open* (ambiguous → show the source anyway, since hiding a real answer is worse than an occasional over-citation). The chart-generation directive fails *closed* (ambiguous → no chart, since a wrong chart is worse than no chart). Same codebase, two different risk profiles, on purpose.
- **Charts are computed, never generated.** `analytics_agent`'s chart feature works like input validation: a real-code keyword pre-pass decides *whether and what* to chart *before* the LLM ever sees the question. The LLM only ever picks bar-vs-line — it never chooses what data goes in, and the aggregation function (`energy_data_search.compute_grouped_series`) whitelists exact column names and operations, never `eval`/`exec`. The same dict that feeds the chart is what's narrated in prose, so the two can't drift apart.
- **A real bug, found and fixed via trace triage, not guessing.** Two eval queries scored zero on fact-recall. Reading the actual LangSmith trace (not the code) showed the system correctly refusing to discuss a well-known industry protocol because the local corpus had no content on it — a real gap between "the corpus doesn't cover this" and "I don't know this." The fix (`agents/technical_document_agent.py`, `agents/direct_reply.py`) lets the model answer from general knowledge when retrieval only partially covers a question, while still refusing when it genuinely doesn't know — verified against the exact failing case, then confirmed with a full re-run: fact-recall 0.693 → 0.833.

Run it: `streamlit run chat_app.py`

---

## Why "Helpful" Is the Standard

A RAG system that answers correctly 60% of the time and confidently fabricates the rest is worse than no system at all — because users will trust it.

The evaluation framework here is designed to surface three specific failure modes:

| Failure | What It Means | Metric |
|---|---|---|
| **Missing facts** | Answer is incomplete — user gets partial information | FactRecall |
| **Ungrounded claims** | Answer pulls from model memory, not documents — not auditable | Groundedness rate |
| **Hallucinations** | Answer contains factually wrong information | Hallucination rate + human review |

Measuring these three things is what separates a helpful RAG from a demo.

---

## Pipeline Architecture

The ingestion pipeline feeds the evaluation framework. Its job is to get clean, relevant documents into the corpus.

```
Discovery → LLM Gate 1 → Download → Extract → LLM Gate 2 → Quality Filter → Chunk → Embed → Vector Index
```

| Stage | What It Does |
|---|---|
| **Discovery** | Searches academic APIs, RSS, Stack Overflow, vendor sites |
| **LLM Gate 1** | Classifies candidates on metadata — skips irrelevant URLs before downloading |
| **ETL** | Extracts text from PDFs, HTML, audio (Whisper transcription) |
| **LLM Gate 2** | Reads extracted text and decides if it belongs in the corpus |
| **Quality Filter** | Statistical gates: length, language, boilerplate ratio, near-duplicate detection |
| **Chunking** | Sentence-aware at max 600 chars — preserves semantic units |
| **Embedding + Indexing** | nomic-embed-text-v1.5 into LanceDB; BM25 for hybrid retrieval |

The dual LLM gate design keeps junk out of the corpus: Gate 1 operates on metadata (cheap, fast), Gate 2 operates on full extracted text (slower, higher signal).

**Corpus stats:** 849 documents, 95,579 chunks — arxiv papers, NIST docs, Stack Overflow PLC Q&A, vendor manuals, industrial automation podcasts.

---

## System Architecture

Runs across two devices:

- **NVIDIA Jetson** — always-on headless server: ingestion, Whisper transcription, ETL, PostgreSQL storage
- **Mac** — LanceDB vector indexing, analysis notebooks, Streamlit dashboards

```
ai_industry_signals/
├── discovery/          # Source discovery (academic, RSS, web, Stack Overflow)
├── etl/                # Extract, transform, load pipeline
│   ├── pipeline.py          # Core ETL + chunking
│   ├── content_screener.py  # LLM gate 2 (content quality)
│   ├── data_quality.py      # Statistical quality gates
│   └── signals.py           # Structured signal extraction
├── agents/             # Chat agent system (see "Chat Agent System" above)
│   ├── graph.py             # LangGraph StateGraph — routes to personas, compiles with checkpointer
│   ├── router.py            # Intent classification + follow-up resolution (one LLM call/turn)
│   ├── technical_document_agent.py  # HyDE + hybrid retrieval, corrective-RAG web fallback
│   ├── plc_expert.py        # PLC code Q&A + iec-checker best-practices analysis
│   ├── analytics_agent.py   # Energy telemetry Q&A + real-code-computed charts
│   ├── direct_reply.py      # Fast path — greetings, meta-questions, off-topic
│   ├── chart_directive.py   # Fail-closed chart-type parsing (see design notes above)
│   └── streaming.py         # Token-streaming helper shared by every persona node
├── tools/              # Source-specific scrapers/extractors + shared infrastructure
│   └── resilient_batch.py   # Subprocess-based defensive batch execution (real timeouts, resumable)
├── workflows/          # Operational scripts (ingest, vectorize, cleanup, rechunk)
├── notebooks/          # Four-notebook evaluation framework + eval_snapshots.json (versioned history)
├── Dashboards/         # Streamlit corpus and quality dashboards
├── chat_app.py          # Streamlit chat UI for the agent system
├── main.py               # CLI chat interface
├── llm_client.py        # BaseLLMClient interface + Gemini/OpenAI/Claude/Ollama implementations
├── db_relational.py    # DuckDB / PostgreSQL abstraction
├── db_vector_lance.py  # LanceDB vector store
├── sync_client.py      # Jetson → Mac data sync
└── device_config.py    # Hardware-aware config (Jetson / Mac / Linux)
```

---

## What This Demonstrates

- **Evaluation-first thinking** — retrieval and answer quality measured with real metrics against human-authored test sets, not vibes. `notebooks/eval_snapshots.json` versions every eval run (v1 baseline → v2 corpus expansion → v3 real-pipeline methodology fix → v4 prompt fix), so improvement claims are backed by before/after numbers, not assertions.
- **Observability over guessing** — every agent node is `@traceable` into LangSmith. Real production bugs (including the fact-recall regression fixed between v3 and v4) were found by reading actual traces, not by staring at code.
- **Guardrails as deliberate tradeoffs, not defaults** — fail-open vs. fail-closed decided per-feature based on which failure mode is worse; chart data is whitelisted and real-code-computed, never LLM-generated; a corrective-RAG fallback chain (corpus → web → explicitly-labeled ungrounded) instead of hoping the model doesn't fabricate.
- **Quality traceability** — when answers are wrong, the framework identifies which layer caused it (corpus, chunking, retrieval, or generation)
- **Pipeline discipline** — multi-stage ingestion with explicit quality gates; corpus quality is a prerequisite, not an afterthought
- **Resilient infrastructure, not just resilient prompts** — `tools/resilient_batch.py` exists because thread-based timeouts don't actually bound wall-clock time against a flaky/rate-limited API; every long-running eval in this project runs through it
- **Systems thinking** — two-device architecture, hardware-aware config, async sync, production-style logging, a formal LLM-client interface (`llm_client.BaseLLMClient`) so swapping providers touches zero call sites

---

## Adapting to a Different Domain

The pipeline and evaluation framework are domain-agnostic.

**1. Swap domain keywords** in `etl/data_quality.py`:
```python
DOMAIN_KEYWORDS = ["your", "domain", "keywords", "here"]
```

**2. Configure discovery sources** in `tools/` — each file targets a source type (arxiv, RSS, Stack Overflow, HTML, PDF).

**3. Initialize and run:**
```bash
pip install -r requirements.txt
python db_relational.py
python workflows/ingestion.py
```

**4. Vectorize:**
```bash
VECTOR_DEVICE=cpu python workflows/vectorize_lance.py --rebuild --corpus-only
```

**5. Build your evaluation test set** — 20–30 queries with human-labeled relevant document IDs and key facts per query. Run `notebooks/03_retrieval_evaluation.ipynb` then `notebooks/04_end_to_end_answer_evaluation.ipynb`.

### Environment Variables

Copy `.env.example` to `.env.mac` (or `.env`), fill in your own keys. Only
`GEMINI_API_KEY` and `TAVILY_API_KEY` are required to run the chat app —
everything else in the template is optional and commented with what it's
for. Never commit the filled-in file; `.env`, `.env.mac`, and `.env.jetson`
are all gitignored.

---

## Quick Start (chat app, Mac)

```bash
pip install -r requirements.txt
cp .env.example .env.mac        # then fill in GEMINI_API_KEY + TAVILY_API_KEY
streamlit run chat_app.py       # or: python main.py for a CLI chat
```

Requires a vectorized corpus already present under `Vectors/lance` (see
`workflows/vectorize_lance.py`) — this repo's own corpus isn't bundled.

---

## Update History

- 07/10/2026 — Chat agent system: multi-persona router (technical docs, PLC expert, analytics + charting, direct reply), token streaming, real-code-computed charts, LangSmith tracing throughout. Found and fixed a real fact-recall regression via trace triage (0.693 → 0.833). Added `tools/resilient_batch.py` (subprocess-based defensive batch execution) and a formal `llm_client.BaseLLMClient` interface. Full NB3/NB4 re-run against the current corpus, versioned in `notebooks/eval_snapshots.json`.
- 05/20/2026 — v2 complete: sentence-aware chunking, quality gates, hybrid retrieval + reranker, 31-query labeled test set, NB4 answer evaluation framework
- 03/03/2026 — Initial prototype: multi-agent system for retrieval and routing
- 02/16/2026 — Initial data infrastructure: ETL pipeline, relational DB, vector DB

## License
MIT — see `LICENSE` for details.
