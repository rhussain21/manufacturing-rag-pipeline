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

Metrics: Recall@k, Precision@k, MRR. Failures classified into four types: hit, buried, miss, absent. Tested across dense, hybrid, and reranked configurations against a 31-query human-labeled test set.

| Method | MRR | Recall@10 |
|---|---|---|
| Dense only | 0.307 | 0.385 |
| Hybrid + Reranker | 0.412 | — |

> Retrieval is measured against a labeled test set built by hand — not a proxy metric. These numbers mean something.

### NB4 — Answer Quality
Does better retrieval actually produce better answers?

Metrics: FactRecall (did the answer contain the required facts?), Groundedness (is each claim traceable to a retrieved document?), Hallucination rate. Evaluated using LLM-as-judge with structured prompts against 15 human-authored test cases with pre-defined key facts.

> This is the final accountability check. A high-retrieval system that generates ungrounded answers is not a helpful system.

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

**Corpus stats:** 736 documents, 78,467 chunks — arxiv papers, NIST docs, Stack Overflow PLC Q&A, vendor manuals, industrial automation podcasts.

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
├── agents/             # Router, SQL, vector, and web agents
├── tools/              # Source-specific scrapers and extractors
├── workflows/          # Operational scripts (ingest, vectorize, cleanup, rechunk)
├── notebooks/          # Four-notebook evaluation framework
├── Dashboards/         # Streamlit corpus and quality dashboards
├── db_relational.py    # DuckDB / PostgreSQL abstraction
├── db_vector_lance.py  # LanceDB vector store
├── sync_client.py      # Jetson → Mac data sync
└── device_config.py    # Hardware-aware config (Jetson / Mac / Linux)
```

---

## What This Demonstrates

- **Evaluation-first thinking** — retrieval and answer quality measured with real metrics against human-authored test sets, not vibes
- **Quality traceability** — when answers are wrong, the framework identifies which layer caused it (corpus, chunking, retrieval, or generation)
- **Pipeline discipline** — multi-stage ingestion with explicit quality gates; corpus quality is a prerequisite, not an afterthought
- **Systems thinking** — two-device architecture, hardware-aware config, async sync, production-style logging

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
| Variable | Description | Default |
|---|---|---|
| `PG_HOST` | PostgreSQL host (Jetson) | `localhost` |
| `PG_PORT` | PostgreSQL port | `5432` |
| `PG_DB` | Database name | `industry_signals` |
| `VECTOR_DEVICE` | Embedding device (`cpu`, `mps`, `cuda`) | auto-detect |
| `LLM_URL` | Ollama endpoint for local LLM | `http://localhost:11434` |

---

## Update History

- 05/20/2026 — v2 complete: sentence-aware chunking, quality gates, hybrid retrieval + reranker, 31-query labeled test set, NB4 answer evaluation framework
- 03/03/2026 — Initial prototype: multi-agent system for retrieval and routing
- 02/16/2026 — Initial data infrastructure: ETL pipeline, relational DB, vector DB

## License
MIT — see `LICENSE` for details.
