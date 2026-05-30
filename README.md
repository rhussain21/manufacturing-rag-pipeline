# Manufacturing RAG Pipeline
*A production-style pipeline for building, curating, and evaluating a domain-specific RAG corpus — built from scratch to understand where retrieval actually breaks.*

---

## Why This Exists

Most RAG tutorials skip the hard part: getting the data right.

They start with clean, well-formatted documents and a few lines of LangChain. Real-world RAG fails before the LLM ever sees a query — because the corpus is noisy, the chunking destroys context, or retrieval returns the wrong documents entirely.

This project was built to understand those failure modes firsthand, in a domain with real complexity: industrial automation and manufacturing. The domain choice matters — you can only write meaningful evaluations when you know what a correct answer actually looks like.

**The goal was not to build yet another RAG chatbot. The goal was to build one I could measure.**

---

## What Makes This Different From "Just Using ChatGPT"

A general-purpose LLM already knows what a PLC is. It can explain SCADA, discuss NIST standards, and summarize industrial automation concepts.

What it cannot do:
- Answer questions about a specific manufacturer's private SOPs
- Cite the exact document and page a technician should reference
- Reason over a corpus of internal maintenance logs, machine configs, and proprietary procedures

This pipeline is the infrastructure layer that makes private document intelligence possible — without sending any documents to a cloud provider.

---

## Pipeline Architecture

```
Discovery → LLM Gate 1 → Download → Extract → LLM Gate 2 → Quality Filter → Chunk → Embed → Vector Index
```

| Stage | What It Does | Why It's Here |
|---|---|---|
| **Discovery** | Searches academic APIs, RSS, Stack Overflow, vendor sites | Finds candidates without manual curation |
| **LLM Gate 1** (Candidate Classifier) | Inspects metadata — decides if a URL is worth downloading | Avoids downloading irrelevant content at scale |
| **ETL** | Extracts text from PDFs, HTML, audio (Whisper transcription) | Handles the format diversity of real technical docs |
| **LLM Gate 2** (Content Screener) | Reads extracted text — decides if it belongs in the corpus | The quality bar that makes retrieval trustworthy |
| **Quality Filter** | Statistical gates: length, language, boilerplate ratio, near-duplicate detection | Catches what the LLM gate misses |
| **Chunking** | Sentence-aware, max 600 chars via NLTK | Preserves semantic units; tested against naive splitting |
| **Embedding + Indexing** | nomic-embed-text-v1.5 (768-dim) into LanceDB; BM25 for hybrid retrieval | Enables both dense and keyword-based retrieval |

The dual LLM gate design is deliberate: Gate 1 operates on metadata (cheap, fast), Gate 2 operates on full extracted text (slower, higher signal). Together they keep junk out of the corpus without requiring manual review.

---

## Corpus and Retrieval Results

**Corpus v2**
- 736 documents — arxiv ML/manufacturing papers, NIST docs, Stack Overflow PLC Q&A, Siemens/Rockwell/Schneider/Beckhoff/ABB manuals, industrial automation podcasts
- 78,467 chunks at max_chars=600, sentence-aware

**Retrieval evaluation** (31 labeled queries, human-judged relevance):

| Method | MRR | Recall@10 |
|---|---|---|
| Dense only | 0.307 | 0.385 |
| Hybrid + Reranker | 0.374 | — |

The labeled test set was built by hand — queries written against known documents — so these numbers reflect actual retrieval quality, not proxy metrics.

---

## What This Demonstrates

- **Pipeline design** — multi-stage ingestion with explicit quality gates, not a single ingest-everything approach
- **Evaluation discipline** — retrieval measured with MRR and Recall@k against a labeled test set; corpus quality assessed before retrieval is even run
- **Systems thinking** — runs across two devices (Jetson for always-on ingestion, Mac for analysis); hardware-aware config, async sync between environments
- **Domain grounding** — evals are only meaningful when you know what correct looks like; manufacturing was chosen specifically because the domain knowledge was already there

---

## System Architecture

Runs across two devices:

- **NVIDIA Jetson** — always-on headless server: ingestion, Whisper transcription, ETL, PostgreSQL storage
- **Mac** — development, LanceDB vector indexing, analysis notebooks, Streamlit dashboards

```
ai_industry_signals/
├── discovery/          # Source discovery (academic, RSS, web, Stack Overflow)
├── etl/                # Extract, transform, load pipeline
│   ├── pipeline.py          # Core ETL + chunking
│   ├── content_screener.py  # LLM gate 2
│   ├── data_quality.py      # Statistical quality gates
│   └── signals.py           # Structured signal extraction
├── agents/             # Router, SQL, vector, and web agents
├── tools/              # Source-specific scrapers and extractors
├── workflows/          # Operational scripts (ingest, vectorize, cleanup, rechunk)
├── notebooks/          # Corpus quality, chunking, retrieval, and E2E evaluation
├── Dashboards/         # Streamlit admin and corpus dashboards
├── db_relational.py    # DuckDB / PostgreSQL abstraction
├── db_vector_lance.py  # LanceDB vector store
├── sync_client.py      # Jetson → Mac data sync
└── device_config.py    # Hardware-aware config (Jetson / Mac / Linux)
```

---

## Adapting to a Different Domain

The pipeline is domain-agnostic. To point it at a different corpus:

**1. Swap the domain keywords** in `etl/data_quality.py`:
```python
DOMAIN_KEYWORDS = ["your", "domain", "keywords", "here"]
```

**2. Configure discovery sources** in `tools/` — each file targets a source type (arxiv, RSS, Stack Overflow, HTML, PDF). Add or remove based on what exists for your domain.

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

**5. Evaluate retrieval** — build a 20-30 query labeled test set and run `notebooks/03_retrieval_evaluation.ipynb`.

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

- 05/20/2026 — v2 complete: sentence-aware chunking, quality gates, hybrid retrieval, reranker evaluation, 31-query labeled test set, corpus re-indexed (78,467 chunks)
- 03/03/2026 — Initial prototype: multi-agent system for retrieval and routing
- 02/16/2026 — Initial data infrastructure: ETL pipeline, relational DB, vector DB

## License
MIT — see `LICENSE` for details.
