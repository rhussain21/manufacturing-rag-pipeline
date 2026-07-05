# AI Industry Signals — Project Context

Manufacturing-domain RAG corpus pipeline. Jetson is the always-on processing server; Mac is used for dev, analysis, and notebooks.

## Hardware roles
- **Jetson**: runs ingestion, ETL, screening, signal extraction. DB lives here (source of truth).
- **Mac**: syncs DB from Jetson (`sync_client.py`), runs vectorization into LanceDB, runs notebooks.

## Pipeline order
```
discovery/ → download → workflows/process.py (extract → DQ filter → LLM screen → signals) → sync to Mac → workflows/vectorize_lance.py → notebooks/
```

## Key commands (Jetson)
```bash
conda activate jetson_cuda          # always use this env
python workflows/ingestion.py       # discover + download new content
python workflows/process.py         # extract → screen (main ETL; signal extraction is off by default)
python workflows/process.py --run-signals   # opt in to signal extraction (cut from the default pipeline — NB3 showed it hurts retrieval)
```

## Key files
- `db_relational.py` — DuckDB wrapper. DB path from `config.DB_PATH`.
- `etl/pipeline.py` — content extraction + DQ filter (`DataQualityFilter`)
- `etl/content_screener.py` — rule-based gate + LLM screen (Ollama on Jetson)
- `etl/data_quality.py` — statistical DQ gates. `NEAR_DUPLICATE_THRESHOLD=2` (recently lowered from 5).
- `etl/signals.py` — langextract signal extraction
- `device_config.py` — resolves all paths and config per device (mac vs jetson)
- `workflows/` — standalone runnable scripts for each pipeline stage

## DB schema (DuckDB)
Main table: `content`
- `screening_status`: `pending`, `approved`, `rejected`, `dq_rejected`, `error`
- `extraction_status`: `pending`, `completed`, `failed`
- `signal_processed`: bool
- `do_not_vectorize`: bool

Other tables: `signals`, `system_logs`, `transcript_segments`

## Current task (June 2026)
A batch of ~620 new docs was ingested but most got `dq_rejected` due to an over-aggressive near-duplicate threshold (was 5 bits, now fixed to 2). Goal: reset those docs to `extraction_status='pending'` and re-run `process.py` so they go through extraction + LLM screening properly.

Reset command:
```python
from db_relational import relationalDB
from device_config import config
db = relationalDB(config.DB_PATH)
db.execute("""
    UPDATE content
    SET extraction_status='pending',
        screening_status=NULL,
        screening_reason=NULL,
        do_not_vectorize=FALSE
    WHERE screening_status='dq_rejected'
""")
```
