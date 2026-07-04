#!/usr/bin/env python3
"""
Contextual chunking — generate per-document context summaries (Mac-only).

Calls Gemini Flash once per approved document to produce a 2-sentence context
description. Stored in content.context_summary in DuckDB. Used by
vectorize_lance.py to prepend context to each chunk before embedding.

Run this before vectorize_lance.py --rebuild whenever new docs are approved.

Usage:
    python workflows/generate_context.py              # all docs missing context
    python workflows/generate_context.py --limit 50   # process first N docs
    python workflows/generate_context.py --dry-run    # show what would run
    python workflows/generate_context.py --overwrite  # regenerate all
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from device_config import config

if not config.is_mac:
    print("ERROR: Context generation uses Gemini — Mac-only.")
    sys.exit(1)

from db_relational import relationalDB
from llm_client import GeminiClient

SYSTEM_PROMPT = (
    "You are a technical librarian indexing a manufacturing and industrial "
    "automation knowledge base. Your descriptions are precise, specific, and "
    "useful for search retrieval."
)

CONTEXT_PROMPT = """\
Write exactly 2 sentences describing the document below. These sentences will be \
prepended to every text chunk from this document so a search system understands \
what each chunk comes from.

Sentence 1: What type of document this is and who published it.
Sentence 2: What specific technology, product, or topic it covers and what a \
reader would find in it.

Do not start with "This document". Be specific — name vendors, protocols, or \
standards where present.

Title: {title}
Source: {source_name}
Type: {content_type}

Document excerpt:
{excerpt}
"""

EXCERPT_CHARS = 3000
LLM_TIMEOUT = 45  # seconds before skipping a hung request


def generate_context(llm: GeminiClient, title: str, source_name: str,
                     content_type: str, transcript: str) -> str:
    excerpt = (transcript or "")[:EXCERPT_CHARS]
    prompt = CONTEXT_PROMPT.format(
        title=title or "Unknown",
        source_name=source_name or "Unknown",
        content_type=content_type or "text",
        excerpt=excerpt,
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(llm.generate, prompt, SYSTEM_PROMPT, 0.3)
        try:
            return future.result(timeout=LLM_TIMEOUT).strip()
        except FuturesTimeout:
            future.cancel()
            raise TimeoutError(f"LLM call timed out after {LLM_TIMEOUT}s")


def run(limit: int = None, dry_run: bool = False, overwrite: bool = False):
    db = relationalDB(config.DB_PATH_ANALYTICS)
    llm = GeminiClient(model=config.LLM_MODEL)

    where = "screening_status = 'approved' AND transcript IS NOT NULL"
    if not overwrite:
        where += " AND (context_summary IS NULL OR context_summary = '')"

    rows = db.query(f"""
        SELECT id, title, source_name, content_type, transcript
        FROM content
        WHERE {where}
        ORDER BY id
        {"LIMIT " + str(limit) if limit else ""}
    """)

    total = len(rows)
    print(f"Docs to process: {total}{' (dry run)' if dry_run else ''}")

    if dry_run:
        for r in rows[:5]:
            print(f"  [{r['id']}] {str(r['title'])[:70]}")
        if total > 5:
            print(f"  ... and {total - 5} more")
        return

    ok = 0
    errors = 0
    for i, row in enumerate(rows, 1):
        doc_id = row['id']
        title = str(row['title'] or '')[:80]
        try:
            summary = generate_context(
                llm,
                title=row['title'],
                source_name=row['source_name'],
                content_type=row['content_type'],
                transcript=row['transcript'],
            )
            db.update_record(doc_id, {'context_summary': summary})
            ok += 1
            print(f"[{i}/{total}] {title[:60]}")
            print(f"  → {summary[:120]}")
            time.sleep(0.5)
        except TimeoutError:
            errors += 1
            print(f"[{i}/{total}] TIMEOUT id={doc_id} — skipping, will retry next run")
            time.sleep(2)
        except Exception as e:
            errors += 1
            print(f"[{i}/{total}] ERROR id={doc_id}: {e}")
            time.sleep(2)

    print(f"\nDone: {ok} generated, {errors} errors")


def main():
    parser = argparse.ArgumentParser(description="Generate context summaries for approved docs.")
    parser.add_argument('--limit', type=int, default=None,
                        help='Max docs to process (default: all)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print what would run without calling the LLM')
    parser.add_argument('--overwrite', action='store_true',
                        help='Regenerate summaries even for docs that already have one')
    args = parser.parse_args()

    run(limit=args.limit, dry_run=args.dry_run, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
