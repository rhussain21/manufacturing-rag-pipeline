#!/usr/bin/env python3
"""
Retroactive Re-chunking — Jetson (PostgreSQL) side.

Replaces the stored segments for every approved document with fresh
sentence-aware chunks at max_chars=600.  Bumps updated_at on each row
so sync_client picks up the changes and propagates them to the Mac DuckDB.

After this runs on Jetson, the Mac workflow is:
    1. python sync_client.py          # pull updated segments to DuckDB
    2. python workflows/vectorize_lance.py --rebuild --corpus-only
                                      # re-embed with new chunks → LanceDB

Usage:
    # Dry run — prints what would change, no DB writes
    python workflows/rechunk.py

    # Apply — writes new segments and bumps updated_at
    python workflows/rechunk.py --apply

    # Limit to N docs (useful for testing)
    python workflows/rechunk.py --apply --limit 10

    # Custom chunk size (default 600)
    python workflows/rechunk.py --apply --max-chars 600
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db_relational import relationalDB
from device_config import config


# ── Self-contained sentence-aware chunker ─────────────────────────────────────
# Copied from etl/pipeline.py so this script works regardless of the Jetson's
# pipeline.py version.

def _split_sentences(text: str):
    """Split text into sentences using NLTK if available, else regex fallback."""
    try:
        import nltk
        try:
            return nltk.sent_tokenize(text)
        except LookupError:
            nltk.download('punkt_tab', quiet=True)
            nltk.download('punkt', quiet=True)
            return nltk.sent_tokenize(text)
    except ImportError:
        # Regex fallback: split on .  !  ? followed by whitespace
        parts = re.split(r'(?<=[.!?])\s+', text)
        return [p.strip() for p in parts if p.strip()]


def chunk_text(text: str, max_chars: int = 600, overlap_sents: int = 1):
    """
    Sentence-aware chunker. Accumulates sentences until adding the next would
    exceed max_chars, then flushes. overlap_sents sentences from the end of the
    previous chunk are prepended to the next for context continuity.

    Returns a list of dicts: {text, char_count, start_char, end_char}
    """
    sentences = _split_sentences(text)
    chunks = []
    current: list = []
    current_len = 0

    for sent in sentences:
        if current_len + len(sent) > max_chars and current:
            chunk_text_str = ' '.join(current)
            chunks.append({
                'text':       chunk_text_str,
                'char_count': len(chunk_text_str),
                'start_char': text.find(current[0]),
                'end_char':   text.find(current[0]) + len(chunk_text_str),
            })
            current = current[-overlap_sents:] if overlap_sents else []
            current_len = sum(len(s) for s in current)
        current.append(sent)
        current_len += len(sent)

    if current:
        chunk_text_str = ' '.join(current)
        chunks.append({
            'text':       chunk_text_str,
            'char_count': len(chunk_text_str),
            'start_char': text.find(current[0]),
            'end_char':   text.find(current[0]) + len(chunk_text_str),
        })

    return chunks


# ── Main ──────────────────────────────────────────────────────────────────────

def run(max_chars: int, apply: bool, limit: int | None):
    db = relationalDB(getattr(config, 'DB_PATH_ANALYTICS', config.DB_PATH))

    query = """
        SELECT id, transcript, segments
        FROM content
        WHERE screening_status = 'approved'
          AND transcript IS NOT NULL
          AND LENGTH(transcript) > 200
        ORDER BY id
    """
    if limit:
        query += f" LIMIT {limit}"

    rows = db.query(query)
    print(f"Found {len(rows)} approved documents to rechunk.")
    print(f"Settings: max_chars={max_chars}  overlap_sents=1  apply={apply}\n")

    total_old_chunks = 0
    total_new_chunks = 0

    for i, row in enumerate(rows, 1):
        doc_id     = row['id']
        transcript = row['transcript']

        old_segs = []
        try:
            raw = row.get('segments') or ''
            if raw:
                old_segs = json.loads(raw)
        except (json.JSONDecodeError, TypeError, AttributeError):
            old_segs = []

        new_chunks = chunk_text(transcript, max_chars=max_chars, overlap_sents=1)
        total_old_chunks += len(old_segs)
        total_new_chunks += len(new_chunks)

        if i <= 5 or i % 100 == 0:
            print(
                f"  [{i:>4}/{len(rows)}] id={doc_id:>5}  "
                f"old={len(old_segs):>4} chunks  →  new={len(new_chunks):>4} chunks"
            )

        if apply:
            db.update_record(doc_id, {'segments': json.dumps(new_chunks)})

    print(f"\nSummary:")
    print(f"  Docs processed : {len(rows)}")
    print(f"  Old chunks     : {total_old_chunks:,}")
    print(f"  New chunks     : {total_new_chunks:,}")
    print(f"  Delta          : {total_new_chunks - total_old_chunks:+,}")

    if not apply:
        print("\nDry run — pass --apply to write changes to PostgreSQL.")
        return

    print("\nDone. updated_at bumped on all rechunked rows.")
    print("Next steps on Mac:")
    print("  1. python sync_client.py")
    print("  2. python workflows/vectorize_lance.py --rebuild --corpus-only --model nomic-ai/nomic-embed-text-v1.5")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Retroactive rechunk — Jetson PostgreSQL side.')
    parser.add_argument('--apply',     action='store_true', help='Write to DB (default: dry run)')
    parser.add_argument('--max-chars', type=int, default=600, help='Chunk size in chars (default: 600)')
    parser.add_argument('--limit',     type=int, default=None, help='Process only N docs (for testing)')
    args = parser.parse_args()

    print(f"Rechunk Workflow  |  max_chars={args.max_chars}  apply={args.apply}")
    print(f"Backend: {config.DB_BACKEND if hasattr(config, 'DB_BACKEND') else 'auto-detect'}\n")

    run(max_chars=args.max_chars, apply=args.apply, limit=args.limit)
