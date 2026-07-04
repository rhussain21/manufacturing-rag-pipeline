#!/usr/bin/env python3
"""
Retroactive Near-Duplicate Deduplication.

Runs SimHash across every non-deleted document in the DB, finds near-duplicate
pairs (hamming distance ≤ threshold), groups them into clusters via union-find,
then marks the lower-quality copy in each cluster for deletion.

Quality score used to pick the survivor in a cluster (higher = better):
  +3   signal_processed = True
  +2   pub_date is non-empty
  +1   source_type in ('pdf', 'html')   — structured sources beat raw text/audio
  +1   transcript is longer than cluster median

Designed to run on both Jetson (PostgreSQL) and Mac (DuckDB) — db_relational.py
auto-detects the backend from device_config.

Usage:
    # Dry run — prints duplicate clusters, no DB writes
    python workflows/dedup.py

    # Apply — marks lower-quality duplicates for deletion
    python workflows/dedup.py --apply

    # Tighter similarity threshold (default 5 bits)
    python workflows/dedup.py --threshold 3 --apply

    # Limit to content ingested after a date
    python workflows/dedup.py --since 2025-01-01 --apply

    # Verbose: print per-gate details for every cluster
    python workflows/dedup.py --verbose
"""

import argparse
import os
import sys
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from device_config import config
from db_relational import relationalDB
from etl.data_quality import DataQualityFilter


# ── Union-Find ───────────────────────────────────────────────────────────────

class UnionFind:
    def __init__(self):
        self._parent: dict = {}

    def find(self, x):
        if x not in self._parent:
            self._parent[x] = x
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]  # path compression
            x = self._parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[rb] = ra

    def clusters(self) -> dict:
        """Return {root: [members]} for all groups with >1 member."""
        groups: dict = defaultdict(list)
        for x in self._parent:
            groups[self.find(x)].append(x)
        return {r: members for r, members in groups.items() if len(members) > 1}


# ── Quality scorer ───────────────────────────────────────────────────────────

def quality_score(row: dict) -> int:
    score = 0
    if row.get('signal_processed'):
        score += 3
    if row.get('pub_date'):
        score += 2
    if row.get('source_type') in ('pdf', 'html'):
        score += 1
    return score


# ── Main ─────────────────────────────────────────────────────────────────────

def run(threshold: int, apply: bool, since: str, verbose: bool):
    db = relationalDB(config.DB_PATH)
    dqf = DataQualityFilter()

    # ── Load records ─────────────────────────────────────────────────────────
    where_clauses = [
        "marked_for_deletion = FALSE",
        "transcript IS NOT NULL",
        "LENGTH(transcript) > 0",
    ]
    if since:
        where_clauses.append(f"created_at >= '{since}'")

    query = f"""
        SELECT id, title, source_type, pub_date, file_size_mb,
               signal_processed, transcript, screening_status
        FROM content
        WHERE {' AND '.join(where_clauses)}
        ORDER BY id
    """
    rows = db.query(query)

    if not rows:
        print("No eligible documents found.")
        return

    records = [dict(r) for r in rows]
    print(f"Loaded {len(records)} documents. Computing SimHashes...")

    # ── Compute fingerprints ──────────────────────────────────────────────────
    for rec in records:
        rec['_hash'] = dqf.simhash(rec['transcript'] or '')
        rec['_len'] = len(rec['transcript'] or '')

    print(f"Done. Running {len(records)}² pairwise comparison ({len(records)**2 // 2:,} pairs)...")

    # ── Find near-dup pairs ───────────────────────────────────────────────────
    uf = UnionFind()
    pair_count = 0
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            dist = DataQualityFilter.hamming_distance(
                records[i]['_hash'], records[j]['_hash']
            )
            if dist <= threshold:
                uf.union(records[i]['id'], records[j]['id'])
                pair_count += 1

    clusters = uf.clusters()
    print(f"Found {pair_count} near-duplicate pairs → {len(clusters)} duplicate cluster(s).\n")

    if not clusters:
        print("Database is clean — no near-duplicates found.")
        return

    # Build id→record lookup
    by_id = {r['id']: r for r in records}

    # ── Resolve each cluster ──────────────────────────────────────────────────
    to_mark: list = []   # list of (loser_id, winner_id, reason)
    report_lines: list = []

    for root, members in sorted(clusters.items()):
        cluster_records = [by_id[mid] for mid in members if mid in by_id]
        if not cluster_records:
            continue

        median_len = sorted(r['_len'] for r in cluster_records)[len(cluster_records) // 2]

        # Score each member (add length bonus against cluster median)
        scored = []
        for rec in cluster_records:
            s = quality_score(rec)
            if rec['_len'] >= median_len:
                s += 1
            scored.append((s, rec))

        scored.sort(key=lambda x: (-x[0], x[1]['id']))  # best score, lowest id as tiebreak
        winner_score, winner = scored[0]

        report_lines.append(
            f"  KEEP  id={winner['id']:>5}  score={winner_score}  "
            f"src={winner.get('source_type','?'):<6}  "
            f"signals={'Y' if winner.get('signal_processed') else 'N'}  "
            f"{(winner.get('title') or '')[:70]}"
        )
        for _, loser in scored[1:]:
            reason = (
                f"near_duplicate: simhash_dist≤{threshold}, "
                f"kept id={winner['id']}"
            )
            to_mark.append((loser['id'], winner['id'], reason))
            report_lines.append(
                f"  MARK  id={loser['id']:>5}  score={_}  "
                f"src={loser.get('source_type','?'):<6}  "
                f"signals={'Y' if loser.get('signal_processed') else 'N'}  "
                f"{(loser.get('title') or '')[:70]}"
            )
        report_lines.append("")

        if verbose:
            for s, rec in scored:
                print(
                    f"    id={rec['id']}  score={s}  len={rec['_len']}  "
                    f"hash={rec['_hash']:016x}  {rec.get('title','')[:60]}"
                )

    # ── Print report ──────────────────────────────────────────────────────────
    print("=== Duplicate Clusters ===")
    for line in report_lines:
        print(line)

    print(f"Summary: {len(to_mark)} document(s) would be marked for deletion.")

    if not apply:
        print("\nDry run — pass --apply to write changes to the database.")
        return

    # ── Apply ─────────────────────────────────────────────────────────────────
    marked = 0
    for loser_id, winner_id, reason in to_mark:
        try:
            db.update_record(loser_id, {
                'marked_for_deletion': True,
                'screening_status': 'dq_rejected',
                'screening_reason': reason,
            })
            marked += 1
        except Exception as e:
            print(f"  ERROR marking id={loser_id}: {e}")

    print(f"\nApplied: {marked}/{len(to_mark)} records marked for deletion.")
    print("Run  python workflows/cleanup.py --delete  to remove their files from disk.")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Retroactive near-duplicate deduplication.')
    parser.add_argument(
        '--apply', action='store_true',
        help='Write marked_for_deletion to DB (default: dry run)'
    )
    parser.add_argument(
        '--threshold', type=int, default=5,
        help='SimHash hamming distance threshold (default: 5)'
    )
    parser.add_argument(
        '--since', default='',
        help='Only consider documents ingested on/after this date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Print per-record scores within each cluster'
    )
    args = parser.parse_args()

    print(f"Near-Dup Dedup  |  threshold={args.threshold}  apply={args.apply}  since={args.since or 'all'}")
    print(f"Backend: {config.DB_BACKEND if hasattr(config, 'DB_BACKEND') else 'auto-detect'}\n")

    run(
        threshold=args.threshold,
        apply=args.apply,
        since=args.since,
        verbose=args.verbose,
    )
