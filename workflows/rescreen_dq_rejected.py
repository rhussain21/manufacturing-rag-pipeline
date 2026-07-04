#!/usr/bin/env python3
"""
Re-screen dq_rejected docs using current DataQualityFilter thresholds.

Use this after adjusting DQ thresholds (e.g. lowering NEAR_DUPLICATE_THRESHOLD)
to give previously-rejected docs a second chance without re-extracting.

Usage:
    python workflows/rescreen_dq_rejected.py              # dry-run (report only)
    python workflows/rescreen_dq_rejected.py --apply      # write changes to DB
    python workflows/rescreen_dq_rejected.py --apply --skip-dedup  # skip near-dup check
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from device_config import config
from db_relational import relationalDB
from etl.data_quality import DataQualityFilter


def run(apply: bool, skip_dedup: bool):
    db = relationalDB(config.DB_PATH_ANALYTICS)

    rows = db.query("""
        SELECT id, title, transcript, source_type, content_type
        FROM content
        WHERE screening_status = 'dq_rejected'
          AND extraction_status = 'completed'
          AND transcript IS NOT NULL AND transcript != ''
        ORDER BY id
    """)

    if not rows:
        print("No dq_rejected docs to re-screen.")
        return

    print(f"Re-screening {len(rows)} dq_rejected docs "
          f"({'APPLY' if apply else 'DRY-RUN'}, skip_dedup={skip_dedup})")
    print(f"Current thresholds: NEAR_DUPLICATE_THRESHOLD="
          f"{DataQualityFilter.__init__.__defaults__}")
    print()

    dqf = DataQualityFilter()
    known_hashes = [] if not skip_dedup else None

    passed = []
    still_rejected = {}

    for row in rows:
        text = row.get('transcript', '')
        kh = known_hashes if not skip_dedup else []
        result = dqf.screen(text, known_hashes=kh)

        if not skip_dedup and result.get('simhash'):
            known_hashes.append(result['simhash'])

        if result['pass']:
            passed.append(row)
        else:
            gate = result.get('failed_gate', 'unknown')
            still_rejected[gate] = still_rejected.get(gate, 0) + 1

    print(f"Results:")
    print(f"  Would pass now:     {len(passed)}")
    print(f"  Still rejected:     {sum(still_rejected.values())}")
    if still_rejected:
        for gate, n in sorted(still_rejected.items(), key=lambda x: -x[1]):
            print(f"    {gate}: {n}")

    if apply and passed:
        print(f"\nApplying: resetting {len(passed)} docs to screening_status='pending'...")
        for row in passed:
            db.update_record(row['id'], {
                'screening_status': 'pending',
                'screening_reason': None,
                'do_not_vectorize': False,
                'marked_for_deletion': False,
            })
        print(f"Done. Re-run `python workflows/process.py` to screen them with the LLM gate.")
    elif not apply:
        print(f"\nDry-run — no changes written. Re-run with --apply to apply.")


def main():
    parser = argparse.ArgumentParser(description="Re-screen dq_rejected docs.")
    parser.add_argument('--apply', action='store_true',
                        help='Write changes to DB (default: dry-run)')
    parser.add_argument('--skip-dedup', action='store_true',
                        help='Skip near-duplicate check entirely')
    args = parser.parse_args()
    run(apply=args.apply, skip_dedup=args.skip_dedup)


if __name__ == '__main__':
    main()
