"""
Tests for tools/resilient_batch.py — deterministic, no real API calls.

run_batch() re-invokes sys.argv[0] as a worker subprocess internally, so it
can't be exercised by calling it directly from a pytest test process (that
would try to re-spawn pytest itself). Instead, each test writes a small
self-contained fixture script to a temp file and runs IT as a subprocess —
exactly how a real caller would use this module — then inspects the
checkpoint file and stdout.

Run:
    pytest test_resilient_batch.py -v
"""

import json
import subprocess
import sys
import textwrap

PROJECT_ROOT = "/Users/redwanhussain/Documents/ai-projects/ai_industry_signals"

_FIXTURE_SCRIPT = textwrap.dedent(f"""
    import sys
    sys.path.insert(0, {PROJECT_ROOT!r})
    from tools.resilient_batch import run_batch

    def setup():
        return {{}}

    def handle(job, ctx):
        import time
        if job["id"] == "hang1":
            time.sleep(30)  # far longer than the test's timeout — must be killed, not waited out
        return {{"doubled": job["n"] * 2}}

    if __name__ == "__main__":
        jobs = [
            {{"id": "fast1", "n": 1}},
            {{"id": "fast2", "n": 2}},
            {{"id": "hang1", "n": 3}},
            {{"id": "fast3", "n": 4}},
        ]
        results = run_batch(jobs, setup=setup, handle=handle,
                             checkpoint_path=sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "checkpoint.json",
                             timeout=2, ready_timeout=30, max_hangs_per_job=2)
        print("SCRIPT_RESULT:" + __import__("json").dumps(results))
""")


def _write_fixture(tmp_path):
    script = tmp_path / "fixture.py"
    script.write_text(_FIXTURE_SCRIPT)
    return script


def test_fast_jobs_succeed_and_hang_job_recovers(tmp_path):
    script = _write_fixture(tmp_path)
    checkpoint = tmp_path / "checkpoint.json"

    proc = subprocess.run(
        [sys.executable, str(script), str(checkpoint)],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr

    results = json.load(open(checkpoint))
    assert results["fast1"] == {"ok": True, "result": {"doubled": 2}}
    assert results["fast2"] == {"ok": True, "result": {"doubled": 4}}
    assert results["fast3"] == {"ok": True, "result": {"doubled": 8}}
    # hang1 sleeps 30s against a 2s timeout with 2 allowed hang attempts —
    # must be killed both times and marked failed, never actually waited on.
    assert results["hang1"]["ok"] is False
    assert "hung" in results["hang1"]["error"]


def test_hang_does_not_block_or_lose_later_jobs(tmp_path):
    """The real bug this module exists to fix: a thread-based timeout lets
    an abandoned call keep running and starve everything queued behind it.
    fast3 (queued after hang1) must complete quickly, not wait ~30s for
    hang1's sleep to finish."""
    script = _write_fixture(tmp_path)
    checkpoint = tmp_path / "checkpoint.json"

    import time
    t0 = time.time()
    subprocess.run([sys.executable, str(script), str(checkpoint)],
                    capture_output=True, text=True, timeout=60)
    elapsed = time.time() - t0

    # Worst case: 2 hang attempts x 2s timeout + fast job overhead — well
    # under the 30s the hang job itself sleeps for, proving it was killed
    # rather than waited out.
    assert elapsed < 15, f"took {elapsed:.1f}s — hang1 likely blocked the queue instead of being killed"

    results = json.load(open(checkpoint))
    assert results["fast3"]["ok"] is True


def test_resumable_checkpoint_skips_completed_jobs(tmp_path):
    script = _write_fixture(tmp_path)
    checkpoint = tmp_path / "checkpoint.json"

    # Pre-seed the checkpoint as if a prior run already finished fast1/fast2.
    checkpoint.write_text(json.dumps({
        "fast1": {"ok": True, "result": {"doubled": 999}},  # sentinel value proving this wasn't recomputed
        "fast2": {"ok": True, "result": {"doubled": 4}},
    }))

    proc = subprocess.run(
        [sys.executable, str(script), str(checkpoint)],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr

    results = json.load(open(checkpoint))
    # fast1 was NOT re-run — the pre-seeded (wrong) sentinel value survives,
    # proving run_batch actually skipped it rather than recomputing.
    assert results["fast1"]["result"]["doubled"] == 999
    assert results["fast3"]["ok"] is True  # the genuinely-remaining job still ran
