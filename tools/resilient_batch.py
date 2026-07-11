"""
Resilient batch execution for flaky or rate-limited external APIs.

## Why this exists

Thread-based timeouts (`ThreadPoolExecutor` + `future.result(timeout=X)`) do
not reliably bound wall-clock time. A Python thread cannot be forcibly
killed — a call that hangs internally (a stuck HTTP request, a retry loop
that never gives up, a laptop going to sleep mid-call) keeps running in the
background even after the caller times out, and can starve every later job
queued behind it on the same thread pool. This was not a theoretical
concern: building the eval pipeline for this project's chat agents, calls
into a rate-limited LLM API repeatedly completed successfully hundreds of
seconds past their stated thread-based timeout — the timeout mechanism
looked like it was working (it raised on schedule) but never actually
stopped the underlying work, so a single stuck call could silently turn a
15-minute job into a multi-hour one.

This module runs each unit of work in a persistent WORKER SUBPROCESS
instead. A subprocess can be forcibly killed (`SIGKILL`) regardless of what
it's doing internally — a real, verified guarantee threads don't offer. The
worker does expensive one-time setup (model loading, DB connections, API
clients) exactly once, then processes jobs one at a time over a
stdin/stdout JSON-lines protocol. If a job doesn't come back within the
timeout, the whole worker process is killed and a fresh one is spawned for
the next job — no zombie thread, no starved queue, no silent hang.

## Two more problems this had to solve

1. **Native libraries write straight to the OS-level stdout file
   descriptor**, which bypasses Python's `contextlib.redirect_stdout`
   entirely (it only patches the Python-level `sys.stdout` object, not the
   underlying fd). A naive "every stdout line is a protocol message" design
   breaks the moment a dependency prints a warning. Every real protocol
   message here is prefixed with a sentinel the reader filters on;
   anything else is silently treated as noise, never a parse error.

2. **Long batch jobs need to survive being interrupted** — a killed
   worker, a laptop sleeping, a crashed orchestrator — without losing
   completed work. Every result is written to the checkpoint file
   immediately after each job. A full restart just skips whatever's
   already there.

## Usage

Define two functions and a list of jobs, then call `run_batch`. The calling
script must dispatch through `run_batch` itself (not conditionally) because
`run_batch` re-invokes the same script as a worker subprocess internally —
see the `if __name__` pattern below.

    def setup():
        # Runs once, in the worker subprocess. Do expensive one-time work
        # here (load a model, open a client) and return whatever handle()
        # needs.
        return {"client": SomeExpensiveClient()}

    def handle(job: dict, ctx) -> dict:
        # Runs once per job, using the context setup() returned. Must
        # return a JSON-serializable dict.
        return {"answer": ctx["client"].ask(job["query"])}

    if __name__ == "__main__":
        from tools.resilient_batch import run_batch

        jobs = [{"id": "q1", "query": "..."}, {"id": "q2", "query": "..."}]
        results = run_batch(
            jobs, setup=setup, handle=handle,
            checkpoint_path="checkpoint.json", timeout=150,
        )

Run the script normally (`python my_script.py`) — `run_batch` detects
whether it's the parent orchestrator or the re-invoked worker subprocess
and does the right thing in each.
"""

import json
import os
import subprocess
import sys
import threading
import time
from queue import Empty, Queue

_SENTINEL = "###RESILIENT_BATCH### "
_WORKER_FLAG = "--resilient-batch-worker"


def _respond(obj: dict) -> None:
    print(_SENTINEL + json.dumps(obj), flush=True)


def _run_worker_loop(setup, handle) -> None:
    """Runs inside the worker subprocess. One-time setup, then a
    read-handle-respond loop over stdin/stdout."""
    ctx = setup()
    _respond({"ready": True})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        job = json.loads(line)
        try:
            result = handle(job, ctx)
            _respond({"id": job["id"], "ok": True, "result": result})
        except Exception as e:
            _respond({"id": job["id"], "ok": False, "error": repr(e)})


class _Worker:
    """One live worker subprocess plus its stdout-reader thread/queue.
    The reader thread exists so `ask()` can use `Queue.get(timeout=X)` — a
    real blocking-with-timeout primitive — instead of a raw blocking
    `readline()` that a hung child could block forever."""

    def __init__(self, worker_argv: list, ready_timeout: float):
        self.proc = subprocess.Popen(
            worker_argv,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr,
            text=True, bufsize=1,
        )
        self.q: Queue = Queue()

        def reader():
            for line in self.proc.stdout:
                if line.startswith(_SENTINEL):
                    self.q.put(line[len(_SENTINEL):])
                # else: noise from a dependency writing to the real stdout
                # fd underneath us — silently dropped, by design.
            self.q.put(None)  # EOF sentinel, in case the process exits

        threading.Thread(target=reader, daemon=True).start()
        ready = self.q.get(timeout=ready_timeout)
        if not (ready and json.loads(ready).get("ready")):
            raise RuntimeError(f"worker failed to start: {ready!r}")

    def ask(self, job: dict, timeout: float) -> dict | None:
        """Returns the parsed response, or None if the worker hung or died
        — in which case this _Worker is dead and must be discarded."""
        try:
            self.proc.stdin.write(json.dumps(job) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, ValueError):
            return None
        try:
            line = self.q.get(timeout=timeout)
        except Empty:
            return None
        return json.loads(line) if line else None

    def kill(self) -> None:
        try:
            self.proc.kill()
            self.proc.wait(timeout=10)
        except Exception:
            pass


def run_batch(
    jobs: list[dict],
    setup,
    handle,
    checkpoint_path: str,
    timeout: float = 150,
    ready_timeout: float = 120,
    inter_job_delay: float = 0.0,
    max_hangs_per_job: int = 2,
) -> dict:
    """Runs `jobs` (each a dict with a unique "id" key) through `handle`,
    resuming from `checkpoint_path` if it already has results for some
    job ids. Returns the full results dict ({job_id: {"ok": bool, ...}}),
    which is also what's on disk at `checkpoint_path` when this returns.

    If a job doesn't respond within `timeout` seconds, the worker is killed
    and a fresh one spawned; the same job is retried once against the new
    worker before being marked failed (`max_hangs_per_job` controls this).
    A killed/restarted worker never silently drops the rest of the batch —
    it just picks up with the next job on a clean process.
    """
    worker_argv = [sys.executable, os.path.abspath(sys.argv[0]), _WORKER_FLAG]

    if _WORKER_FLAG in sys.argv:
        _run_worker_loop(setup, handle)
        sys.exit(0)

    results: dict = {}
    if os.path.exists(checkpoint_path):
        results = json.load(open(checkpoint_path))

    remaining = [j for j in jobs if j["id"] not in results]

    def _spawn() -> _Worker:
        return _Worker(worker_argv, ready_timeout)

    worker = _spawn()
    for job in remaining:
        attempts_left = max_hangs_per_job
        resp = None
        while attempts_left > 0:
            resp = worker.ask(job, timeout)
            if resp is not None:
                break
            worker.kill()
            worker = _spawn()
            attempts_left -= 1

        if resp is None:
            results[job["id"]] = {"ok": False, "error": f"hung {max_hangs_per_job}x, gave up"}
        elif resp.get("ok"):
            results[job["id"]] = {"ok": True, "result": resp["result"]}
        else:
            results[job["id"]] = {"ok": False, "error": resp.get("error")}

        json.dump(results, open(checkpoint_path, "w"), indent=2)
        if inter_job_delay:
            time.sleep(inter_job_delay)

    worker.kill()
    return results
