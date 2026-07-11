"""
Shared token-streaming helper for the terminal LLM call in each persona
node. Streams text deltas to the UI live via LangGraph's custom stream
mode (get_stream_writer) while the node is still running, then returns
the FULL raw text — so the existing post-hoc parsing
(agents/structured_answer.parse_structured_answer) keeps working on the
return value completely unchanged. Only what reaches the user mid-flight
changes.

The one wrinkle streaming introduces: the structured-output marker line
("USED_CONTEXT: true/false", appended by the model at the very end) would
flash on screen before parse_structured_answer ever gets a chance to strip
it. So this helper withholds only the trailing portion of the stream while
it could still be a marker line — a completed line of real content always
flushes immediately; only the in-progress trailing line is ever held back,
and only while it's still a case-insensitive prefix of (or starts with)
one of marker_prefixes. At stream end, a held tail that actually starts
with a full marker prefix is silently discarded from the DISPLAY (the
parser strips it from the stored answer anyway); anything else held gets
flushed as real content — fail-open, same spirit as parse_structured_answer.

Writer resolution note (verified empirically on langgraph 1.2.7, not
assumed): get_stream_writer() RAISES RuntimeError ("Called get_config
outside of a runnable context") when called outside a runnable context —
which is exactly where multi_intent_step's ThreadPoolExecutor-spawned
persona calls land, since contextvars don't propagate into plain threads.
It does NOT return a no-op writer there, contrary to the original design
assumption. The try/except fallback below converts that into a real no-op,
which is what keeps the same node functions serving both the streamed
single-persona path and the non-streamed multi_intent path with zero
branching in agents/graph.py (and also makes direct node calls in
tests/notebooks safe). Under graph.invoke() the writer exists and simply
discards writes — also verified — so the CLI (main.py) is unaffected.
"""

import logging

from langgraph.config import get_stream_writer

logger = logging.getLogger(__name__)


def _noop_writer(_payload) -> None:
    pass


def _resolve_writer():
    try:
        return get_stream_writer()
    except RuntimeError:
        # Outside a runnable context: multi_intent's worker threads, or a
        # node function called directly as a plain Python function.
        return _noop_writer


def _hold_start(text: str, prefixes_upper: tuple) -> int:
    """Index in `text` where the withheld (not-yet-emitted) tail begins.

    The withheld tail is the last non-whitespace line (plus any trailing
    whitespace after it) IF that line could still be, or already is, a
    marker line: its content is a case-insensitive prefix of one of the
    markers (still being typed out), or starts with a full marker. Any
    other trailing content — including a partial line that has already
    diverged from every marker — is safe to emit immediately.
    """
    if not prefixes_upper:
        return len(text)
    core = text.rstrip()
    line_start = core.rfind("\n") + 1
    line_upper = core[line_start:].lstrip().upper()
    for prefix in prefixes_upper:
        if prefix.startswith(line_upper) or line_upper.startswith(prefix):
            return line_start
    return len(text)


def _is_marker(held: str, prefixes_upper: tuple) -> bool:
    """True if the held tail is an actual marker line (full prefix present),
    as opposed to a partial prefix that never completed (which flushes)."""
    held_upper = held.strip().upper()
    return any(held_upper.startswith(prefix) for prefix in prefixes_upper)


def stream_llm_answer(llm_client, prompt, system_prompt, temperature,
                      marker_prefixes=("USED_CONTEXT:",)) -> str:
    """Drop-in replacement for llm_client.generate(...) inside a node:
    same return value (the full, unstripped raw text), but each delta is
    also emitted live as a {"delta": text} custom-stream payload — except
    the trailing marker line, which is withheld from display (see module
    docstring for the exact buffering rule).

    marker_prefixes=() disables withholding entirely (direct_reply, which
    has no marker) — every delta flushes as soon as it arrives.
    """
    writer = _resolve_writer()

    # Every llm_client.BaseLLMClient implementation has generate_stream —
    # GeminiClient overrides it with real token streaming; everything else
    # (OllamaClient/ClaudeClient/OpenAIClient) falls back to the base
    # class's default, one non-streamed chunk via generate(). No hasattr()
    # check needed: the interface guarantees the method exists either way.
    chunks = llm_client.generate_stream(
        prompt, system_prompt=system_prompt, temperature=temperature
    )

    prefixes_upper = tuple(p.upper() for p in marker_prefixes)
    text = ""
    emitted = 0  # index into `text` up to which deltas have been flushed
    for chunk in chunks:
        text += chunk
        # max() keeps the boundary monotonic: whitespace already emitted
        # ahead of a marker line that only became recognizable later can't
        # be un-emitted (harmless — it's whitespace).
        boundary = max(emitted, _hold_start(text, prefixes_upper))
        if boundary > emitted:
            writer({"delta": text[emitted:boundary]})
            emitted = boundary

    held = text[emitted:]
    if held and not _is_marker(held, prefixes_upper):
        # Fail-open: a held tail that never turned into a real marker is
        # actual content the user should see.
        writer({"delta": held})

    return text
