"""
Shared structured-output mechanism: sources should only ever be shown if
the answer actually relied on them. A doc/file clearing a similarity or
keyword threshold is a different question from whether the model's answer
actually used it — a greeting, gibberish, or meta-conversation question can
retrieve content that passes the threshold by chance without the answer
engaging with it at all. So the model reports used_context itself, rather
than us inferring it from retrieval/match scores.

Originally built for Technical Document Agent; extracted here once PLC
Expert and Analytics Agent needed the same discipline (both were showing
stale keyword-matched sources on turns that actually answered from
conversation history instead — the model's own signal is what determines
that, not the fact that a search function returned something).

Uses a plain-text delimiter line, not JSON — a raw-JSON version of this
broke in practice: Gemini emits literal (unescaped) newlines inside JSON
string values for multi-paragraph/bulleted answers, which is invalid JSON,
so json.loads failed and the whole unparsed JSON blob got shown to the
user as the "answer" (a real bug this fixes, not a hypothetical one).
"""

import re

_MARKER_RE = re.compile(r"USED_CONTEXT:\s*(true|false)\s*$", re.IGNORECASE)

JSON_INSTRUCTION = (
    " End your response with exactly one line, on its own: "
    "\"USED_CONTEXT: true\" or \"USED_CONTEXT: false\" (nothing else on that "
    "line). Put your actual answer as plain text before that line — no JSON, "
    "no code fences, no other wrapping. used_context is true only if the "
    "provided context/data/file content actually informed your answer. Set "
    "it false if that context was irrelevant, if the question isn't "
    "answerable from it, if the question is really a meta-question about "
    "the conversation itself (answered from RECENT CONVERSATION instead), "
    "or if the question itself isn't a real informational question (e.g. a "
    "greeting or gibberish).\n\n"
    "Never refer to your own context sections by their internal labels — "
    "CORPUS OVERVIEW, MATCHED DATA, DATA OVERVIEW, MATCHED FILE CONTENT, "
    "RECENT CONVERSATION, ALL FUNCTION BLOCKS, BEST PRACTICES CHECK RESULT, "
    "or any variant of these. Those are internal organization for you, not "
    "vocabulary for the user — talk about what you actually know or don't "
    "know in plain language, the way a knowledgeable person would, not by "
    "naming which section of your prompt did or didn't contain something."
)


def parse_structured_answer(raw: str) -> tuple:
    """Returns (answer_text, used_context). Falls back to treating the raw
    text as the answer with used_context=True if the marker line is
    missing — fails open rather than silently hiding real sources."""
    stripped = raw.strip()
    match = _MARKER_RE.search(stripped)
    if not match:
        return raw, True
    used_context = match.group(1).lower() == "true"
    answer = stripped[: match.start()].strip()
    return answer or raw, used_context
