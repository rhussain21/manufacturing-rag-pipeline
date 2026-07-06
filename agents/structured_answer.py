"""
Shared structured-output mechanism: sources should only ever be shown if
the answer actually relied on them. A doc/file clearing a similarity or
keyword threshold is a different question from whether the model's answer
actually used it — a greeting, gibberish, or meta-conversation question can
retrieve content that passes the threshold by chance without the answer
engaging with it at all. So the model reports used_context itself, rather
than us inferring it from retrieval/match scores.

Originally built for Technical Document Agent; extracted here once PLC
Expert and Diagnosis Agent needed the same discipline (both were showing
stale keyword-matched sources on turns that actually answered from
conversation history instead — the JSON output is what determines
that, not the fact that a search function returned something).
"""

import json
import re

JSON_INSTRUCTION = (
    ' Respond with ONLY a JSON object, no markdown fences: '
    '{"answer": "your answer text", "used_context": true or false}. '
    "used_context is true only if the provided context/data/file content actually "
    "informed your answer. Set it false if that context was irrelevant, if the "
    "question isn't answerable from it, if the question is really a meta-question "
    "about the conversation itself (answered from RECENT CONVERSATION instead), or "
    "if the question itself isn't a real informational question (e.g. a greeting "
    "or gibberish)."
)


def parse_structured_answer(raw: str) -> tuple:
    """Returns (answer_text, used_context). Falls back to treating the raw
    text as the answer with used_context=True if the model didn't return
    valid JSON — fails open rather than silently hiding real sources."""
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        parsed = json.loads(cleaned)
        return parsed.get("answer", raw), bool(parsed.get("used_context", True))
    except (json.JSONDecodeError, AttributeError):
        return raw, True
