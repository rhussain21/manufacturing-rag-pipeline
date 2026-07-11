"""
Chart directive — parallel structure to agents/structured_answer.py (same
JSON_INSTRUCTION + parse_structured_answer split), but for whether/how to
render an inline chart alongside an answer.

Deliberately fails CLOSED, the opposite of parse_structured_answer's
fail-open USED_CONTEXT handling: a missing/wrong USED_CONTEXT risks hiding a
real answer (harmful), a missing/wrong CHART directive just skips an
optional visualization (harmless) — so any malformed input here just means
no chart, never a hard error and never a guessed chart type.

The LLM only ever picks a chart TYPE (bar/line/none) — it never picks what
data goes in the chart. That's decided by real code before the LLM call
(agents/analytics_agent.py's _detect_chart_intent + compute_grouped_series)
specifically so the numbers narrated in prose and the numbers plotted can
never drift apart the way they could if the LLM were also choosing the
data. Forbidding invented numbers is enforced by CHART_INSTRUCTION's wording,
not by anything this parser checks — the parser only ever strips the
CHART: line and returns yes/no + type.
"""

import re

_CHART_RE = re.compile(r"^CHART:\s*(yes|none)\s*(?:\|\s*(bar|line))?\s*$", re.IGNORECASE | re.MULTILINE)

# Appended to a persona's system prompt, immediately BEFORE the
# USED_CONTEXT instruction (agents/structured_answer.JSON_INSTRUCTION) — the
# node parses structured_answer's marker first (last line), then this one
# (now the new last line) on what's left, so no change is needed to
# structured_answer.py itself.
CHART_INSTRUCTION = (
    " If a GROUPED DATA section is present above, end your response with one "
    "more line, on its own, immediately before the USED_CONTEXT line: "
    "\"CHART: yes | bar\" or \"CHART: yes | line\" if a chart of that data "
    "would genuinely help answer the question, or \"CHART: none\" if it "
    "wouldn't (e.g. the question only needed one number, not a comparison). "
    "Use \"line\" for a trend over time (daily/hourly), \"bar\" for a "
    "comparison across sites or categories. If no GROUPED DATA section is "
    "present, always write \"CHART: none\" — never claim a chart when no "
    "data was computed for one. This CHART line only picks the chart type; "
    "it never invents numbers of its own — the chart is built entirely from "
    "the GROUPED DATA you were already given."
)


def parse_chart_directive(raw: str) -> tuple:
    """Returns (remaining_text, wants_chart, chart_type). Strips the CHART:
    line from the text regardless of outcome. Fails closed: any missing or
    malformed directive (no match, "none" with no type is valid, anything
    else off-grammar) resolves to wants_chart=False, chart_type=None."""
    match = _CHART_RE.search(raw)
    if not match:
        return raw.strip(), False, None

    remaining = (raw[: match.start()] + raw[match.end():]).strip()
    decision = match.group(1).lower()
    chart_type = match.group(2).lower() if match.group(2) else None

    if decision != "yes" or chart_type not in ("bar", "line"):
        return remaining, False, None

    return remaining, True, chart_type
