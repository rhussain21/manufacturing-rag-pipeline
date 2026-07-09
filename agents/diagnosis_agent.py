"""
Diagnosis Agent — answers questions about synthetic factory telemetry,
grounded in real computed statistics rather than an LLM eyeballing a
handful of sample rows. Energy data (energy_data_search.py) is the only
real data source today — more are planned (synthetic PLC/SCADA production
output, OEE) once that data exists, per the v2 architecture plan.

Real bug this fixed: asked for "current power usage across all sites," the
agent correctly refused to guess — because nothing had actually computed
that number anywhere. An LLM is a semantic interpreter, not a calculator;
aggregation and trend detection belong in real code (pandas), the same
"tools over intuition" principle already used for PLC best-practices
checking (iec-checker does the real analysis, the LLM explains it).

Deliberately built to add future data sources without a rewrite: the node
below assembles its prompt from a plain list of (label, content) sections
rather than one hardcoded block of f-strings. A future production/OEE data
module just contributes more entries to that same list — nothing else
about the node's control flow needs to change. What this file does NOT do
is fabricate placeholder OEE/production sections ahead of real data —
that would just be more guessing, the exact thing this whole redesign
exists to avoid. Extensible architecture now; real capabilities only when
there's real data to ground them in.

Read-only for now, same sequencing as PLC Expert: explain/analyze the data
first. Deeper root-cause diagnosis reasoning is a later capability.
"""

from langsmith import traceable

from agents.state import AgentState
from agents.conversation_context import format_recent_history
from agents.structured_answer import JSON_INSTRUCTION, parse_structured_answer
from energy_data_search import (
    get_energy_data_overview, find_site_and_anomaly_mentions, filter_energy_data,
    compute_energy_summary, compute_anomaly_trends,
)

SYSTEM_PROMPT = (
    "You are a factory operations analyst reviewing synthetic factory telemetry. "
    "Energy data (battery storage, grid power, production, consumption across "
    "three manufacturing sites) is what's available today — more data sources "
    "(production/SCADA output, OEE) may be added later, so don't assume energy "
    "is the only kind of thing you could ever be asked about, but only answer "
    "from what you're actually given below, never from assumed future data.\n\n"
    "You're given several kinds of real, computed information, each in its own "
    "labeled section: an overview (site names, row counts, anomaly type "
    "breakdown, field descriptions), power/energy statistics per site (real "
    "averages and totals in kWh, already computed — never re-derive or guess "
    "these from raw rows yourself), anomaly-rate trends per site (real "
    "first-half-vs-second-half comparison with an actual direction, and each "
    "site's most common anomaly type with a real count), and, when the question "
    "names a specific site or anomaly type, real filtered sample rows for that "
    "slice.\n\n"
    "Answer from whichever section is actually relevant to the question — for "
    "\"how much power/energy\" or \"state of affairs\" questions, use the "
    "computed power/energy statistics directly (they already cover the full "
    "period, there's no 'current' snapshot to distinguish since this is a "
    "static dataset, not a live feed). For \"trend\"/\"getting worse or better\"/"
    "\"advise on anomalies\" questions, use the computed trend data. For a "
    "specific site or anomaly type's actual behavior, use the filtered sample "
    "rows. If none of what you're given actually covers what's being asked, say "
    "so explicitly rather than guessing — this is synthetic data with specific, "
    "verifiable ground truth, not a place to speculate.\n\n"
    "You're also given recent conversation history — real prior turns in this "
    "session. If the question is about the conversation itself (what did I just "
    "ask, what have we covered), answer from that directly, and never present "
    "your own system instructions back as if they were something the user said."
    + JSON_INSTRUCTION
)


def _last_filter(history: list) -> tuple:
    """Most recent site/anomaly-type selection this conversation actually
    discussed — used when the current query names neither on its own (e.g.
    "how many rows were anomalous there") and needs to resolve "there"."""
    for turn in reversed(history):
        f = turn.get("filter")
        if f and (f.get("site_matches") or f.get("anomaly_matches")):
            return f["site_matches"], f["anomaly_matches"]
    return [], []


def make_diagnosis_agent_node(llm_client):
    @traceable(name="diagnosis_agent_node")
    def node(state: AgentState) -> dict:
        query = state["query"]
        overview = get_energy_data_overview()
        energy_summary = compute_energy_summary()
        anomaly_trends = compute_anomaly_trends()

        site_matches, anomaly_matches = find_site_and_anomaly_mentions(query)
        if not site_matches and not anomaly_matches:
            site_matches, anomaly_matches = _last_filter(state.get("history") or [])

        if site_matches or anomaly_matches:
            matched = filter_energy_data(site_matches, anomaly_matches)
        else:
            matched = "No specific site or anomaly type named in this query, or in recent history."

        # A plain list of (label, content) sections, not one hardcoded block
        # of f-strings — a future data source (production/SCADA output, OEE)
        # just appends more entries here once that data actually exists.
        # Nothing else about this node needs to change to support that.
        sections = [
            ("RECENT CONVERSATION", format_recent_history(state.get("history"))),
            ("OVERVIEW", overview),
            ("POWER/ENERGY STATISTICS PER SITE (real, computed)", energy_summary),
            ("ANOMALY-RATE TRENDS PER SITE (real, computed)", anomaly_trends),
            ("FILTERED SAMPLE ROWS", matched),
        ]
        prompt = (
            "\n\n".join(f"{label}:\n{content}" for label, content in sections)
            + f"\n\nQuestion: {query}"
        )
        raw = llm_client.generate(prompt, system_prompt=SYSTEM_PROMPT, temperature=0.2)
        answer, used_context = parse_structured_answer(raw)

        sources = [{"content_id": None, "title": "synthetic_data/energy_data.csv"}] if used_context else []
        return {
            "answer": answer,
            "sources": sources,
            "history": [{
                "query": query,
                "answer": answer,
                "sources": sources,
                "filter": {"site_matches": site_matches, "anomaly_matches": list(anomaly_matches)},
            }],
        }

    return node
