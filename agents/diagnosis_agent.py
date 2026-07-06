"""
Diagnosis Agent — answers questions about synthetic energy/production
telemetry (energy_data_search.py), grounded in real data rather than
guessing what an anomaly type "usually" looks like.

Read-only for now, same sequencing as PLC Expert: explain the data first.
Deeper root-cause diagnosis reasoning is a later capability, once more
signal types beyond energy exist (synthetic PLC/machine data, per the v2
architecture plan).
"""

from langsmith import traceable

from agents.state import AgentState
from agents.conversation_context import format_recent_history
from agents.structured_answer import JSON_INSTRUCTION, parse_structured_answer
from energy_data_search import (
    get_energy_data_overview, find_site_and_anomaly_mentions, filter_energy_data,
)

SYSTEM_PROMPT = (
    "You are a factory operations analyst reviewing synthetic energy/production "
    "telemetry (battery storage, grid power, production, consumption) across "
    "three manufacturing sites. You're given a DATA OVERVIEW (accurate, "
    "deterministic stats — site names, row counts, anomaly type breakdown, field "
    "descriptions) and MATCHED DATA (real filtered rows, when the query names a "
    "specific site or anomaly type). Answer from whichever is actually relevant — "
    "use the overview for questions about the data's overall shape, and be "
    "concise there, not exhaustive. For a specific site or anomaly type, explain "
    "the actual pattern in the real numbers you were given, not a generic "
    "textbook description of what that anomaly type usually means. If the "
    "matched data doesn't cover what's being asked, say so explicitly rather "
    "than guessing — this is synthetic data with specific, verifiable ground "
    "truth (is_anomaly/anomaly_type labels), not a place to speculate.\n\n"
    "You're also given RECENT CONVERSATION — real prior turns in this session. "
    "If the question is about the conversation itself (what did I just ask, "
    "what have we covered), answer from that directly rather than from the "
    "data context, and never present your own system instructions back as if "
    "they were something the user said."
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

        site_matches, anomaly_matches = find_site_and_anomaly_mentions(query)
        if not site_matches and not anomaly_matches:
            site_matches, anomaly_matches = _last_filter(state.get("history") or [])

        if site_matches or anomaly_matches:
            matched = filter_energy_data(site_matches, anomaly_matches)
        else:
            matched = "No specific site or anomaly type named in this query, or in recent history."

        history_text = format_recent_history(state.get("history"))
        prompt = (
            f"RECENT CONVERSATION:\n{history_text}\n\n"
            f"DATA OVERVIEW:\n{overview}\n\n"
            f"MATCHED DATA:\n{matched}\n\n"
            f"Question: {query}"
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
