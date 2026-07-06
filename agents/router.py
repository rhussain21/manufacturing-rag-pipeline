"""
Router — decides which persona handles a query: Technical Document Agent
(general manufacturing/industry questions, grounded in the document corpus),
PLC Expert (questions about specific PLC/Structured Text code in the
plc_simulation corpus, including best-practices/guideline checks), or
Diagnosis Agent (questions about the synthetic energy/production telemetry
data). This is a conditional entry point, not a node — LangGraph uses it to
pick the first real node before anything else runs.

LLM-based on purpose, not keyword-based — keyword matching already showed
real blind spots elsewhere in this system (plc_corpus_search.py's history:
an exact filename query nearly lost to noise words), and routing intent
here isn't reliably separable by keywords either — "what is a function
block" could mean either persona depending on what's actually being asked.

Sees recent conversation history (from the checkpointer), not just the bare
current-turn query — a follow-up like "does that follow best practices
too?" has no PLC-specific words in it at all, and would misroute without
knowing the previous exchange was about a specific PLC file.
"""

from langsmith import traceable

from agents.conversation_context import format_recent_history

ROUTER_SYSTEM_PROMPT = (
    "You route a user's question to one of three specialists, using the "
    "recent conversation history for context when the current question "
    "alone is ambiguous (e.g. a follow-up like \"does that follow best "
    "practices too?\" with no topic words of its own — check what the prior "
    "exchange was actually about). Respond with ONLY one word: "
    "\"technical_document_agent\", \"plc_expert\", or \"diagnosis_agent\". "
    "No punctuation, no explanation.\n\n"
    "technical_document_agent: general industrial automation / manufacturing "
    "questions — standards, safety, protocols, vendor products, industry "
    "concepts (e.g. \"what is IEC 62443\", \"what safety category for an "
    "e-stop circuit\").\n\n"
    "plc_expert: questions specifically about PLC / Structured Text code in "
    "our reference corpus — explaining what a specific program or function "
    "block does, summarizing what code exists, checking code against "
    "PLCopen best practices/coding guidelines, or PLC programming concepts "
    "grounded in actual code examples (e.g. \"what does FB_EL3423 do\", "
    "\"summarize the PLC code we have\", \"does this follow best "
    "practices\").\n\n"
    "diagnosis_agent: questions about the synthetic energy/production "
    "telemetry data — battery storage, grid power, production/consumption, "
    "anomalies at specific sites (Willowbrook, Meridian, Northgate), or "
    "anomaly types (battery_capacity_fade, sensor_dropout, demand_spike, "
    "grid_instability, production_underperformance) (e.g. \"what happened at "
    "Willowbrook\", \"what does grid instability look like in the data\", "
    "\"how many anomalies are there\").\n\n"
    "If genuinely ambiguous even with history, prefer technical_document_agent "
    "as the default."
)


def make_router(llm_client):
    @traceable(name="router")
    def route_query(state) -> str:
        history_text = format_recent_history(state.get("history"), n=3, answer_chars=200)
        prompt = f"Recent conversation:\n{history_text}\n\nCurrent question: {state['query']}"

        raw = llm_client.generate(prompt, system_prompt=ROUTER_SYSTEM_PROMPT, temperature=0.0)
        decision = raw.strip().lower()
        if "diagnosis_agent" in decision:
            return "diagnosis_agent"
        if "plc_expert" in decision:
            return "plc_expert"
        return "technical_document_agent"

    return route_query
