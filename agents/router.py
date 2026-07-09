"""
Router — decides which persona(s) handle a query: Technical Document Agent
(general manufacturing/industry questions, grounded in the document corpus),
PLC Expert (questions about specific PLC/Structured Text code in the
plc_simulation corpus, including best-practices/guideline checks),
Diagnosis Agent (questions about the synthetic energy/production telemetry
data), or Direct Reply (agents/direct_reply.py — greetings, small talk,
meta-questions about the conversation, or real questions with no
connection to any of the above; skips retrieval/tools entirely).

Direct Reply exists because of a real, measured bug: routing a query like
"sup" or "reword this" (first turn, nothing to reword) through Technical
Document Agent meant paying its full HyDE + hybrid-search + web-search +
generate pipeline — 3-4 sequential network calls — before it eventually
said "I can't answer that" ("sup" measured 36s end-to-end). The router
already makes one LLM call every turn regardless, so classifying this here
costs nothing extra, unlike adding a second gate/LLM call in front of it.

An actual graph node now, not just a conditional-entry-point function — it
needs to write its decision into state (routed_personas) so the node right
after it can act on more than one persona, not just return a single routing
key. A real, confirmed bug this fixes: "how many PLC programs do I have,
and what's my current power usage" only ever reached one persona before,
because the entry point could only pick exactly one destination. Now the
router can name more than one persona when the question genuinely has more
than one distinct part, and a multi_intent node (graph.py) runs each one
and merges their answers — instead of the question's other half being
silently dropped or answered wrong by whichever persona happened to "win."

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

_PERSONAS = ("technical_document_agent", "plc_expert", "diagnosis_agent")
_DIRECT_REPLY = "direct_reply"

ROUTER_SYSTEM_PROMPT = (
    "You route a user's question to one or more of three domain specialists, "
    "OR, when none of them actually apply, to direct_reply alone. Use the "
    "recent conversation history for context when the current question "
    "alone is ambiguous (e.g. a follow-up like \"does that follow best "
    "practices too?\" with no topic words of its own — check what the prior "
    "exchange was actually about). Respond with ONLY the relevant name(s), "
    "comma-separated if more than one: \"technical_document_agent\", "
    "\"plc_expert\", \"diagnosis_agent\", \"direct_reply\". No punctuation "
    "beyond the commas, no explanation.\n\n"
    "Name more than one specialist ONLY when the question genuinely has "
    "multiple distinct parts that each belong to a different specialist "
    "(e.g. \"how many PLC programs do I have, and what's my current power "
    "usage\" needs both plc_expert and diagnosis_agent — neither one alone "
    "covers the whole question). Don't split a question that's really just "
    "one topic phrased with multiple words — that's still exactly one "
    "specialist. direct_reply never combines with anything else — if any "
    "part of the question is a real domain question, name only the domain "
    "specialist(s), not direct_reply.\n\n"
    "technical_document_agent: general industrial automation / manufacturing "
    "questions — standards, safety, protocols, vendor products, industry "
    "concepts (e.g. \"what is IEC 62443\", \"what safety category for an "
    "e-stop circuit\"); ALSO any question about the document corpus itself "
    "— how many documents exist, what's in it, a summary/inventory/list of "
    "the files or documents available, breakdowns by source or topic (e.g. "
    "\"give me a summary of the files I have\", \"how many documents are in "
    "the corpus\", \"what documents do you have on Siemens\"). These corpus "
    "questions are real domain questions with a real answer, not a "
    "meta-question about the conversation — they must NOT go to "
    "direct_reply.\n\n"
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
    "direct_reply: greetings and small talk (\"hi\", \"sup\", \"thanks\"); "
    "meta-questions about the conversation itself (\"what did I just ask\", "
    "\"what have we covered\", \"what was the first question\") — these "
    "ALWAYS go to direct_reply, even if earlier turns touched a domain "
    "topic, because the answer comes from conversation history, not domain "
    "content; and real questions with no connection at all to industrial "
    "automation, PLC code, or energy telemetry (e.g. general grammar or "
    "trivia questions). None of these need retrieval or domain data, so "
    "routing them to a domain specialist would just waste a retrieval call "
    "before it eventually gives up.\n\n"
    "If genuinely ambiguous but plausibly domain-related even with history, "
    "prefer technical_document_agent as the default over direct_reply."
)


def _parse_personas(raw: str) -> list:
    decision = raw.strip().lower()
    found = [p for p in (*_PERSONAS, _DIRECT_REPLY) if p in decision]
    if not found:
        return ["technical_document_agent"]
    # A real domain specialist always wins over direct_reply — it should
    # only ever appear alone, but this guards against the router hedging
    # by naming it alongside a real specialist on a borderline question.
    real = [p for p in found if p != _DIRECT_REPLY]
    return real or [_DIRECT_REPLY]


def make_router_node(llm_client):
    @traceable(name="router")
    def router_node(state) -> dict:
        history_text = format_recent_history(state.get("history"), n=3, answer_chars=200)
        prompt = f"Recent conversation:\n{history_text}\n\nCurrent question: {state['query']}"

        raw = llm_client.generate(prompt, system_prompt=ROUTER_SYSTEM_PROMPT, temperature=0.0)
        return {"routed_personas": _parse_personas(raw)}

    return router_node


def pick_next_node(state) -> str:
    """Plain state read, no LLM call — the router node already decided;
    this just picks where to go based on how many personas it named."""
    personas = state.get("routed_personas") or ["technical_document_agent"]
    if len(personas) > 1:
        return "multi_intent"
    return personas[0]
