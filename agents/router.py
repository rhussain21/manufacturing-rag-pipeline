"""
Router — decides which persona(s) handle a query: Technical Document Agent
(general manufacturing/industry questions, grounded in the document corpus),
PLC Expert (questions about specific PLC/Structured Text code in the
plc_simulation corpus, including best-practices/guideline checks),
Analytics Agent (questions about the synthetic energy/production telemetry
data, including real-code-computed charts), or Direct Reply
(agents/direct_reply.py — greetings, small talk,
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

Also resolves the current turn into a standalone query (`resolved_query`),
in this same LLM call rather than a separate one. Real, confirmed bug this
fixes: retrieval (agents/technical_document_agent.py's retrieve_step) reads
state["query"] and only ever saw the literal current-turn text — for a
follow-up like "so answer it then" or "yes that's what I meant", that
literal text carries no retrievable content at all, so retrieval ran
against "so answer it then" instead of whatever the actual question was,
and came back empty even though the corpus had the answer. Bundled into
the router's existing call (not a new one) because the router already
makes one LLM call every turn regardless — same reasoning as everything
else in this file.
"""

from langsmith import traceable

from agents.conversation_context import format_recent_history

_PERSONAS = ("technical_document_agent", "plc_expert", "analytics_agent")
_DIRECT_REPLY = "direct_reply"

ROUTER_SYSTEM_PROMPT = (
    "You do two things with the current question, using recent conversation "
    "history for context when the question alone is ambiguous:\n\n"
    "1. Resolve it into a standalone query. If it depends on prior turns to "
    "mean anything — a confirmation (\"yes that's what I meant\"), a "
    "correction (\"no I meant 21 CFR part 11\"), a continuation (\"so answer "
    "it then\"), or a pronoun/reference (\"does that follow best practices "
    "too?\") — rewrite it as the real, self-contained question being asked, "
    "using the topic from recent history. If it's already standalone (or is "
    "a greeting/small talk/meta-question with nothing to resolve), repeat it "
    "unchanged. This resolved form is what actually gets searched against "
    "the document corpus, so it must contain the real subject matter, not "
    "just the surface words of the current turn.\n\n"
    "2. Route it to one or more of three domain specialists, OR, when none "
    "of them actually apply, to direct_reply alone. A confirmation or "
    "correction of a previous domain question (\"yes that's what I meant\", "
    "\"so answer it then\", \"no I meant X\") ALWAYS routes to whichever "
    "specialist handled that previous question, never to direct_reply — "
    "direct_reply cannot search the corpus or continue the answer, so "
    "routing a confirmation there silently drops the question.\n\n"
    "Respond in exactly this two-line format, no other text:\n"
    "QUERY: <the resolved standalone question>\n"
    "ROUTE: <name(s), comma-separated if more than one>\n\n"
    "Valid names for ROUTE: \"technical_document_agent\", \"plc_expert\", "
    "\"analytics_agent\", \"direct_reply\". No punctuation beyond the commas, "
    "no explanation.\n\n"
    "Name more than one specialist ONLY when the question genuinely has "
    "multiple distinct parts that each belong to a different specialist "
    "(e.g. \"how many PLC programs do I have, and what's my current power "
    "usage\" needs both plc_expert and analytics_agent — neither one alone "
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
    "analytics_agent: questions about the synthetic energy/production "
    "telemetry data — battery storage, grid power, production/consumption, "
    "anomalies at specific sites (Willowbrook, Meridian, Northgate), or "
    "anomaly types (battery_capacity_fade, sensor_dropout, demand_spike, "
    "grid_instability, production_underperformance) (e.g. \"what happened at "
    "Willowbrook\", \"what does grid instability look like in the data\", "
    "\"how many anomalies are there\", \"compare production across sites\", "
    "\"chart battery state of charge over time\").\n\n"
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


def _parse_personas(route_line: str) -> list:
    decision = route_line.strip().lower()
    found = [p for p in (*_PERSONAS, _DIRECT_REPLY) if p in decision]
    if not found:
        return ["technical_document_agent"]
    # A real domain specialist always wins over direct_reply — it should
    # only ever appear alone, but this guards against the router hedging
    # by naming it alongside a real specialist on a borderline question.
    real = [p for p in found if p != _DIRECT_REPLY]
    return real or [_DIRECT_REPLY]


def _parse_router_output(raw: str, fallback_query: str) -> tuple:
    """Splits the router's two-line QUERY:/ROUTE: response. Falls back to
    the raw current-turn query and a whole-response persona scan if the
    model didn't follow the format — same tolerance-for-drift approach
    _parse_personas already took before this, since matching exact LLM
    output formatting isn't worth a hard failure over."""
    query_line, route_line = None, None
    for line in raw.splitlines():
        stripped = line.strip()
        if query_line is None and stripped.lower().startswith("query:"):
            query_line = stripped[len("query:"):].strip()
        elif route_line is None and stripped.lower().startswith("route:"):
            route_line = stripped[len("route:"):].strip()

    resolved_query = query_line or fallback_query
    personas = _parse_personas(route_line if route_line is not None else raw)
    return resolved_query, personas


def make_router_node(llm_client):
    @traceable(name="router")
    def router_node(state) -> dict:
        history_text = format_recent_history(state.get("history"), n=3, answer_chars=200)
        prompt = f"Recent conversation:\n{history_text}\n\nCurrent question: {state['query']}"

        raw = llm_client.generate(prompt, system_prompt=ROUTER_SYSTEM_PROMPT, temperature=0.0)
        resolved_query, personas = _parse_router_output(raw, state["query"])
        return {"routed_personas": personas, "resolved_query": resolved_query}

    return router_node


def pick_next_node(state) -> str:
    """Plain state read, no LLM call — the router node already decided;
    this just picks where to go based on how many personas it named."""
    personas = state.get("routed_personas") or ["technical_document_agent"]
    if len(personas) > 1:
        return "multi_intent"
    return personas[0]
