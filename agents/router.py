"""
Router — decides which persona handles a query: Technical Document Agent
(general manufacturing/industry questions, grounded in the document corpus)
or PLC Expert (questions about specific PLC/Structured Text code in the
plc_simulation corpus, including best-practices/guideline checks). This is
a conditional entry point, not a node — LangGraph uses it to pick the first
real node before anything else runs.

LLM-based on purpose, not keyword-based — keyword matching already showed
real blind spots elsewhere in this system (plc_corpus_search.py's history:
an exact filename query nearly lost to noise words), and routing intent
here isn't reliably separable by keywords either — "what is a function
block" could mean either persona depending on what's actually being asked.
"""

from langsmith import traceable

ROUTER_SYSTEM_PROMPT = (
    "You route a user's question to one of two specialists. Respond with "
    "ONLY one word: either \"technical_document_agent\" or \"plc_expert\". "
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
    "If genuinely ambiguous, prefer technical_document_agent as the default."
)


def make_router(llm_client):
    @traceable(name="router")
    def route_query(state) -> str:
        raw = llm_client.generate(
            state["query"], system_prompt=ROUTER_SYSTEM_PROMPT, temperature=0.0
        )
        decision = raw.strip().lower()
        return "plc_expert" if "plc_expert" in decision else "technical_document_agent"

    return route_query
