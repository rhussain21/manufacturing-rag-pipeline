"""
Shared state for the LangGraph-based agent system.

Kept minimal and additive on purpose: a plain TypedDict, not a strict
Pydantic schema, so new personas (e.g. a future data-grounded persona) can
add fields later without redesigning this or touching nodes that don't read
them.
"""

import operator
from typing import Annotated

from typing_extensions import TypedDict


class AgentState(TypedDict):
    query: str
    # Standalone form of `query`, resolved against RECENT CONVERSATION by
    # the router's own LLM call (agents/router.py). A follow-up like "so
    # answer it then" or "yes that's what I meant" carries no retrievable
    # content on its own — retrieval and generation read this field (falling
    # back to `query` when it's empty, e.g. for direct_reply), while `query`
    # itself stays untouched so chat history/transcripts show what the user
    # actually typed, not the model's rewrite.
    resolved_query: str
    retrieved_docs: list
    web_results: list
    answer: str
    sources: list
    # Populated only by the Analytics Agent (agents/analytics_agent.py) when
    # its real-code chart-intent pre-pass fires AND the LLM's own CHART
    # directive agrees a chart would help — None otherwise (most personas
    # never touch this field). Plain field, no reducer: resets fresh each
    # turn like `answer`/`sources`, so a chart from a prior turn never
    # silently persists onto an unrelated later one.
    chart_spec: dict | None
    # Written by the router node, read by the conditional edge right after
    # it to pick the next node(s). A list, not a single string — a compound
    # question ("how many PLC programs, and what's my power usage") needs
    # more than one persona to actually answer it, not just whichever one
    # the router happens to pick first.
    routed_personas: list
    # Annotated[..., operator.add] is a LangGraph reducer: instead of each
    # turn's return value overwriting this field, it gets appended to
    # whatever the checkpointer already has for this thread. Every other
    # field above resets fresh each turn (retrieved_docs/web_results are
    # scratch space for a single question) — history is the one field
    # that's meant to survive and grow across turns within a thread.
    history: Annotated[list, operator.add]
