"""
Shared state for the LangGraph-based agent system.

Kept minimal and additive on purpose: a plain TypedDict, not a strict
Pydantic schema, so new personas (e.g. a future Diagnosis Agent) can add
fields later without redesigning this or touching nodes that don't read
them.
"""

import operator
from typing import Annotated

from typing_extensions import TypedDict


class AgentState(TypedDict):
    query: str
    retrieved_docs: list
    web_results: list
    answer: str
    sources: list
    # Annotated[..., operator.add] is a LangGraph reducer: instead of each
    # turn's return value overwriting this field, it gets appended to
    # whatever the checkpointer already has for this thread. Every other
    # field above resets fresh each turn (retrieved_docs/web_results are
    # scratch space for a single question) — history is the one field
    # that's meant to survive and grow across turns within a thread.
    history: Annotated[list, operator.add]
