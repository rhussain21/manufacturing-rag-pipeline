"""
Shared state for the LangGraph-based agent system.

Kept minimal and additive on purpose: a plain TypedDict, not a strict
Pydantic schema, so new personas (e.g. a future Diagnosis Agent) can add
fields later without redesigning this or touching nodes that don't read
them.
"""

from typing_extensions import TypedDict


class AgentState(TypedDict):
    query: str
    retrieved_docs: list
    web_results: list
    answer: str
    sources: list
