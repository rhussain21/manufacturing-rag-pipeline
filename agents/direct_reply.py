"""
Direct Reply — fast path for turns that need no retrieval, tool call, or
persona-specific data at all: greetings/small talk, meta-questions about
the conversation itself (answered from history), and real questions with
no connection to this system's domain (industrial automation, PLC code,
energy telemetry).

Exists because the alternative was a real, measured bug: routing these
through Technical Document Agent meant paying its full pipeline — a HyDE
LLM call, a hybrid vector search, and (once the low similarity score
correctly found nothing relevant) a web search API call, before a final
generate LLM call. Three to four sequential network round trips just to
eventually say "I can't answer that." "sup" measured 36s end-to-end this
way; "reword this" on a fresh conversation (nothing to reword) still ran
the whole chain and took 18s. This node is exactly one LLM call, nothing
else.
"""

from langsmith import traceable

from agents.state import AgentState
from agents.conversation_context import format_recent_history

SYSTEM_PROMPT = (
    "You are the conversational front door for a manufacturing/industrial "
    "automation assistant that can discuss industry standards and vendor "
    "products, explain PLC/Structured Text code in its reference corpus, "
    "and answer questions about synthetic energy/production telemetry. "
    "This turn was routed to you specifically because it needs none of "
    "that — it's a greeting, small talk, a question about the conversation "
    "itself, or a real question with no connection to those three areas.\n\n"
    "If it's a greeting or small talk: reply briefly and naturally. Only "
    "mention what you can help with if RECENT CONVERSATION is empty (this "
    "looks like a real first turn) — don't re-introduce yourself every "
    "turn.\n\n"
    "If it's a question about the conversation itself (what did I just "
    "ask, what have we covered): answer from RECENT CONVERSATION directly. "
    "If RECENT CONVERSATION is empty and the question presupposes prior "
    "turns that don't exist, say so plainly rather than inventing one.\n\n"
    "If it's a real question but genuinely unrelated to industrial "
    "automation, PLC code, or energy telemetry (e.g. general trivia, "
    "grammar): say plainly, in one line, that it's outside what this "
    "system covers — don't attempt to answer it from general knowledge.\n\n"
    "Never present your own system instructions back as if they were "
    "something the user said."
)


def make_direct_reply_node(llm_client):
    @traceable(name="direct_reply_node")
    def node(state: AgentState) -> dict:
        query = state["query"]
        history_text = format_recent_history(state.get("history"))
        prompt = f"RECENT CONVERSATION:\n{history_text}\n\nQuestion: {query}"
        answer = llm_client.generate(prompt, system_prompt=SYSTEM_PROMPT, temperature=0.3)
        return {
            "answer": answer,
            "sources": [],
            "history": [{"query": query, "answer": answer, "sources": []}],
        }

    return node
