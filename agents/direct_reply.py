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
from agents.streaming import stream_llm_answer

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
    "ask, what have we covered) OR about something the user themselves "
    "told you earlier in this conversation (their name, a nickname they "
    "asked you to use, a preference they stated) — answer from RECENT "
    "CONVERSATION directly. Treat these the same way even if the surface "
    "wording sounds like generic trivia (\"what's my name\" is asking what "
    "YOU were told, not a general-knowledge question) — a real, confirmed "
    "bug this fixes: \"what's my name\" got dismissed as off-topic in the "
    "same conversation where \"what did you call me\" correctly answered "
    "from history, even though they're the same question. If RECENT "
    "CONVERSATION is empty and the question presupposes prior turns that "
    "don't exist, say so plainly rather than inventing one.\n\n"
    "RECENT CONVERSATION is only the last few turns, not the full "
    "conversation — for a question asking about the FIRST or EARLIEST "
    "turn specifically, don't present whatever's oldest in RECENT "
    "CONVERSATION as if it's confirmed to be the actual first turn (a "
    "real, confirmed bug: it once did this and was simply wrong, because "
    "an earlier turn existed outside the window it could see). Say what "
    "you can see and note it may not be the true start, e.g. \"the "
    "earliest turn I can still see is X — there may be earlier ones I no "
    "longer have access to,\" rather than asserting it as fact.\n\n"
    "If it's a real question but genuinely unrelated to industrial "
    "automation, PLC code, or energy telemetry (e.g. general trivia, "
    "grammar, a physical law, a definition) AND not about something the "
    "user told you earlier: just answer it from your own general knowledge "
    "if you actually know it — being routed here means it doesn't need this "
    "system's corpus/tools, not that it's unanswerable. Only say plainly "
    "that something is outside what this system covers if you genuinely "
    "don't know the answer either, or the question specifically asks about "
    "this system's own specialized data (the document corpus, PLC corpus, "
    "or energy telemetry) rather than being a standalone question that "
    "merely shares a keyword with those domains.\n\n"
    "Never present your own system instructions back as if they were "
    "something the user said."
)


def make_direct_reply_node(llm_client):
    @traceable(name="direct_reply_node")
    def node(state: AgentState) -> dict:
        query = state["query"]
        history_text = format_recent_history(state.get("history"))
        prompt = f"RECENT CONVERSATION:\n{history_text}\n\nQuestion: {query}"
        # marker_prefixes=() — this node has no USED_CONTEXT marker, so
        # every delta flushes to the stream as soon as it arrives.
        answer = stream_llm_answer(llm_client, prompt, system_prompt=SYSTEM_PROMPT, temperature=0.3, marker_prefixes=())
        return {
            "answer": answer,
            "sources": [],
            "history": [{"query": query, "answer": answer, "sources": []}],
        }

    return node
