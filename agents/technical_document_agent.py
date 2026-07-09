"""
Technical Document Agent — answers questions grounded in the manufacturing
corpus via HyDE + hybrid retrieval (retrieval.py, MRR 0.489), with a
Corrective-RAG-style fallback: corpus retrieval first (free, local,
already fast); only if nothing survives the similarity threshold does it
try a web search (an actual API call, so it should be rare); only if that
also comes up empty does it fall back to an ungrounded answer from the
model's own knowledge, clearly labeled as such.

vdb, llm_client, web_search_tool, and db are injected via closures so this
stays swappable — pass an OllamaClient instead of GeminiClient and
nothing else here changes. db (a relationalDB) is used only for a
deterministic corpus-inventory query, not for retrieval itself.
"""

from langsmith import traceable

from agents.state import AgentState
from agents.conversation_context import format_recent_history
from agents.structured_answer import JSON_INSTRUCTION, parse_structured_answer
from retrieval import retrieve

# Without this, a meta-question ("what did I just ask", "what have we
# covered") has no real transcript to draw from, and the model will guess
# rather than admit it — a real bug found in testing: it pattern-matched
# onto its own system prompt text and presented that back as the user's
# first question, since that was the only "meta" text available to it.
_HISTORY_INSTRUCTION = (
    " You're also given RECENT CONVERSATION — real prior turns in this "
    "session. If the question is about the conversation itself (what did I "
    "just ask, what have we covered, summarize this conversation), answer "
    "from that directly rather than from the corpus context. Don't answer "
    "a meta-question about the conversation using retrieved documents, and "
    "don't confuse your own system instructions with something the user "
    "said."
)

# A real, confirmed bug: "give me a summary of the files i have" got
# answered from whatever docs were semantically nearest (e.g. unrelated
# Mitsubishi Electric content) instead of a real inventory — HyDE + hybrid
# retrieval always returns *something*, even for a question that isn't
# actually about any single document's content. CORPUS INVENTORY is a real
# aggregate computed from the DB (relationalDB.get_corpus_inventory, same
# filter workflows/vectorize_lance.py uses to decide what's actually
# searchable), always attached below regardless of what retrieval found —
# same pattern PLC Expert already uses for its own corpus overview.
# Placed BEFORE the "answer using ONLY the provided context passages" rule
# below, and phrased as a decision to make first — an earlier version
# appended this after that rule instead, and in testing the model kept
# answering "summary of the files I have" from the retrieved passages
# anyway, treating the blanket "ONLY context passages" instruction as the
# controlling one. Telling it to decide the question type FIRST, before
# that rule even applies, fixed it.
_INVENTORY_INSTRUCTION = (
    " FIRST, decide what kind of question this is. You're given CORPUS "
    "INVENTORY — real, deterministic counts of what's in the searchable "
    "corpus (by content type, document type, source, and topic). If the "
    "question asks how many documents exist, what's in the corpus, or for "
    "a summary/inventory/list of the files or documents available (e.g. "
    "\"give me a summary of the files I have\", \"how many documents do you "
    "have\", \"what's in the corpus\") — answer ONLY from CORPUS INVENTORY, "
    "and ignore the retrieved context passages below entirely, even if they "
    "look topically related. The retrieved passages are just the few docs "
    "that scored highest on semantic similarity to the question text, not "
    "a real count or list, and using them for an inventory question is "
    "wrong even when one of them happens to mention files or documents. "
    "Otherwise, for a real content question, follow the context-passage "
    "rule below."
)

SYSTEM_PROMPT_GROUNDED = (
    "You are a technical documentation expert for industrial automation and "
    "manufacturing systems (PLCs, SCADA, safety standards, industrial networking, "
    "robotics)."
    + _INVENTORY_INSTRUCTION +
    " Answer the user's question using ONLY the provided context passages. "
    "If the context doesn't contain enough information to answer confidently, say so "
    "explicitly rather than guessing. Mention which document(s) the answer draws from."
    + JSON_INSTRUCTION + _HISTORY_INSTRUCTION
)

SYSTEM_PROMPT_WEB = (
    "You are a technical documentation expert for industrial automation and "
    "manufacturing systems. The internal corpus had nothing relevant, so you're "
    "answering from the web search results provided instead."
    + _INVENTORY_INSTRUCTION +
    " Otherwise, answer using ONLY those results, and say so if they don't actually "
    "answer the question. Mention which source(s) the answer draws from."
    + JSON_INSTRUCTION + _HISTORY_INSTRUCTION
)

SYSTEM_PROMPT_UNGROUNDED = (
    "You are a technical documentation expert for industrial automation and "
    "manufacturing systems. Neither the internal corpus nor a web search found "
    "anything relevant to this question. Answer from your own general knowledge "
    "if you can, but say clearly and explicitly that this answer is not grounded "
    "in any retrieved source — it's the model's own knowledge, unverified."
    + JSON_INSTRUCTION + _HISTORY_INSTRUCTION + _INVENTORY_INSTRUCTION
)


def _format_inventory(inv: dict) -> str:
    lines = [f"{inv['total']} documents currently in the searchable corpus."]
    if inv["by_content_type"]:
        lines.append("By content type: " + ", ".join(f"{v} {k}" for k, v in inv["by_content_type"].items()))
    if inv["by_doc_type"]:
        lines.append("By document type: " + ", ".join(f"{v} {k}" for k, v in inv["by_doc_type"].items()))
    if inv["by_source"]:
        lines.append("Top sources: " + ", ".join(f"{v} {k}" for k, v in inv["by_source"].items()))
    if inv["top_topics"]:
        lines.append("Top topics: " + ", ".join(f"{v} {k}" for k, v in inv["top_topics"].items()))
    return "\n".join(lines)


def make_technical_document_agent_nodes(vdb, llm_client, web_search_tool, db, top_k: int = 5):
    @traceable(name="retrieve_step")
    def retrieve_step(state: AgentState) -> dict:
        docs = retrieve(state["query"], vdb, llm_client, top_k=top_k)
        return {"retrieved_docs": docs}

    def route_after_retrieve(state: AgentState) -> str:
        return "generate" if state["retrieved_docs"] else "web_search"

    @traceable(name="web_search_step")
    def web_search_step(state: AgentState) -> dict:
        result = web_search_tool.search(state["query"], max_results=5)
        return {"web_results": result.get("results", [])}

    @traceable(name="generate_step")
    def generate_step(state: AgentState) -> dict:
        docs = state.get("retrieved_docs") or []
        web_results = state.get("web_results") or []

        if docs:
            context = "\n\n---\n\n".join(
                f"[{d['metadata'].get('title', 'Untitled')}]\n{d['document'][:2000]}"
                for d in docs
            )
            system_prompt = SYSTEM_PROMPT_GROUNDED
            sources = [
                {"content_id": d["metadata"].get("content_id"), "title": d["metadata"].get("title")}
                for d in docs
            ]
        elif web_results:
            context = "\n\n---\n\n".join(
                f"[{r['title']}]\n{r['snippet']}" for r in web_results
            )
            system_prompt = SYSTEM_PROMPT_WEB
            sources = [
                {"content_id": None, "title": r["title"], "url": r.get("url")}
                for r in web_results
            ]
        else:
            context = "No relevant documents were found in the corpus, and web search found nothing either."
            system_prompt = SYSTEM_PROMPT_UNGROUNDED
            sources = []

        inventory_text = _format_inventory(db.get_corpus_inventory())
        history_text = format_recent_history(state.get("history"))
        prompt = (
            f"RECENT CONVERSATION:\n{history_text}\n\n"
            f"CORPUS INVENTORY:\n{inventory_text}\n\n"
            f"Context:\n{context}\n\nQuestion: {state['query']}"
        )
        raw = llm_client.generate(prompt, system_prompt=system_prompt, temperature=0.2)
        answer, used_context = parse_structured_answer(raw)
        final_sources = sources if used_context else []
        return {
            "answer": answer,
            "sources": final_sources,
            "history": [{"query": state["query"], "answer": answer, "sources": final_sources}],
        }

    return retrieve_step, route_after_retrieve, web_search_step, generate_step
