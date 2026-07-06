"""
Technical Document Agent — answers questions grounded in the manufacturing
corpus via HyDE + hybrid retrieval (retrieval.py, MRR 0.489), with a
Corrective-RAG-style fallback: corpus retrieval first (free, local,
already fast); only if nothing survives the similarity threshold does it
try a web search (an actual API call, so it should be rare); only if that
also comes up empty does it fall back to an ungrounded answer from the
model's own knowledge, clearly labeled as such.

vdb, llm_client, and web_search_tool are injected via closures so this
stays swappable — pass an OllamaClient instead of GeminiClient and
nothing else here changes.
"""

import json
import re

from langsmith import traceable

from agents.state import AgentState
from retrieval import retrieve

# Sources should only ever be shown if the answer actually relied on them —
# a doc clearing the similarity threshold is a different question from
# whether the model's answer used it (a greeting or gibberish query can
# retrieve docs that pass the score cutoff by chance, without the answer
# engaging with them at all). So the model reports used_context itself
# rather than us inferring it from the retrieval score.
_JSON_INSTRUCTION = (
    ' Respond with ONLY a JSON object, no markdown fences: '
    '{"answer": "your answer text", "used_context": true or false}. '
    "used_context is true only if the provided context passages actually informed "
    "your answer. Set it false if the context was irrelevant, if the question isn't "
    "answerable from it, or if the question itself isn't a real informational "
    "question (e.g. a greeting or gibberish)."
)

SYSTEM_PROMPT_GROUNDED = (
    "You are a technical documentation expert for industrial automation and "
    "manufacturing systems (PLCs, SCADA, safety standards, industrial networking, "
    "robotics). Answer the user's question using ONLY the provided context passages. "
    "If the context doesn't contain enough information to answer confidently, say so "
    "explicitly rather than guessing. Mention which document(s) the answer draws from."
    + _JSON_INSTRUCTION
)

SYSTEM_PROMPT_WEB = (
    "You are a technical documentation expert for industrial automation and "
    "manufacturing systems. The internal corpus had nothing relevant, so you're "
    "answering from the web search results provided instead. Answer using ONLY "
    "those results, and say so if they don't actually answer the question. "
    "Mention which source(s) the answer draws from."
    + _JSON_INSTRUCTION
)

SYSTEM_PROMPT_UNGROUNDED = (
    "You are a technical documentation expert for industrial automation and "
    "manufacturing systems. Neither the internal corpus nor a web search found "
    "anything relevant to this question. Answer from your own general knowledge "
    "if you can, but say clearly and explicitly that this answer is not grounded "
    "in any retrieved source — it's the model's own knowledge, unverified."
    + _JSON_INSTRUCTION
)


def _parse_structured_answer(raw: str) -> tuple:
    """Returns (answer_text, used_context). Falls back to treating the raw
    text as the answer with used_context=True if the model didn't return
    valid JSON — fails open rather than silently hiding real sources."""
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        parsed = json.loads(cleaned)
        return parsed.get("answer", raw), bool(parsed.get("used_context", True))
    except (json.JSONDecodeError, AttributeError):
        return raw, True


def make_technical_document_agent_nodes(vdb, llm_client, web_search_tool, top_k: int = 5):
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

        prompt = f"Context:\n{context}\n\nQuestion: {state['query']}"
        raw = llm_client.generate(prompt, system_prompt=system_prompt, temperature=0.2)
        answer, used_context = _parse_structured_answer(raw)
        return {"answer": answer, "sources": sources if used_context else []}

    return retrieve_step, route_after_retrieve, web_search_step, generate_step
