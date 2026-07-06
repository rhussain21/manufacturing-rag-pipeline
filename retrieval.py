"""
Tuned retrieval for the Technical Doc Expert agent.

Extracted from notebooks/03_retrieval_evaluation.ipynb — HyDE + hybrid
(dense + BM25 + RRF) was the best-performing config across every setup
tested (MRR 0.489, Recall@10 0.438), beating plain dense, hybrid alone,
and reranked variants. See Craft notes (2026-07-04) for the full
comparison table.
"""

from langsmith import traceable

from llm_client import GeminiClient
from db_vector_lance import LanceVectorDB

HYDE_SYSTEM_PROMPT = (
    "You are an industrial automation and manufacturing technology expert. "
    "Write a short, confident, factual-sounding paragraph (3-5 sentences) "
    "that plausibly answers the user's question, in the style of a "
    "technical reference document or vendor whitepaper. It's fine if some "
    "details are approximate — the goal is realistic document phrasing, "
    "not guaranteed accuracy."
)


@traceable(name="generate_hyde_passage")
def generate_hyde_passage(query: str, llm_client: GeminiClient) -> str:
    """Generate a hypothetical answer passage to search with instead of the raw query.

    Closes the phrasing gap between short questions and the long technical
    passages that answer them — a generated answer resembles real documents
    (vocabulary, structure, length) far more than the raw question does.
    """
    return llm_client.generate(query, system_prompt=HYDE_SYSTEM_PROMPT, temperature=0.3)


def aggregate_to_docs(chunk_hits: list, score_key: str = "similarity") -> list:
    """Dedupe chunk-level hits down to one best-scoring hit per document.

    score_key lets the same function aggregate either raw similarity results
    or reranker-scored results (score_key="reranker_score").
    """
    best = {}
    for chunk in chunk_hits:
        cid = chunk["metadata"]["content_id"]
        if cid not in best or chunk[score_key] > best[cid][score_key]:
            best[cid] = chunk

    ranked = sorted(best.values(), key=lambda x: x[score_key], reverse=True)
    for i, doc in enumerate(ranked):
        doc["rank"] = i + 1
    return ranked


# Empirical, not guessed: nomic-embed-text-v1.5 doesn't produce near-zero
# cosine similarity for off-topic text on this corpus — a genuinely
# unrelated query ("show me a picture of a cat") still scored 0.59-0.64
# dense_score, while an on-topic query ("what is a PLC") scored 0.89-0.91.
# 0.75 sits in the gap between those two clusters. Note: this is
# dense_score (real 0-1 cosine similarity), NOT the "similarity" field on
# hybrid results, which is an RRF rank-fusion score (~0.003-0.03 range,
# not comparable to a cosine threshold at all).
MIN_DENSE_SCORE = 0.7


@traceable(name="retrieve_hyde_hybrid")
def retrieve(query: str, vdb: LanceVectorDB, llm_client: GeminiClient, top_k: int = 10,
             min_dense_score: float = MIN_DENSE_SCORE) -> list:
    """HyDE + hybrid retrieval — the best config from NB3 (MRR 0.489).

    Generates a hypothetical passage, searches with it via hybrid
    (dense + BM25 + RRF), dedupes chunk hits, then drops anything below
    min_dense_score — docs that only surfaced via keyword overlap or
    RRF's rank fusion, not genuine semantic relevance, so an off-topic
    query returns nothing instead of confidently-wrong "sources."
    """
    passage = generate_hyde_passage(query, llm_client)
    chunks = vdb.search_hybrid(passage, top_k=100)
    docs = aggregate_to_docs(chunks)
    docs = [d for d in docs if d.get("dense_score", 0.0) >= min_dense_score]
    return docs[:top_k]
