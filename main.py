"""
AI Industry Signals - Agent System

LangGraph-based, three personas routed automatically based on the question:
Technical Document Agent (manufacturing corpus, HyDE + hybrid retrieval),
PLC Expert (Structured Text code explanation + best-practices checks
against the plc_simulation corpus), and Diagnosis Agent (synthetic
energy/production telemetry, synthetic_data/energy_data.csv). Conversation
history persists within a session via a SQLite checkpointer, so follow-up
questions can refer back to prior turns.

Run:
    python main.py                          # interactive chat (loads once, ask multiple questions)
    python main.py "What is IEC 62443?"      # single question, answer, exit
    python main.py --debug                  # either mode, but show setup/init logging
"""

import argparse
import contextlib
import io
import logging
import uuid
import warnings


def _setup():
    import os

    from device_config import config
    from db_relational import relationalDB
    from db_vector_lance import LanceVectorDB
    from llm_client import GeminiClient
    from tools.web_search import InternetSearchTool
    from agents.graph import build_graph

    db = relationalDB(config.DB_PATH)
    vdb = LanceVectorDB(
        config.LANCE_VECTOR_PATH,
        embedding_dim=768,
        model_name="nomic-ai/nomic-embed-text-v1.5",
        trust_remote_code=True,
    )
    llm_client = GeminiClient(model="gemini-2.5-flash")
    web_search_tool = InternetSearchTool(provider="tavily", api_key=os.getenv("TAVILY_API_KEY"))
    return build_graph(vdb, llm_client, web_search_tool, db)


def _ask(graph, query: str, thread_id: str) -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    return graph.invoke(
        {"query": query, "retrieved_docs": [], "web_results": [], "answer": "", "sources": [], "history": [], "routed_personas": []},
        config=config,
    )


def _print_answer(query: str, result: dict):
    print("=" * 60)
    print(f"Query: {query}")
    print("=" * 60)
    print(result["answer"])
    print()
    if result["sources"]:
        print("Sources:")
        for s in result["sources"]:
            if s.get("content_id") is not None:
                print(f"  - [{s['content_id']}] {s['title']}")
            elif "url" in s:
                print(f"  - (web) {s['title']} — {s.get('url', '')}")
            elif s["title"].endswith(".csv"):
                print(f"  - (energy data) {s['title']}")
            else:
                print(f"  - (plc corpus) {s['title']}")
    print()


def _quiet_ask(graph, query: str, thread_id: str) -> dict:
    with contextlib.redirect_stdout(io.StringIO()):
        return _ask(graph, query, thread_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="*")
    parser.add_argument("--debug", action="store_true", help="show setup/init logging (device config, model loading, etc.)")
    args = parser.parse_args()
    one_shot_query = " ".join(args.query)
    ask = _ask if args.debug else _quiet_ask
    thread_id = str(uuid.uuid4())

    if args.debug:
        graph = _setup()
    else:
        logging.disable(logging.WARNING)
        warnings.filterwarnings("ignore")
        with contextlib.redirect_stdout(io.StringIO()):
            graph = _setup()

    if one_shot_query:
        result = ask(graph, one_shot_query, thread_id)
        _print_answer(one_shot_query, result)
        return

    print("Technical Document Agent / PLC Expert / Diagnosis Agent — ask a question, or 'exit' to quit.")
    print()
    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            continue
        if query.lower() in ("exit", "quit"):
            break
        try:
            result = ask(graph, query, thread_id)
        except Exception as e:
            # A transient failure (API rate limit, a dropped connection) on
            # one turn shouldn't kill the whole session — print it plainly
            # and let the user try again, rather than crashing out with a
            # raw traceback and losing the conversation.
            print(f"That turn failed: {e}")
            print("Try again — this doesn't end the conversation.")
            print()
            continue
        _print_answer(query, result)


if __name__ == "__main__":
    main()
