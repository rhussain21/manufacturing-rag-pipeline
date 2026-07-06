"""
Standalone test harness for the PLC Expert node, isolated from the main
LangGraph system. Loads once, then lets you ask questions interactively.

Run:
    python test_plc_expert.py
"""

import contextlib
import io
import logging
import warnings


def main():
    logging.disable(logging.WARNING)
    warnings.filterwarnings("ignore")

    with contextlib.redirect_stdout(io.StringIO()):
        from llm_client import GeminiClient
        from agents.plc_expert import make_plc_expert_node

        llm_client = GeminiClient(model="gemini-2.5-flash")
        node = make_plc_expert_node(llm_client)

    print("PLC Expert (explain code, check best practices) — ask a question, or 'exit' to quit.")
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

        result = node({"query": query})
        print("=" * 60)
        print(result["answer"])
        print()
        if result["sources"]:
            print("Sources:")
            for s in result["sources"]:
                print(f"  - {s['title']}")
        print()


if __name__ == "__main__":
    main()
