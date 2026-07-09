"""
Builds the LangGraph StateGraph.

Three domain personas: Technical Document Agent (retrieve -> generate, or
retrieve -> web_search -> generate when nothing clears the similarity
threshold — the Corrective RAG flow), PLC Expert (a single node that
explains PLC/Structured Text code and checks it against PLCopen coding
guidelines via iec-checker, grounded in the plc_simulation corpus), and
Diagnosis Agent (a single node answering questions about synthetic
energy/production telemetry, grounded in synthetic_data/energy_data.csv).
Plus Direct Reply (agents/direct_reply.py), a fast path for anything that
doesn't need a domain persona at all — see its own module docstring for
the real latency bug that motivated it.

The router (agents/router.py) is now an actual node, not a conditional
entry-point function — it writes its decision (routed_personas) into state,
and a conditional edge right after it (pick_next_node) reads that to decide
where to go: straight to a single persona's flow, or to multi_intent when
the router named more than one. multi_intent exists for compound questions
a single persona can't fully answer on its own (a real, confirmed case:
"how many PLC programs do I have, and what's my current power usage" —
neither plc_expert nor diagnosis_agent alone covers both halves). It calls
each named persona's existing node function directly as a plain Python
function (not more graph nodes/edges) and merges their answers — this
keeps the graph structure simple and leaves every individual persona's own
logic completely untouched.

Compiled with a SQLite checkpointer for short-term (thread-scoped) memory —
conversation history within one session. Not MemorySaver: that's in-process
only and gone on restart, same "not for production" caveat `langgraph dev`
itself warns about. Not Postgres: that's real multi-instance production
scale, which this single-user, single-machine system doesn't need. One
local file, survives restarts, zero new infrastructure — the same
reasoning that picked OpenPLC over CODESYS and Docker over building from
source throughout this project: match the tool to the actual scale.
"""

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from agents.state import AgentState
from agents.technical_document_agent import make_technical_document_agent_nodes
from agents.plc_expert import make_plc_expert_node
from agents.diagnosis_agent import make_diagnosis_agent_node
from agents.direct_reply import make_direct_reply_node
from agents.router import make_router_node, pick_next_node

_CHECKPOINT_DB = Path(__file__).resolve().parent.parent / "Database" / "agent_checkpoints.sqlite"

_PERSONA_LABELS = {
    "technical_document_agent": "General manufacturing/industry knowledge",
    "plc_expert": "PLC programs",
    "diagnosis_agent": "Energy/production data",
}


def _run_technical_document_agent(state: dict, retrieve_step, route_after_retrieve, web_search_step, generate_step) -> dict:
    """Drives the same retrieve -> (web_search) -> generate chain the graph
    itself wires up, but as plain Python calls — needed so multi_intent can
    get a full answer out of this multi-step persona without adding more
    graph nodes/edges just for the compound-question case."""
    working_state = {**state, **retrieve_step(state)}
    if route_after_retrieve(working_state) == "web_search":
        working_state = {**working_state, **web_search_step(working_state)}
    return generate_step(working_state)


def make_graph_nodes(vdb, llm_client, web_search_tool, db):
    retrieve_step, route_after_retrieve, web_search_step, generate_step = \
        make_technical_document_agent_nodes(vdb, llm_client, web_search_tool, db)
    plc_expert_step = make_plc_expert_node(llm_client)
    diagnosis_agent_step = make_diagnosis_agent_node(llm_client)
    direct_reply_step = make_direct_reply_node(llm_client)

    def run_persona(name: str, state: dict) -> dict:
        if name == "plc_expert":
            return plc_expert_step(state)
        if name == "diagnosis_agent":
            return diagnosis_agent_step(state)
        return _run_technical_document_agent(
            state, retrieve_step, route_after_retrieve, web_search_step, generate_step
        )

    def multi_intent_step(state: AgentState) -> dict:
        personas = state.get("routed_personas") or []

        # Personas run concurrently, not sequentially — each one is
        # I/O-bound (an LLM API call, sometimes a vector search or a
        # subprocess call to iec-checker), so it spends most of its time
        # waiting on a network or subprocess, not holding the GIL. Running
        # them one after another was a real, measured bug: a 3-persona
        # question took 50-173s in testing, roughly the SUM of each
        # persona's own latency. Threads (not asyncio) because none of the
        # persona node functions are actually async — this parallelizes
        # them without rewriting their internals.
        with ThreadPoolExecutor(max_workers=len(personas)) as executor:
            futures = {name: executor.submit(run_persona, name, state) for name in personas}
            results = {name: future.result() for name, future in futures.items()}

        parts = []
        all_sources = []
        for name in personas:
            result = results[name]
            label = _PERSONA_LABELS.get(name, name)
            parts.append(f"**{label}:**\n{result['answer']}")
            all_sources.extend(result.get("sources") or [])

        answer = "\n\n".join(parts)
        return {
            "answer": answer,
            "sources": all_sources,
            "history": [{"query": state["query"], "answer": answer, "sources": all_sources}],
        }

    return retrieve_step, route_after_retrieve, web_search_step, generate_step, \
        plc_expert_step, diagnosis_agent_step, direct_reply_step, multi_intent_step


def build_graph(vdb, llm_client, web_search_tool, db):
    retrieve_step, route_after_retrieve, web_search_step, generate_step, \
        plc_expert_step, diagnosis_agent_step, direct_reply_step, multi_intent_step = \
        make_graph_nodes(vdb, llm_client, web_search_tool, db)
    router_node = make_router_node(llm_client)

    graph = StateGraph(AgentState)
    graph.add_node("router", router_node)
    graph.add_node("retrieve", retrieve_step)
    graph.add_node("web_search", web_search_step)
    graph.add_node("generate", generate_step)
    graph.add_node("plc_expert", plc_expert_step)
    graph.add_node("diagnosis_agent", diagnosis_agent_step)
    graph.add_node("direct_reply", direct_reply_step)
    graph.add_node("multi_intent", multi_intent_step)

    graph.set_entry_point("router")
    graph.add_conditional_edges("router", pick_next_node, {
        "technical_document_agent": "retrieve",
        "plc_expert": "plc_expert",
        "diagnosis_agent": "diagnosis_agent",
        "direct_reply": "direct_reply",
        "multi_intent": "multi_intent",
    })
    graph.add_conditional_edges("retrieve", route_after_retrieve, {
        "generate": "generate",
        "web_search": "web_search",
    })
    graph.add_edge("web_search", "generate")
    graph.add_edge("generate", END)
    graph.add_edge("plc_expert", END)
    graph.add_edge("diagnosis_agent", END)
    graph.add_edge("direct_reply", END)
    graph.add_edge("multi_intent", END)

    # SqliteSaver.from_conn_string is a contextmanager whose cleanup runs
    # when the wrapper object gets garbage-collected — entering it manually
    # without holding a reference to that wrapper (only to the yielded
    # SqliteSaver) let Python GC it almost immediately, closing the
    # connection out from under the graph on the very next call. Passing a
    # plain sqlite3.Connection directly to SqliteSaver's constructor
    # sidesteps that lifetime problem entirely — the connection's lifetime
    # is just however long this Python process runs.
    _CHECKPOINT_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_CHECKPOINT_DB), check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    return graph.compile(checkpointer=checkpointer)
