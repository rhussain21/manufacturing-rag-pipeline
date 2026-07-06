"""
Builds the LangGraph StateGraph.

Three personas now: Technical Document Agent (retrieve -> generate, or
retrieve -> web_search -> generate when nothing clears the similarity
threshold — the Corrective RAG flow), PLC Expert (a single node that
explains PLC/Structured Text code and checks it against PLCopen coding
guidelines via iec-checker, grounded in the plc_simulation corpus), and
Diagnosis Agent (a single node answering questions about synthetic
energy/production telemetry, grounded in synthetic_data/energy_data.csv). A
router (agents/router.py) is the graph's conditional entry point, picking
which persona's flow to enter based on the query. This file is mostly
LangGraph wiring (add_node/add_conditional_edges/compile), not custom
logic — the actual decision-making lives in each persona's own module.

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
from pathlib import Path

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from agents.state import AgentState
from agents.technical_document_agent import make_technical_document_agent_nodes
from agents.plc_expert import make_plc_expert_node
from agents.diagnosis_agent import make_diagnosis_agent_node
from agents.router import make_router

_CHECKPOINT_DB = Path(__file__).resolve().parent.parent / "Database" / "agent_checkpoints.sqlite"


def build_graph(vdb, llm_client, web_search_tool):
    retrieve_step, route_after_retrieve, web_search_step, generate_step = \
        make_technical_document_agent_nodes(vdb, llm_client, web_search_tool)
    plc_expert_step = make_plc_expert_node(llm_client)
    diagnosis_agent_step = make_diagnosis_agent_node(llm_client)
    route_query = make_router(llm_client)

    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve_step)
    graph.add_node("web_search", web_search_step)
    graph.add_node("generate", generate_step)
    graph.add_node("plc_expert", plc_expert_step)
    graph.add_node("diagnosis_agent", diagnosis_agent_step)

    graph.set_conditional_entry_point(route_query, {
        "technical_document_agent": "retrieve",
        "plc_expert": "plc_expert",
        "diagnosis_agent": "diagnosis_agent",
    })
    graph.add_conditional_edges("retrieve", route_after_retrieve, {
        "generate": "generate",
        "web_search": "web_search",
    })
    graph.add_edge("web_search", "generate")
    graph.add_edge("generate", END)
    graph.add_edge("plc_expert", END)
    graph.add_edge("diagnosis_agent", END)

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
