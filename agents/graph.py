"""
Builds the LangGraph StateGraph.

Two personas now: Technical Document Agent (retrieve -> generate, or
retrieve -> web_search -> generate when nothing clears the similarity
threshold — the Corrective RAG flow) and PLC Expert (a single node that
explains PLC/Structured Text code and checks it against PLCopen coding
guidelines via iec-checker, grounded in the plc_simulation corpus). A
router (agents/router.py) is the graph's conditional entry point, picking
which persona's flow to enter based on the query. This file is mostly
LangGraph wiring (add_node/add_conditional_edges/compile), not custom
logic — the actual decision-making lives in each persona's own module.
"""

from langgraph.graph import StateGraph, END
from agents.state import AgentState
from agents.technical_document_agent import make_technical_document_agent_nodes
from agents.plc_expert import make_plc_expert_node
from agents.router import make_router


def build_graph(vdb, llm_client, web_search_tool):
    retrieve_step, route_after_retrieve, web_search_step, generate_step = \
        make_technical_document_agent_nodes(vdb, llm_client, web_search_tool)
    plc_expert_step = make_plc_expert_node(llm_client)
    route_query = make_router(llm_client)

    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve_step)
    graph.add_node("web_search", web_search_step)
    graph.add_node("generate", generate_step)
    graph.add_node("plc_expert", plc_expert_step)

    graph.set_conditional_entry_point(route_query, {
        "technical_document_agent": "retrieve",
        "plc_expert": "plc_expert",
    })
    graph.add_conditional_edges("retrieve", route_after_retrieve, {
        "generate": "generate",
        "web_search": "web_search",
    })
    graph.add_edge("web_search", "generate")
    graph.add_edge("generate", END)
    graph.add_edge("plc_expert", END)

    return graph.compile()
