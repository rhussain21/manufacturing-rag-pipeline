"""
Builds the LangGraph StateGraph.

One persona right now (Technical Doc Expert), expressed as three nodes with
a real conditional edge (Corrective RAG): retrieve -> generate directly if
docs survived the similarity threshold, otherwise -> web_search -> generate.
No router between personas yet — nothing to route between until the
Diagnosis Agent exists. This file is mostly LangGraph wiring
(add_node/add_conditional_edges/compile), not custom logic; the actual
decision-making lives in technical_doc_expert.py.
"""

from langgraph.graph import StateGraph, END
from agents.state import AgentState
from agents.technical_doc_expert import make_technical_doc_expert_nodes


def build_graph(vdb, llm_client, web_search_tool):
    retrieve_step, route_after_retrieve, web_search_step, generate_step = \
        make_technical_doc_expert_nodes(vdb, llm_client, web_search_tool)

    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve_step)
    graph.add_node("web_search", web_search_step)
    graph.add_node("generate", generate_step)

    graph.set_entry_point("retrieve")
    graph.add_conditional_edges("retrieve", route_after_retrieve, {
        "generate": "generate",
        "web_search": "web_search",
    })
    graph.add_edge("web_search", "generate")
    graph.add_edge("generate", END)

    return graph.compile()
