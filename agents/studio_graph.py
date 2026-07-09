"""
Entry point for LangGraph Studio (`langgraph dev`).

Studio needs a bare compiled graph object to import, not a parameterized
function — this module does the vdb/llm_client instantiation once at
import time and exposes the result as `graph`. Only imported by the
langgraph dev server (see langgraph.json); main.py builds its own
instance and never imports this file, so nothing gets loaded twice.
"""

import os

from device_config import config
from db_relational import relationalDB
from db_vector_lance import LanceVectorDB
from llm_client import GeminiClient
from tools.web_search import InternetSearchTool
from agents.graph import build_graph

_db = relationalDB(config.DB_PATH)
_vdb = LanceVectorDB(
    config.LANCE_VECTOR_PATH,
    embedding_dim=768,
    model_name="nomic-ai/nomic-embed-text-v1.5",
    trust_remote_code=True,
)
_llm_client = GeminiClient(model="gemini-2.5-flash")
_web_search_tool = InternetSearchTool(provider="tavily", api_key=os.getenv("TAVILY_API_KEY"))

graph = build_graph(_vdb, _llm_client, _web_search_tool, _db)
