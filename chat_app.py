"""
Standalone Streamlit chat UI for the LangGraph agent system — separate
from Dashboards/, on purpose (that's corpus oversight, this is just a
nicer way to talk to the agents than the CLI in main.py).

Run:
    streamlit run chat_app.py
"""

import contextlib
import io
import itertools
import os
import time
import uuid

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Industry Signals Chat",
    page_icon=":material/precision_manufacturing:",
    layout="wide",
)

# Plain text labels, no emoji — Material Symbols (":material/...:") are used
# instead wherever Streamlit needs an icon (button, avatar), since those are
# real vector icons, not emoji glyphs.
PERSONA_LABELS = {
    "technical_document_agent": "Technical Docs",
    "plc_expert": "PLC Expert",
    "analytics_agent": "Analytics",
    "direct_reply": "Direct Reply",
    "multi_intent": "Multi-Intent",
}
ASSISTANT_AVATAR = ":material/precision_manufacturing:"

# Palette + chat-bubble layout modeled on Claude's own chat UI (warm neutral
# canvas, coral accent, plain-text assistant turns vs. bubbled user turns)
# — the default Streamlit chrome (blue primary, square avatars, bold black
# "Chat" H1, visible Deploy/menu toolbar) read as an unstyled dev tool.
# stChatMessage/stChatMessageContent/etc. are Streamlit 1.59's real DOM
# testids, confirmed by inspecting the rendered app, not guessed — a user
# turn's content div carries aria-label="Chat message from user", which is
# the only reliable hook for giving it bubble/right-align styling without
# a role class on the row itself.
st.markdown(
    """
    <style>
    :root {
        --bg: #FAF9F6;
        --bg-sidebar: #F3F1EA;
        --border: #E8E4DA;
        --text: #2D2A26;
        --text-muted: #8A8478;
        --accent: #CC6A47;
        --accent-soft: #F3E4DB;
        --bubble-user: #F0EBE3;
        --shadow: 0 1px 2px rgba(45, 42, 38, 0.04), 0 1px 8px rgba(45, 42, 38, 0.03);
    }

    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Inter",
            "Segoe UI", Helvetica, Arial, sans-serif !important;
    }

    [data-testid="stAppViewContainer"], .stApp { background: var(--bg); color: var(--text); }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu, footer { display: none; }
    /* The chat input's fixed-bottom wrapper is a separate DOM layer with its
       own white background, layered above .stApp — without this the input
       row sits in a white strip that doesn't match the rest of the page. */
    [data-testid="stBottom"], [data-testid="stBottomBlockContainer"] {
        background: var(--bg) !important;
    }

    .block-container { padding-top: 3rem; max-width: 860px; }

    /* ── Sidebar ─────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: var(--bg-sidebar);
        border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] .block-container { padding-top: 2rem; }
    [data-testid="stSidebar"] hr { border-color: var(--border); margin: 1.25rem 0; }
    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4 {
        font-weight: 600; color: var(--text); letter-spacing: -0.01em;
    }
    [data-testid="stMetric"] {
        background: #fff; border: 1px solid var(--border); border-radius: 12px;
        padding: 10px 14px; box-shadow: var(--shadow);
    }
    [data-testid="stMetricLabel"] { color: var(--text-muted); }
    [data-testid="stMetricValue"] { color: var(--text); font-weight: 600; }
    [data-testid="stExpander"] {
        background: #fff; border: 1px solid var(--border) !important;
        border-radius: 12px !important; box-shadow: none;
    }

    /* ── Buttons / inputs ────────────────────────────────────────── */
    .stButton button {
        border-radius: 999px !important; border: 1px solid var(--border) !important;
        background: #fff !important; color: var(--text) !important;
        font-weight: 500; box-shadow: var(--shadow); transition: all 0.15s ease;
    }
    .stButton button:hover {
        border-color: var(--accent) !important; color: var(--accent) !important;
    }
    [data-testid="stChatInput"] {
        background: #fff; border: 1px solid var(--border); border-radius: 22px;
        box-shadow: var(--shadow);
    }
    [data-testid="stChatInput"]:focus-within { border-color: var(--accent); }
    [data-testid="stChatInput"] textarea { color: var(--text) !important; }
    [data-testid="stChatInput"] button {
        background: var(--accent) !important; border-radius: 50% !important;
    }

    /* ── Chat messages ───────────────────────────────────────────── */
    [data-testid="stChatMessage"] {
        gap: 10px; padding: 4px 0; align-items: flex-start;
    }
    /* User bubbles are one line tall — align that row to the bottom of the
       (empty, hidden) avatar slot instead of the top, or it looks stranded. */
    [data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) {
        align-items: flex-end;
    }
    [data-testid="stChatMessageAvatarAssistant"],
    [data-testid="stChatMessageAvatarCustom"] {
        background: var(--accent) !important; border-radius: 50% !important;
        overflow: hidden; display: flex; align-items: center; justify-content: center;
        width: 2rem; height: 2rem;
    }
    [data-testid="stChatMessageAvatarCustom"] [data-testid="stIconMaterial"] {
        color: #fff !important; font-size: 1.1rem;
    }
    [data-testid="stChatMessageAvatarUser"] { display: none; }
    [data-testid="stChatMessageContent"] { color: var(--text); line-height: 1.55; }
    [data-testid="stChatMessageContent"] p { margin-bottom: 0.4em; }
    [data-testid="stChatMessageContent"] pre {
        background: #F5F3EC; border: 1px solid var(--border); border-radius: 10px;
    }
    [data-testid="stChatMessageContent"] code { color: #A6552F; }

    /* User turn: bubble, right-aligned, no avatar */
    [data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) {
        flex-direction: row-reverse;
    }
    [data-testid="stChatMessage"]:has([aria-label="Chat message from user"])
        [data-testid="stChatMessageContent"] {
        background: var(--bubble-user); border-radius: 18px; padding: 10px 16px;
        max-width: 72%; margin-left: auto;
    }

    /* Assistant turn: plain text, no bubble, full width */
    [data-testid="stChatMessage"]:has([aria-label="Chat message from assistant"])
        [data-testid="stChatMessageContent"] {
        max-width: 100%;
    }

    .persona-badge {
        display: inline-block; padding: 3px 11px; border-radius: 999px;
        background: var(--accent-soft); color: #A6552F; font-size: 0.72rem;
        font-weight: 600; margin-right: 6px; letter-spacing: 0.01em;
    }
    .latency-tag { font-size: 0.72rem; color: var(--text-muted); }
    .source-chip {
        display: inline-block; padding: 2px 10px; border-radius: 999px;
        background: #fff; border: 1px solid var(--border); color: var(--text-muted);
        font-size: 0.72rem; margin: 3px 4px 0 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading agents (embedding model, DB, LLM client)...")
def load_graph():
    from device_config import config
    from db_relational import relationalDB
    from db_vector_lance import LanceVectorDB
    from llm_client import GeminiClient
    from tools.web_search import InternetSearchTool
    from agents.graph import build_graph

    with contextlib.redirect_stdout(io.StringIO()):
        # DB_PATH_ANALYTICS, not DB_PATH — DB_PATH points at the near-empty
        # test DB. Everything else that needs the real corpus (vectorize_lance,
        # sync_client, dashboard export) already uses DB_PATH_ANALYTICS; this
        # was the one place still wired to the wrong constant, which is why
        # corpus-inventory questions ("how many documents do you have")
        # answered 0 in the live chat app even though the vector index (built
        # from DB_PATH_ANALYTICS) had real content to retrieve from.
        db = relationalDB(config.DB_PATH_ANALYTICS)
        vdb = LanceVectorDB(
            config.LANCE_VECTOR_PATH,
            embedding_dim=768,
            model_name="nomic-ai/nomic-embed-text-v1.5",
            trust_remote_code=True,
        )
        llm_client = GeminiClient(model="gemini-2.5-flash")
        web_search_tool = InternetSearchTool(provider="tavily", api_key=os.getenv("TAVILY_API_KEY"))
        graph = build_graph(vdb, llm_client, web_search_tool, db)
    return graph, db


def _render_chart(chart_spec: dict | None) -> None:
    """Shared between the live turn and history replay so a chart looks the
    same whether it just streamed in or is being redrawn from session state.
    chart_spec["data"] is exactly what energy_data_search.compute_grouped_series
    returned — {"x": [...], "series": {metric: [...]}} — turned into a
    DataFrame here only because that's what st.*_chart wants, not because
    the underlying data changes shape."""
    if not chart_spec:
        return
    data = chart_spec["data"]
    if not data.get("x"):
        return
    df = pd.DataFrame(data["series"], index=data["x"])
    st.caption(chart_spec["title"])
    chart_fn = {"line": st.line_chart, "area": st.area_chart}.get(chart_spec["type"], st.bar_chart)
    chart_fn(df, color="#CC6A47")


graph, db = load_graph()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of dicts: role, content, personas, sources, latency

# ── Sidebar: corpus metrics + session controls ──────────────────────────
with st.sidebar:
    st.markdown("### Industry Signals")
    st.caption("Manufacturing corpus chat")

    if st.button("New conversation", icon=":material/refresh:", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown("#### Corpus")
    inv = db.get_corpus_inventory()
    c1, c2 = st.columns(2)
    c1.metric("Documents", inv["total"])
    c2.metric("Turns", len(st.session_state.messages) // 2)

    if inv["by_content_type"]:
        st.caption("By content type")
        st.bar_chart(inv["by_content_type"], horizontal=True, height=180, color="#CC6A47")

    if inv["by_source"]:
        with st.expander("Top sources"):
            for name, count in list(inv["by_source"].items())[:10]:
                st.markdown(f"**{count}** · {name}")

    st.divider()
    st.markdown("#### Session")
    assistant_turns = [m for m in st.session_state.messages if m["role"] == "assistant"]
    timed_turns = [m for m in assistant_turns if m.get("latency") is not None]
    if timed_turns:
        avg_latency = sum(m["latency"] for m in timed_turns) / len(timed_turns)
        st.metric("Avg response time", f"{avg_latency:.1f}s")
        persona_counts = {}
        for m in assistant_turns:
            for p in m.get("personas") or []:
                persona_counts[p] = persona_counts.get(p, 0) + 1
        for p, count in persona_counts.items():
            label = PERSONA_LABELS.get(p, p)
            st.caption(f"{label}: {count}")
    else:
        st.caption("No turns yet.")

# ── Main chat ────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    avatar = ASSISTANT_AVATAR if msg["role"] == "assistant" else None
    with st.chat_message(msg["role"], avatar=avatar):
        if msg["role"] == "assistant":
            badges = "".join(
                f'<span class="persona-badge">{PERSONA_LABELS.get(p, p)}</span>'
                for p in (msg.get("personas") or [])
            )
            latency = msg.get("latency")
            latency_html = f'<span class="latency-tag">{latency:.1f}s</span>' if latency else ""
            st.markdown(f"{badges} {latency_html}", unsafe_allow_html=True)
        st.markdown(msg["content"])
        if msg.get("sources"):
            chips = "".join(f'<span class="source-chip">{s}</span>' for s in msg["sources"])
            st.markdown(chips, unsafe_allow_html=True)
        _render_chart(msg.get("chart_spec"))

query = st.chat_input("Ask about PLCs, standards, vendor docs, or energy telemetry...")
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        # Badge placeholder created BEFORE st.write_stream so it keeps its
        # DOM position above the streamed text — it gets filled in after
        # the stream completes (personas/latency aren't known until then).
        badge_slot = st.empty()

        # A node exception (LLM call, vector search, etc.) used to propagate
        # all the way up and dump Streamlit's raw traceback in the middle of
        # the chat — a real, observed failure mode (LanceDB "too many open
        # files" crashed a turn mid-session with no graceful handling at
        # all). With streaming, a mid-answer LLM failure is converted into a
        # visible trailing note by GeminiClient.generate_stream (never
        # retried, never raised), so an exception here means a pre-content
        # failure (router/retrieval, or first-chunk exhausted retries) —
        # same case the old graph.invoke() handling covered.
        try:
            start = time.time()
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            stream_iter = graph.stream(
                {
                    "query": query, "resolved_query": "", "retrieved_docs": [], "web_results": [],
                    "answer": "", "sources": [], "chart_spec": None, "history": [], "routed_personas": [],
                },
                config=config,
                stream_mode="custom",
            )
            # Spinner covers the silent pre-first-token stretch (router +
            # retrieval — or the whole run for multi_intent, which emits no
            # custom-stream deltas at all). next() blocks until the first
            # custom event arrives or the graph finishes.
            with st.spinner("Thinking..."):
                first_chunk = next(stream_iter, None)

            def token_stream():
                if first_chunk is None:
                    return
                for chunk in itertools.chain([first_chunk], stream_iter):
                    if isinstance(chunk, dict) and chunk.get("delta"):
                        yield chunk["delta"]

            streamed_text = st.write_stream(token_stream())
            latency = time.time() - start
            # sources/routed_personas/latency aren't part of the token
            # stream — they're state written by the nodes, read back from
            # the checkpointer once the run is done.
            final_state = graph.get_state(config).values
        except Exception as e:
            print(f"[chat_app] graph.stream failed: {e!r}")
            st.error("Something went wrong answering that — try again.")
            st.session_state.messages.append({
                "role": "assistant", "content": "_Something went wrong answering that — try again._",
                "personas": [], "sources": [], "latency": None, "error": True,
            })
            st.stop()

        answer = final_state.get("answer") or ""
        if not (streamed_text or "").strip():
            # No deltas streamed — the multi_intent path (its persona calls
            # run in worker threads, where the stream writer no-ops). Render
            # the merged answer directly, identical to pre-streaming behavior.
            st.markdown(answer)

        personas = final_state.get("routed_personas") or []
        badges = "".join(
            f'<span class="persona-badge">{PERSONA_LABELS.get(p, p)}</span>'
            for p in personas
        )
        badge_slot.markdown(f'{badges} <span class="latency-tag">{latency:.1f}s</span>', unsafe_allow_html=True)

        source_labels = []
        for s in final_state.get("sources") or []:
            if s.get("content_id") is not None:
                source_labels.append(s["title"])
            elif "url" in s:
                source_labels.append(f"(web) {s['title']}")
            else:
                source_labels.append(f"(corpus) {s['title']}")
        if source_labels:
            chips = "".join(f'<span class="source-chip">{s}</span>' for s in source_labels)
            st.markdown(chips, unsafe_allow_html=True)

        chart_spec = final_state.get("chart_spec")
        _render_chart(chart_spec)

    st.session_state.messages.append({
        "role": "assistant", "content": answer,
        "personas": personas, "sources": source_labels, "latency": latency,
        "chart_spec": chart_spec,
    })
