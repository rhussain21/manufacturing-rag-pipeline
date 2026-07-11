#!/usr/bin/env python3
"""
AI Industry Signals -- Intelligence Dashboard

Edge AI pipeline dashboard for industrial automation knowledge extraction.
Reads pre-exported JSON snapshots -- no live DB connection needed.

Run:
    streamlit run Dashboards/dashboard.py
    DATA_URL=https://your-site.netlify.app streamlit run Dashboards/dashboard.py
"""

import json
import os
import sys
import time
import requests
from collections import Counter

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from streamlit_javascript import st_javascript

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Config ────────────────────────────────────────────────────────────────

LOCAL_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dashboard_data')
REMOTE_DATA_URL = os.getenv('DATA_URL', '')

st.set_page_config(
    page_title="Private RAG Evaluation Dashboard",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Mobile detection ─────────────────────────────────────────────────────
# Reads the real browser viewport width so Plotly figures (which are built
# server-side with fixed pixel fonts/margins) can be laid out differently on
# phones instead of just being squeezed into a narrower container. Matches the
# 640px breakpoint used by the CSS media queries below. Falls back to desktop
# on the first render, before the browser has reported a width, and on any
# read failure (e.g. no JS runtime available) so nothing regresses.
_viewport_width = st_javascript("window.innerWidth")
IS_MOBILE = isinstance(_viewport_width, (int, float)) and 0 < _viewport_width < 641

# ── Theme ─────────────────────────────────────────────────────────────────

GRN       = '#225347'
GRN_LT    = '#2f6b5b'
GRN_PALE  = '#d4edda'
GRN_DK    = '#193A32'
OFF_WHITE = '#FBF1EC'
CREAM     = '#FAF7F4'
BEIGE     = '#F5EFE6'   # outer frame / mat
INNER_BG  = '#EDEAE4'  # inner content container
CARD_BG   = '#F8F5F0'  # metric cards — warm off-white, not jarring on INNER_BG
BG        = GRN_DK
TXT       = OFF_WHITE
TXT_DARK  = '#1a1a1a'
TXT2      = '#555555'
BDR       = '#D8D2CA'
WARN      = '#e67e22'
ERR       = '#e74c3c'
TBL_CELL  = '#F5F2EE'  # table row cells
TBL_HDR   = '#DDD8D0'  # table header
GRID      = 'rgba(0,0,0,0.18)'  # plotly gridlines — subtle dark

# Color palettes
TABLEAU = ['#4e79a7', '#59a14f', '#e15759', '#76b7b2', '#5a9e6f',
           '#f28e2b', '#b07aa1', '#ff9da7', '#9c755f', '#bab0ac']
SEQ     = ['#4e79a7', '#6baed6', '#9ecae1', '#c6dbef', '#a1d99b', '#74c476']
MULTI   = ['#4e79a7', '#59a14f', '#e15759', '#76b7b2', '#f28e2b',
           '#b07aa1', '#9c755f', '#5a9e6f', '#ff9da7', '#bab0ac']

# ── CSS ───────────────────────────────────────────────────────────────────

st.markdown(f"""<style>
    /* ── glide-data-grid canvas vars at :root ── */
    :root {{
        --gdg-bg-cell: {TBL_CELL};
        --gdg-bg-cell-medium: {TBL_CELL};
        --gdg-bg-header: {TBL_HDR};
        --gdg-bg-header-has-focus: #ccc6be;
        --gdg-bg-header-hovered: #e4ded8;
        --gdg-text-dark: {TXT_DARK};
        --gdg-text-medium: {TXT_DARK};
        --gdg-text-light: {TXT2};
        --gdg-text-header: {TXT_DARK};
        --gdg-text-group-header: {TXT_DARK};
        --gdg-border-color: {BDR};
        --gdg-bg-bubble: #e8e4de;
        --gdg-bg-bubble-selected: {TBL_HDR};
        --gdg-accent-color: {GRN};
        --gdg-accent-light: #d4edda;
    }}

    /* ── Global: black text everywhere ── */
    *, *::before, *::after {{ color: {TXT_DARK} !important; }}

    /* ── Outermost: green app background ── */
    .stApp {{ background-color: {GRN_DK} !important; }}
    header[data-testid="stHeader"] {{ background: {GRN_DK} !important; }}

    /* ── Layer 1: beige "mat" — thin padding, no border (just shadow) ── */
    .stMainBlockContainer, .block-container {{
        background: {BEIGE} !important;
        border-radius: 6px;
        padding: 14px 16px !important;
        margin: 22px auto;
        max-width: calc(100% - 48px) !important;
        box-shadow: 0 4px 28px rgba(0,0,0,0.28);
        border: none;
    }}
    section[data-testid="stMain"] > div {{
        background: transparent !important;
        padding: 0 !important;
    }}

    /* ── Layer 2: inner container — header + tabs form one unified box ── */
    .dash-inner {{
        background: {INNER_BG};
        border-radius: 4px 4px 0 0;
        overflow: hidden;
    }}
    /* Streamlit's vertical flex gap between top-level blocks is what shows as a
       dark seam between the hero and the tab bar (empty flex gap exposing the
       page behind it, not a color mismatch) — remove the gap at the source,
       scoped to the top-level block only so column/widget spacing elsewhere
       is untouched. */
    section[data-testid="stMain"] > div > div[data-testid="stVerticalBlock"] {{
        gap: 0 !important;
    }}
    div[data-testid="stMarkdownContainer"]:has(.dash-inner),
    div[data-testid="element-container"]:has(.dash-inner),
    div[data-testid="stElementContainer"]:has(.dash-inner) {{
        margin-bottom: 0 !important;
        padding-bottom: 0 !important;
        line-height: 0;
    }}

    .stTabs {{
        background: {INNER_BG} !important;
        border-radius: 0 0 4px 4px;
        padding: 0;
        border: none;
        margin-top: 0 !important;
    }}
    .stTabs [data-baseweb="tab-list"] {{
        background: {INNER_BG} !important;
        border-radius: 0;
        padding: 0;
        border-bottom: 2px solid {BDR};
        justify-content: center !important;
        gap: 0;
    }}
    .stTabs [data-baseweb="tab"] {{
        color: {TXT_DARK} !important;
        font-weight: 600;
        border-radius: 0;
        padding: 10px 18px;
        font-size: 1.02rem;
        border-bottom: 3px solid transparent;
        background: transparent !important;
    }}
    .stTabs [aria-selected="true"] {{
        background: transparent !important;
        color: {GRN} !important;
        border-bottom: 3px solid {GRN} !important;
        font-weight: 700 !important;
    }}
    .stTabs [data-baseweb="tab"]:hover {{ color: {GRN_LT} !important; }}
    .stTabs [data-baseweb="tab-panel"] {{
        background: {INNER_BG} !important;
        border-radius: 0;
        padding: 28px 52px 36px 52px !important;
    }}

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {{ background: {GRN_DK} !important; }}
    section[data-testid="stSidebar"] * {{ color: {OFF_WHITE} !important; }}

    /* ── Metric cards ── */
    div[data-testid="stMetric"] {{
        background: {CARD_BG};
        border: 1px solid {BDR};
        border-radius: 8px;
        padding: 12px 16px;
    }}
    div[data-testid="stMetric"] label {{ color: {TXT2} !important; font-size: 0.82rem; }}
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
        color: {TXT_DARK} !important; font-size: 1.6rem; font-weight: 700;
    }}

    /* ── DataFrames — warm-toned, no jarring white ── */
    .stDataFrame, div[data-testid="stDataFrame"] {{
        border: 1px solid {BDR} !important;
        border-radius: 8px;
        background: {TBL_CELL} !important;
    }}
    .stDataFrame table {{ background: {TBL_CELL} !important; }}
    .stDataFrame th {{
        background: {TBL_HDR} !important;
        color: {TXT_DARK} !important;
        font-weight: 700 !important;
        border-bottom: 2px solid {BDR} !important;
    }}
    .stDataFrame td {{
        background: {TBL_CELL} !important;
        color: {TXT_DARK} !important;
        border-bottom: 1px solid {BDR} !important;
    }}
    div[data-testid="stDataFrame"] > div {{ background-color: {TBL_CELL} !important; }}

    /* ── Section headers ── */
    .section-label {{
        font-size: 0.92rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        color: {TXT_DARK} !important;
        border-bottom: 2px solid {GRN_LT};
        padding-bottom: 2px;
        margin: 18px 0 2px 0;
    }}
    .section-so {{
        font-size: 0.88rem;
        color: {TXT2} !important;
        font-style: italic;
        margin: 0 0 10px 0;
        line-height: 1.4;
    }}

    /* ── Custom metric card (HTML) ── */
    .metric-card {{
        background: {CARD_BG};
        border: 1px solid {BDR};
        border-radius: 8px;
        padding: 14px 18px;
        height: 116px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        gap: 4px;
    }}
    .metric-card .metric-value {{
        font-size: 1.95rem; font-weight: 700; color: {TXT_DARK} !important;
        line-height: 1.15;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }}
    .metric-card .metric-label {{
        font-size: 0.82rem; font-weight: 600; color: {TXT2} !important;
        text-transform: uppercase; letter-spacing: 0.05em;
    }}

    /* ── Plotly containers ── */
    .stPlotlyChart {{ border-radius: 8px; overflow: hidden; }}

    /* ── Expander ── */
    .streamlit-expanderHeader {{ color: {TXT_DARK} !important; font-weight: 600; }}

    /* ── Captions and markdown ── */
    .stCaption, .stMarkdown p {{ color: {TXT_DARK} !important; }}

    /* ── Form controls — visible on INNER_BG ── */
    .stSelectbox label, .stSlider label, .stMultiSelect label {{
        color: {TXT_DARK} !important; font-weight: 600;
    }}
    .stSelectbox > div > div,
    .stSelectbox div[data-baseweb="select"] > div,
    .stMultiSelect div[data-baseweb="select"] > div {{
        background: {CARD_BG} !important;
        border: 1.5px solid #b0a898 !important;
        border-radius: 6px !important;
        color: {TXT_DARK} !important;
    }}
    .stSelectbox div[data-baseweb="select"] *,
    .stMultiSelect div[data-baseweb="select"] * {{ color: {TXT_DARK} !important; }}

    /* ── Checkbox-style multiselect dropdown items ── */
    [data-baseweb="menu"] [role="option"] {{
        padding-left: 36px !important;
        position: relative;
        background: {CARD_BG} !important;
        color: {TXT_DARK} !important;
    }}
    [data-baseweb="menu"] [role="option"]::before {{
        content: '';
        position: absolute;
        left: 10px;
        top: 50%;
        transform: translateY(-50%);
        width: 14px;
        height: 14px;
        border: 2px solid #8a8078;
        border-radius: 3px;
        background: #ffffff;
        box-sizing: border-box;
    }}
    [data-baseweb="menu"] [role="option"][aria-selected="true"]::before {{
        background: {GRN} !important;
        border-color: {GRN} !important;
    }}
    [data-baseweb="menu"] [role="option"][aria-selected="true"]::after {{
        content: '';
        position: absolute;
        left: 14px;
        top: 50%;
        transform: translateY(-65%) rotate(45deg);
        width: 4px;
        height: 8px;
        border-right: 2px solid #ffffff;
        border-bottom: 2px solid #ffffff;
    }}
    [data-baseweb="menu"] [role="option"]:hover {{
        background: {TBL_CELL} !important;
    }}

    .stSlider > div {{ color: {TXT_DARK} !important; }}
    .stSlider [data-baseweb="slider"] {{ background: #d4cec6 !important; }}
    .stNumberInput > div > div input,
    .stTextInput > div > div input {{
        background: {CARD_BG} !important;
        border: 1.5px solid #b0a898 !important;
        color: {TXT_DARK} !important;
        border-radius: 6px !important;
    }}

    /* ── Links ── */
    a {{ color: {GRN} !important; }}
    a:hover {{ color: {GRN_LT} !important; }}

    /* ── Tab explainer card — "what / why / watch for", 3 cols desktop, stacked mobile ── */
    .tab-explainer {{
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 18px;
        background: {CARD_BG};
        border: 1px solid {BDR};
        border-radius: 10px;
        padding: 14px 22px;
        margin: 2px 0 22px 0;
    }}
    .tab-explainer-label {{
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        color: {GRN} !important;
        margin-bottom: 3px;
    }}
    .tab-explainer-text {{
        font-size: 0.92rem;
        line-height: 1.5;
        color: {TXT_DARK} !important;
    }}
    @media (max-width: 900px) {{
        .tab-explainer {{ grid-template-columns: 1fr; }}
    }}

    /* ── Mobile (phone-width viewports) — additive only, desktop rules above
       are untouched so this can't regress the browser/desktop layout ── */
    @media (max-width: 640px) {{
        .stMainBlockContainer, .block-container {{
            margin: 8px auto !important;
            padding: 8px 8px !important;
            max-width: calc(100% - 16px) !important;
        }}
        .dash-hero-wrap {{ padding: 14px 16px 12px 16px !important; }}
        .dash-hero-title {{ font-size: 1.2rem !important; }}
        .dash-hero-text {{ white-space: normal !important; font-size: 0.74rem !important; }}
        .dash-hero-note {{ font-size: 0.66rem !important; }}

        .stTabs [data-baseweb="tab-list"] {{ gap: 0 !important; }}
        .stTabs [data-baseweb="tab"] {{
            padding: 8px 8px !important;
            font-size: 0.76rem !important;
        }}
        .stTabs [data-baseweb="tab-panel"] {{ padding: 16px 12px 24px 12px !important; }}

        .tab-explainer {{ padding: 12px 14px !important; gap: 10px !important; }}
        .tab-explainer-text {{ font-size: 0.78rem !important; }}

        .metric-card {{
            height: auto !important;
            min-height: 84px;
            padding: 10px 14px !important;
        }}
        .metric-card .metric-value {{ font-size: 1.35rem !important; }}

        div[data-testid="stMetric"] {{ padding: 10px 12px !important; }}
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {{ font-size: 1.3rem !important; }}
    }}

    /* ── Behavior Demo tab ── */
    .rd-badge {{
        display: inline-block; padding: 3px 12px; border-radius: 999px;
        background: {GRN_PALE}; color: {GRN}; font-size: 0.75rem; font-weight: 600;
        letter-spacing: 0.01em;
    }}
    .rd-cached-note {{
        font-size: 0.75rem; color: {TXT2}; margin: 2px 0 6px 0;
        display: flex; align-items: center; gap: 6px; flex-wrap: wrap;
    }}
    .rd-turn {{ margin-bottom: 14px; display: flex; }}
    .rd-turn.user {{ justify-content: flex-end; }}
    .rd-bubble-user {{
        background: {TBL_CELL}; border-radius: 16px; padding: 9px 15px;
        max-width: 78%; color: {TXT_DARK}; font-size: 0.92rem;
    }}
    .rd-assistant-meta {{ display: flex; align-items: center; gap: 8px; margin-bottom: 4px; }}
    .rd-persona-badge {{
        display: inline-block; padding: 2px 10px; border-radius: 999px;
        background: {GRN_PALE}; color: {GRN}; font-size: 0.7rem; font-weight: 600;
    }}
    .rd-latency {{ font-size: 0.72rem; color: {TXT2}; }}
    .rd-source-chip {{
        display: inline-block; padding: 2px 10px; border-radius: 999px;
        background: #fff; border: 1px solid {BDR}; color: {TXT2};
        font-size: 0.7rem; margin: 4px 4px 0 0;
    }}
    .rd-toggle-label {{ font-size: 0.85rem; font-weight: 600; color: {TXT_DARK}; margin-bottom: 6px; }}
</style>""", unsafe_allow_html=True)

# ── Color scales ──────────────────────────────────────────────────────────

VIBRANT_SCALE = [
    [0.0, '#2166ac'], [0.25, '#67a9cf'], [0.5, '#f7f7f7'],
    [0.75, '#ef8a62'], [1.0, '#b2182b'],
]
WARM_COOL = [
    [0.0, '#4e79a7'], [0.25, '#76b7b2'], [0.5, '#f7f7f7'],
    [0.75, '#f28e2b'], [1.0, '#e15759'],
]

# ── Rejection reason categories ───────────────────────────────────────────

# Keys are matched as substrings (lowercase). Order matters for specificity —
# longer/more-specific keys should come before shorter ones that overlap.
_REJECTION_MAP = [
    # Quality gate codes (structured)
    ('low_diversity',     'Low Content Diversity'),   # [quality_gate] low_diversity (X%)
    ('empty_transcript',  'Empty / Blank Content'),   # [quality_gate] empty_transcript
    ('too_short',         'Too Short'),               # [quality_gate] too_short (X chars < 500)
    # LLM-generated rejection text patterns
    ('excerpt is empty',  'Empty / Blank Content'),   # "The content excerpt is empty..."
    ('0 characters',      'Empty / Blank Content'),   # "...empty (0 characters)..."
    ('generic',           'Generic / No Signals'),    # "generic, introductory..."
    ('no substantive',    'Empty / Blank Content'),   # "no substantive information..."
    # General fallbacks
    ('duplicate',         'Duplicate Content'),
    ('already_exists',    'Duplicate Content'),
    ('not_relevant',      'Out of Scope'),
    ('off_topic',         'Out of Scope'),
    ('not_manufacturing', 'Out of Scope'),
    ('not_industrial',    'Out of Scope'),
    ('paywall',           'Access Issues'),
    ('language',          'Language / Format'),
    ('non_english',       'Language / Format'),
    ('extraction',        'Processing Error'),
    ('timeout',           'Processing Error'),
]

def _safe_div_pct(a, b) -> float:
    return (a / b * 100) if b else 0.0


def _categorize_rejection(reason: str) -> str:
    r = str(reason).lower().strip()
    for key, cat in _REJECTION_MAP:
        if key in r:
            return cat
    return 'Uncategorized'


def _aggregate_rejection_reasons(reasons: list) -> pd.DataFrame:
    df = pd.DataFrame(reasons)
    if df.empty or 'screening_reason' not in df.columns:
        return df
    df['category'] = df['screening_reason'].apply(_categorize_rejection)
    agg = df.groupby('category')['count'].sum().reset_index()
    return agg.sort_values('count', ascending=False)


# ── Data loading ──────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_json(filename: str) -> dict:
    if REMOTE_DATA_URL:
        try:
            r = requests.get(f"{REMOTE_DATA_URL.rstrip('/')}/{filename}", timeout=10)
            r.raise_for_status()
            return r.json()
        except Exception:
            pass
    path = os.path.join(LOCAL_DATA_DIR, filename)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def load_all() -> dict:
    return {
        'pipeline':  load_json('pipeline_stats.json'),
        'temporal':  load_json('temporal_data.json'),
        'logs':      load_json('system_logs.json'),
        'discovery': load_json('discovery_stats.json'),
        'corpus':    load_json('corpus_quality.json'),
        'topic_map': load_json('topic_map.json'),
    }


# ── Helpers ───────────────────────────────────────────────────────────────

def sl(text):
    """Section label — bold uppercase heading."""
    st.markdown(f'<div class="section-label">{text}</div>', unsafe_allow_html=True)


def so(text):
    """Section subtitle — the 'so what' in one line."""
    st.markdown(f'<div class="section-so">{text}</div>', unsafe_allow_html=True)


def mc(label, value, color="g", small=False):
    bg = GRN_PALE if color == "g" else ('#fce4ec' if color == "r" else '#fff3e0')
    value_style = "font-size:1.2rem;" if small else ""
    st.markdown(f"""<div class="metric-card" style="background:{bg};">
        <div class="metric-value" style="{value_style}">{value}</div>
        <div class="metric-label">{label}</div>
    </div>""", unsafe_allow_html=True)


# Reusable per-tab explainer config — one place to edit copy, one renderer for all tabs.
TAB_EXPLAINERS = {
    "Pipeline Overview": {
        "what": "The full document pipeline from intake to searchable knowledge base.",
        "why": "It reveals whether documents are flowing through the system cleanly or getting stuck before they become useful.",
        "watch_for": "Ingestion failures, unapproved documents, missing metadata, or large drops between stages.",
    },
    "Dataset Quality": {
        "what": "What is actually inside the searchable corpus — topics, formats, sources, duplication, and coverage.",
        "why": "Retrieval quality depends on corpus quality. A RAG system cannot answer well if the underlying documents are thin, redundant, or poorly distributed.",
        "watch_for": "Coverage gaps, duplicate-heavy topics, overrepresented sources, or important document types missing from the vector index.",
    },
    "Retrieval Quality": {
        "what": "How well the system retrieves the right supporting evidence for labeled test questions.",
        "why": "This is the core quality measure for RAG. It checks whether the system finds useful evidence before an LLM ever generates an answer.",
        "watch_for": "Low recall, weak MRR, unresolved queries, or test questions where the right document exists but is not being retrieved.",
    },
    "Answer Quality": {
        "what": "Whether the final answers from the real, live agent pipeline (not a notebook shortcut) actually contain correct, grounded information — tracked across every real pipeline change, not just retrieval in isolation.",
        "why": "Good retrieval doesn't guarantee a good answer. This is the accountability layer: fact recall, groundedness, and hallucination rate on the exact system a user would actually talk to.",
        "watch_for": "Fact recall below target, rising hallucination rate after a change, or a config comparison where a routing/generation improvement doesn't actually move the numbers.",
    },
    "Behavior Demo": {
        "what": "Real, cached conversations captured from the live system — not live calls.",
        "why": "Numbers show something improved. This shows what improved.",
        "watch_for": "Who answered, what they cited, and how the answer changed.",
    },
    "System Logs": {
        "what": "Operational events from ingestion, processing, embedding, retrieval, and evaluation runs.",
        "why": "Logs make the pipeline auditable and help explain why metrics changed.",
        "watch_for": "Failed jobs, skipped files, parsing errors, embedding failures, or sudden changes after a pipeline update.",
    },
    "Project Information": {
        "what": "The architecture, design choices, and implementation details behind the dashboard.",
        "why": "It gives clients or employers enough context to understand what was built, why certain tradeoffs were made, and how the system could generalize to other private document environments.",
        "watch_for": "Deployment model, data flow, privacy assumptions, evaluation design, and next-step roadmap.",
    },
}


def tab_explainer(tab_name: str):
    """Compact 'what am I looking at' card for the top of a tab — same neutral card
    style every time, regardless of tab, so it never reads as a warning/error state."""
    info = TAB_EXPLAINERS.get(tab_name)
    if not info:
        return
    st.markdown(f"""<div class="tab-explainer">
        <div>
            <div class="tab-explainer-label">What this shows</div>
            <div class="tab-explainer-text">{info['what']}</div>
        </div>
        <div>
            <div class="tab-explainer-label">Why it matters</div>
            <div class="tab-explainer-text">{info['why']}</div>
        </div>
        <div>
            <div class="tab-explainer-label">Watch for</div>
            <div class="tab-explainer-text">{info['watch_for']}</div>
        </div>
    </div>""", unsafe_allow_html=True)


# Standard sizes so every chart's axes and legend read at the same scale.
# Smaller on mobile so labels have a fighting chance in a much narrower chart.
AXIS_FONT_SIZE = 11 if IS_MOBILE else 14
LEGEND_FONT_SIZE = 11 if IS_MOBILE else 14

# Shared Plotly config for interactive charts: keeps hover tooltips and the
# autoscale ("fit to view") button, drops drag-to-zoom/pan/select so charts
# can't get scrolled/dragged into a confusing state.
# On mobile, go fully static instead: Plotly's own drag/hover layer otherwise
# captures touch input, which blocks the browser's native pinch-to-zoom. With
# staticPlot, that layer is gone, hover/drag is out (nothing to accidentally
# trigger with a thumb), and pinch-zoom on the rendered chart works again.
PLOTLY_CONFIG = {
    'displaylogo': False,
    'modeBarButtonsToRemove': [
        'zoom2d', 'pan2d', 'select2d', 'lasso2d', 'zoomIn2d', 'zoomOut2d',
    ],
    'staticPlot': IS_MOBILE,
}


def _lay(height=400, **kw):
    """Build a Plotly layout dict with black fonts and transparent background.
    Pass xaxis/yaxis/coloraxis inside kw — they will be merged, not duplicated."""
    _ax = dict(
        tickfont=dict(color='#000000', size=AXIS_FONT_SIZE),
        title=dict(font=dict(color='#000000', size=AXIS_FONT_SIZE)),
        gridcolor=GRID,
        zerolinecolor=GRID,
    )
    _colorax = dict(
        colorbar=dict(
            tickfont=dict(color='#000000', size=AXIS_FONT_SIZE),
            title=dict(font=dict(color='#000000', size=AXIS_FONT_SIZE)),
        )
    )
    base = dict(
        font=dict(family='Inter, system-ui, sans-serif', size=15, color='#000000'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=20, t=30, b=80),
        height=height,
        legend=dict(
            orientation='h',
            x=0.5, xanchor='center',
            y=-0.22, yanchor='top',
            font=dict(color='#000000', size=LEGEND_FONT_SIZE),
            bgcolor='rgba(0,0,0,0)',
            # Click a legend entry to isolate it (fades/hides the rest) instead of
            # removing just that one series — double-click to restore everything.
            itemclick='toggleothers',
            itemdoubleclick='toggle',
        ),
        xaxis=dict(**_ax),
        yaxis=dict(**_ax),
        coloraxis=dict(**_colorax),
    )
    # Merge caller-provided overrides rather than replacing
    for ax_key in ('xaxis', 'yaxis', 'coloraxis', 'coloraxis_colorbar', 'legend'):
        if ax_key in kw:
            merged = dict(base.get(ax_key, {}))
            merged.update(kw.pop(ax_key))
            base[ax_key] = merged
    base.update(kw)
    return base


# ══════════════════════════════════════════════════════════════════════════
# TAB: Project Information  (light — premise + version history, no live data)
# ══════════════════════════════════════════════════════════════════════════

_VERSION_HISTORY = [
    ("v1", "Built the ingestion pipeline and produced the first quality-gated corpus."),
    ("v2", "Benchmarked retrieval quality against a hand-labeled test set."),
    ("v3", "Measured whether better retrieval actually produces better answers."),
    ("v4", "Improved chunking and repaired corpus quality issues."),
    ("v5", "Fixed an eval bug, tested HyDE/reranking, adopted a smarter PDF extractor, cut low-value features."),
]


def tab_project_info(_data):
    tab_explainer("Project Information")

    sl("PROJECT VERSION HISTORY")
    for ver, desc in _VERSION_HISTORY:
        st.markdown(f"**{ver}** — {desc}")


# ══════════════════════════════════════════════════════════════════════════
# TAB: Overview  (merged with Pipeline)
# ══════════════════════════════════════════════════════════════════════════

def tab_overview(data):
    ps = data.get('pipeline', {})
    t = ps.get('totals', {})

    tab_explainer("Pipeline Overview")

    # ── Top-level KPIs ────────────────────────────────────────────────────
    # Each rate uses the denominator for the stage it actually measures, not a blanket
    # docs-ingested denominator — e.g. DQ Rejection Rate is of docs that reached the DQ
    # gate (Extracted), not of all 1,786 ever downloaded.
    extraction_pct = t.get('extraction_success_rate', 0)
    dq_rejection_pct = _safe_div_pct(t.get('dq_rejected', 0), t.get('extracted', 0))
    llm_rejection_pct = _safe_div_pct(t.get('llm_rejected', 0), t.get('dq_passed', 0))

    # Same source as Dataset Quality / Retrieval Quality (unique docs actually in the
    # LanceDB vector index) — not the DuckDB vectorization_status flag, which can drift
    # a few docs out of sync. One number for "docs vectorized," used everywhere.
    docs_vectorized = len(data.get('topic_map', {}).get('docs', [])) or t.get('vectorized', 0)

    c = st.columns(5)
    with c[0]: mc("DOCS INGESTED", t.get('content', 0))
    with c[1]: mc("DOCS VECTORIZED", docs_vectorized)
    with c[2]: mc("EXTRACTION SUCCESS RATE", f"{extraction_pct:.0f}%")
    with c[3]: mc("DQ REJECTION RATE", f"{dq_rejection_pct:.0f}%", "r")
    with c[4]: mc("LLM REJECTION RATE", f"{llm_rejection_pct:.0f}%", "r")

    # ── Sankey — same pipeline, as flows instead of a narrowing funnel ─────
    sl("PIPELINE FLOW")
    so("How documents move through each stage — the surviving path runs along the top, and each drop-off "
       "(extraction failure, DQ rejection, LLM rejection) peels off below the stage where it happens.")

    total_dl = t.get('content', 0) or 1
    # Canonical keys — used to wire up links below — stay full-length regardless
    # of device; only the on-node display text shortens for mobile.
    node_names = [
        'Downloaded', 'Extracted', 'Extraction Failed', 'DQ Passed', 'DQ Rejected',
        'Approved', 'LLM Rejected', 'Docs Vectorized',
    ]
    node_display = (
        # 'Approved' and 'Docs Vectorized' sit close together at the right edge —
        # keep those two extra short so their labels don't run into each other.
        ['DL', 'Extracted', 'Extr. Failed', 'DQ Passed', 'DQ Rejected',
         'Appr', 'LLM Rejected', 'Vec']
        if IS_MOBILE else node_names
    )
    node_counts = [
        t.get('content', 0), t.get('extracted', 0), t.get('failed_extraction', 0),
        t.get('dq_passed', 0), t.get('dq_rejected', 0), t.get('approved', 0),
        t.get('llm_rejected', 0), docs_vectorized,
    ]
    idx = {name: i for i, name in enumerate(node_names)}
    # Percent is dropped on mobile — narrower nodes don't have room for a third
    # line, and the count alone is still legible at a glance.
    labels = (
        [f"{n}<br>{c:,}" for n, c in zip(node_display, node_counts)] if IS_MOBILE else
        [f"{n}<br>{c:,} ({_safe_div_pct(c, total_dl):.0f}%)" for n, c in zip(node_display, node_counts)]
    )

    # Fixed layout: the surviving chain (Downloaded -> ... -> Docs Vectorized) pinned along
    # the top (low y); each drop-off node sits below the stage it branches from.
    node_x = [0.001, 0.24, 0.24, 0.49, 0.49, 0.74, 0.74, 0.999]
    node_y = [0.05, 0.05, 0.55, 0.05, 0.45, 0.05, 0.35, 0.05]
    node_colors = ['#4e79a7', '#59a14f', '#9c755f', '#59a14f', '#e15759',
                   '#59a14f', '#e15759', '#59a14f']

    links = [
        ('Downloaded', 'Extracted', t.get('extracted', 0), 'good'),
        ('Downloaded', 'Extraction Failed', t.get('failed_extraction', 0), 'bad'),
        ('Extracted', 'DQ Passed', t.get('dq_passed', 0), 'good'),
        ('Extracted', 'DQ Rejected', t.get('dq_rejected', 0), 'bad'),
        ('DQ Passed', 'Approved', t.get('approved', 0), 'good'),
        ('DQ Passed', 'LLM Rejected', t.get('llm_rejected', 0), 'bad'),
        ('Approved', 'Docs Vectorized', docs_vectorized, 'good'),
    ]
    link_color = {'good': 'rgba(89,161,79,0.35)', 'bad': 'rgba(225,87,89,0.35)'}
    link_pct = [_safe_div_pct(v, total_dl) for _, _, v, _ in links]
    link_labels = [f"{v:,} ({p:.0f}% of downloaded)" for (_, _, v, _), p in zip(links, link_pct)]

    fig_sankey = go.Figure(go.Sankey(
        orientation='h',
        arrangement='fixed',
        node=dict(
            label=labels, color=node_colors, x=node_x, y=node_y,
            pad=16 if IS_MOBILE else 28, thickness=20,
            line=dict(color=BDR, width=1),
        ),
        link=dict(
            source=[idx[s] for s, _, _, _ in links],
            target=[idx[tgt] for _, tgt, _, _ in links],
            value=[v for _, _, v, _ in links],
            color=[link_color[kind] for _, _, _, kind in links],
            customdata=link_labels,
            hovertemplate='%{source.label} → %{target.label}<br>%{customdata}<extra></extra>',
        ),
        textfont=dict(color='#000000', size=10 if IS_MOBILE else 14),
    ))
    # Taller on mobile despite the narrower width — the node labels wrap to more
    # lines when there's less horizontal room, so they need more vertical space
    # to avoid overlapping the row below.
    fig_sankey.update_layout(**_lay(height=360 if IS_MOBILE else 280, margin=dict(l=10, r=10, t=15, b=10)))
    st.plotly_chart(fig_sankey, use_container_width=True, config=PLOTLY_CONFIG)

    by_type = ps.get('by_source_type', [])

    # ── Corpus composition + Drop-off by format ────────────────────────────
    cl, cr = st.columns(2)
    with cl:
        sl("CORPUS COMPOSITION")
        so("Where approved content is coming from — a balanced mix reduces over-reliance on any single source type.")
        if by_type:
            agg = pd.DataFrame(by_type).groupby('content_type')['count'].sum().reset_index()
            fig = px.pie(agg, values='count', names='content_type',
                         color_discrete_sequence=MULTI, hole=0.45)
            fig.update_layout(**_lay(height=380, showlegend=True,
                                     legend=dict(orientation='h', y=-0.15, font=dict(size=LEGEND_FONT_SIZE)),
                                     legend_title_text=''))
            fig.update_traces(textinfo='percent+label', textposition='outside', textfont_size=13,
                              texttemplate='%{label}<br>%{percent:.0%}')
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    with cr:
        sl("DROP-OFF BY FORMAT")
        so("Where each format actually loses documents on the way to the final corpus — sorted by survival rate, worst first.")
        if by_type:
            dof_df = pd.DataFrame(by_type).copy()
            dof_df['survival_rate'] = dof_df.apply(lambda r: _safe_div_pct(r['approved'], r['count']), axis=1)
            dof_df = dof_df.sort_values('survival_rate', ascending=True)
            for col in ['approved', 'llm_rejected', 'dq_rejected', 'extraction_failed']:
                dof_df[f'{col}_pct'] = dof_df.apply(lambda r, c=col: _safe_div_pct(r[c], r['count']), axis=1)

            fig = go.Figure()
            for col, name, color in [
                ('approved', 'Approved', '#59a14f'),
                ('llm_rejected', 'LLM Rejected', '#f28e2b'),
                ('dq_rejected', 'DQ Rejected', '#e15759'),
                ('extraction_failed', 'Extraction Failed', '#9c755f'),
            ]:
                # Same rule as Top Sources: below 35% width, Plotly rotates cramped
                # inside-text 90° rather than shrink it further — worse than blank.
                if IS_MOBILE:
                    seg_text = [f"{v:,.0f} ({p:.0f}%)" if p >= 35 else ''
                                for v, p in zip(dof_df[col], dof_df[f'{col}_pct'])]
                else:
                    seg_text = [f"{v:,.0f} ({p:.0f}%)" for v, p in zip(dof_df[col], dof_df[f'{col}_pct'])]
                fig.add_trace(go.Bar(
                    y=dof_df['content_type'], x=dof_df[f'{col}_pct'], name=name,
                    orientation='h', marker_color=color,
                    text=seg_text,
                    textposition='inside', textfont=dict(color='#ffffff', size=10 if IS_MOBILE else 12),
                ))
            fig.update_layout(**_lay(
                height=max(380, len(dof_df) * 48), barmode='stack',
                xaxis=dict(title='% of documents', range=[0, 100]), yaxis=dict(title=''),
                legend=dict(),
            ))
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    # ── Rejection reasons (categorized) + Errors ──────────────────────────
    reasons = ps.get('rejection_reasons', [])
    a, b = st.columns(2)
    with a:
        sl("REJECTION REASONS")
        so("Why documents are filtered out — a shift toward 'Out of Scope' over 'Low Quality' signals a smarter gate.")
        if reasons:
            agg_reasons = _aggregate_rejection_reasons(reasons)
            fig = px.bar(agg_reasons, x='count', y='category',
                         orientation='h', color_discrete_sequence=['#4e79a7'],
                         text='count')
            fig.update_traces(textposition='outside', textfont_color='#000000')
            fig.update_layout(**_lay(height=max(240, len(agg_reasons) * 52),
                                     yaxis=dict(autorange='reversed', title=''),
                                     xaxis=dict(title='Documents Rejected')))
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
        else:
            st.caption("No rejection data.")

    with b:
        sl("ERRORS (LAST 7 DAYS)")
        so("Recent pipeline failures by stage — a healthy pipeline should trend toward zero errors over time.")
        errors_7d = data.get('logs', {}).get('error_summary_7d', [])
        if errors_7d:
            fig = px.bar(pd.DataFrame(errors_7d), x='count', y='action', color='source',
                         orientation='h', color_discrete_sequence=MULTI, text='count')
            fig.update_traces(textposition='outside', textfont_color='#000000')
            fig.update_layout(**_lay(height=max(240, len(errors_7d) * 40),
                                     yaxis=dict(autorange='reversed', title=''),
                                     xaxis=dict(title='Error Count'),
                                     legend_title_text=''))
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
        else:
            st.caption("No errors in the last 7 days.")


# ══════════════════════════════════════════════════════════════════════════
# TAB: Sources  (improved legibility and usefulness)
# ══════════════════════════════════════════════════════════════════════════

def tab_sources(data):
    cq = data.get('corpus', {})

    # Top sources — approved vs rejected stacked + acceptance rate
    sl("TOP SOURCES — VOLUME & ACCEPTANCE")
    so("How selective the pipeline is about each source — a low acceptance rate may mean the source is noisy or off-domain.")
    ts = cq.get('top_sources', [])
    if ts:
        tsdf = pd.DataFrame(ts).head(20)
        tsdf['accept_pct'] = (tsdf['approved'] / tsdf['count'] * 100).round(0)
        tsdf['rejected_count'] = tsdf['count'] - tsdf['approved']
        tsdf['rejected_pct'] = 100 - tsdf['accept_pct']
        tsdf = tsdf.sort_values('accept_pct', ascending=True)

        # On mobile the plot area is only ~330px, so a narrow segment doesn't have
        # room for its text — Plotly's 'inside' textposition then rotates it 90°
        # to force a fit, which reads worse than no label at all (tried a percent-
        # only middle tier first; even "23%" alone still got rotated below ~25%
        # width, so there's no legible partial label — it's full text or nothing).
        # The opposing segment's label already implies the split either way.
        def _seg_text(vals, pcts, fmt):
            if not IS_MOBILE:
                return [fmt(v, p) for v, p in zip(vals, pcts)]
            return [fmt(v, p) if p >= 35 else '' for v, p in zip(vals, pcts)]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=tsdf['source_name'], x=tsdf['accept_pct'],
            name='Approved', orientation='h',
            marker_color='#59a14f', opacity=0.9,
            text=_seg_text(tsdf['approved'], tsdf['accept_pct'], lambda a, p: f"{a:.0f} ({p:.0f}%)"),
            textposition='inside',
            textfont=dict(color='#ffffff', size=10 if IS_MOBILE else 13),
        ))
        fig.add_trace(go.Bar(
            y=tsdf['source_name'], x=tsdf['rejected_pct'],
            name='Rejected', orientation='h',
            marker_color='#e15759', opacity=0.9,
            text=_seg_text(tsdf['rejected_count'], tsdf['rejected_pct'], lambda r, p: f"{r:.0f} ({p:.0f}%)"),
            textposition='inside',
            textfont=dict(color='#ffffff', size=10 if IS_MOBILE else 13),
        ))
        fig.update_layout(**_lay(
            height=max(480, len(tsdf) * 36),
            barmode='stack',
            legend=dict(),
            xaxis=dict(title='% of documents', range=[0, 100]),
            yaxis=dict(title=''),
        ))
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    else:
        st.caption("No top sources data available.")


# ══════════════════════════════════════════════════════════════════════════
# TAB: Trends
# ══════════════════════════════════════════════════════════════════════════

def tab_trends(data):
    tmp = data.get('temporal', {})

    cl, cr = st.columns(2)
    with cl:
        sl("CONTENT OVER TIME")
        so("Document inflow rate by source type — are you keeping pace with activity in the domain?")
        ct = tmp.get('content_timeline', [])
        if ct:
            df = pd.DataFrame(ct)
            df['date'] = pd.to_datetime(df['date'])
            fig = px.bar(df, x='date', y='count', color='source_type',
                         barmode='stack', color_discrete_sequence=MULTI)
            fig.update_layout(**_lay(height=310, xaxis=dict(title=''), yaxis=dict(title='Documents'),
                                     legend_title_text=''))
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    with cr:
        sl("CUMULATIVE CONTENT")
        so("Total corpus growth trajectory — the slope tells you whether discovery is accelerating or plateauing.")
        cu = tmp.get('cumulative_content', [])
        if cu:
            cdf = pd.DataFrame(cu)
            cdf['date'] = pd.to_datetime(cdf['date'])
            cdf = cdf.sort_values('date')
            cdf['cumulative'] = cdf['count'].cumsum()
            fig = px.area(cdf, x='date', y='cumulative', color_discrete_sequence=['#4e79a7'])
            fig.update_layout(**_lay(height=310, xaxis=dict(title=''),
                                     yaxis=dict(title='Cumulative')))
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    sl("SCREENING OVER TIME")
    so("Approval vs. rejection balance as the pipeline matures — rising approval rate means the LLM gate is better calibrated.")
    sc = tmp.get('screening_timeline', [])
    if sc:
        scdf = pd.DataFrame(sc)
        scdf['date'] = pd.to_datetime(scdf['date'])
        fig = go.Figure()
        fig.add_trace(go.Bar(x=scdf['date'], y=scdf['approved'],
                             name='Approved', marker_color='#59a14f'))
        fig.add_trace(go.Bar(x=scdf['date'], y=scdf['rejected'],
                             name='Rejected', marker_color='#e15759'))
        fig.update_layout(**_lay(barmode='stack', height=300,
                                 legend=dict(),
                                 xaxis=dict(title=''), yaxis=dict(title='Count')))
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)


# ══════════════════════════════════════════════════════════════════════════
# TAB: Corpus Map  (doc-level topic clusters from embeddings + eval queries
# projected into the same space, colored by retrieval hit/miss)
# ══════════════════════════════════════════════════════════════════════════

def tab_corpus_map(data):
    tm = data.get('topic_map', {})
    docs = tm.get('docs', [])
    clusters = tm.get('clusters', [])

    tab_explainer("Dataset Quality")

    if not docs:
        st.caption("No topic map data available — run workflows/export_topic_map.py "
                   "(needs the project's ML env: lancedb/sentence-transformers/umap-learn).")
        return

    doc_df = pd.DataFrame(docs)
    cl_df = pd.DataFrame(clusters).sort_values('doc_count', ascending=False)
    largest = cl_df.iloc[0]
    n_formats = doc_df['content_type'].nunique()

    n_near_dup = int(doc_df['is_near_dup'].sum()) if 'is_near_dup' in doc_df.columns else 0

    c = st.columns(5)
    with c[0]: mc("DOCS IN VECTOR INDEX", f"{len(doc_df):,}")
    with c[1]: mc("TOPIC CLUSTERS", len(clusters))
    with c[2]: mc("LARGEST TOPIC", largest['label'], small=True)
    with c[3]: mc("FORMATS SPANNED", n_formats)
    with c[4]: mc("LIKELY DUPLICATES", f"{n_near_dup}/{len(doc_df)}", "r" if n_near_dup else "g")

    sl("CORPUS TOPIC MAP")
    so("Each point is a document, positioned by semantic similarity (UMAP over mean-pooled chunk embeddings) "
       "and colored by an auto-discovered topic cluster — no manual tagging, just what the embeddings found in common.")

    n_clusters = len(clusters)
    palette = MULTI if n_clusters <= len(MULTI) else (MULTI * (n_clusters // len(MULTI) + 1))

    fig = px.scatter(
        doc_df, x='x', y='y', color='cluster_label',
        color_discrete_sequence=palette,
        hover_data={'title': True, 'content_type': True, 'x': False, 'y': False, 'cluster_label': False},
        opacity=0.75,
    )
    # Smaller markers on mobile so overlapping points in dense clusters separate a
    # little instead of fusing into a solid blob at a narrower chart width.
    fig.update_traces(marker=dict(size=5 if IS_MOBILE else 7, line=dict(width=0)))
    fig.update_layout(**_lay(
        height=620,
        xaxis=dict(title='', showticklabels=False, zeroline=False),
        yaxis=dict(title='', showticklabels=False, zeroline=False),
        legend_title_text='',
    ))
    # On desktop, hover is essential — 846 points can't all be direct-labeled. On
    # mobile this chart goes static (see PLOTLY_CONFIG) so hover isn't available;
    # the per-cluster breakdown and CLUSTER SIZES table below cover that gap.
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    cl, cr = st.columns([3, 2])
    with cl:
        sl("FORMATS WITHIN EACH TOPIC")
        so("The same topic pulls together PDFs, web pages, and other formats — proof the grouping is by meaning, not file type.")
        fmt_df = doc_df.groupby(['cluster_label', 'content_type']).size().reset_index(name='count')
        fmt_df = fmt_df.merge(cl_df[['label', 'doc_count']], left_on='cluster_label', right_on='label')
        fmt_df = fmt_df.sort_values('doc_count', ascending=True)
        fig2 = px.bar(fmt_df, x='count', y='cluster_label', color='content_type',
                      orientation='h', color_discrete_sequence=MULTI)
        fig2.update_layout(**_lay(
            height=max(360, len(clusters) * 34), barmode='stack',
            xaxis=dict(title='Documents'), yaxis=dict(title=''),
            legend_title_text='',
        ))
        st.plotly_chart(fig2, use_container_width=True, config=PLOTLY_CONFIG)

    with cr:
        sl("CLUSTER SIZES")
        so("Table view — useful when colors get hard to tell apart at a glance.")
        table_df = cl_df.rename(columns={'label': 'Topic', 'doc_count': 'Documents'})[['Topic', 'Documents']]
        st.dataframe(table_df, use_container_width=True, hide_index=True, height=max(360, len(clusters) * 34))

    st.divider()
    sl("REDUNDANCY BY TOPIC")
    dup_pct_threshold = tm.get('near_dup_threshold')
    threshold_note = f" (nearest-neighbor cosine similarity ≥ {dup_pct_threshold:.3f}, this corpus's own 95th percentile)" if dup_pct_threshold else ""
    so(f"Share of documents in each topic that are likely near-duplicates of another document{threshold_note} — "
       f"this is where a file share is bloated with copies, not where the knowledge itself is thin.")
    dup_cl_df = cl_df.sort_values('near_dup_pct', ascending=True)
    fig3 = px.bar(dup_cl_df, x='near_dup_pct', y='label', orientation='h',
                  color_discrete_sequence=['#e15759'],
                  text=dup_cl_df['near_dup_pct'].map(lambda v: f"{v:.0f}%"))
    fig3.update_traces(textposition='outside', textfont=dict(color='#000000', size=12))
    fig3.update_layout(**_lay(
        height=max(320, len(clusters) * 32),
        xaxis=dict(title='% of docs flagged as likely duplicates', range=[0, max(dup_cl_df['near_dup_pct'].max() * 1.3, 10)]),
        yaxis=dict(title=''),
    ))
    st.plotly_chart(fig3, use_container_width=True, config=PLOTLY_CONFIG)

    dup_pairs = tm.get('duplicate_pairs', [])
    if dup_pairs:
        sl("LIKELY DUPLICATE PAIRS")
        so("The most similar document pairs corpus-wide — same content re-hosted, re-titled, or reissued across years/formats.")
        pairs_df = pd.DataFrame(dup_pairs)[['title_a', 'title_b', 'similarity']].rename(
            columns={'title_a': 'Document A', 'title_b': 'Document B', 'similarity': 'Similarity'})
        pairs_df['Similarity'] = pairs_df['Similarity'].round(4)
        st.dataframe(pairs_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════
# TAB: System Logs  (Discovery + Logs merged)
# ══════════════════════════════════════════════════════════════════════════

def tab_system_logs(data):
    disc = data.get('discovery', {})
    logs_data = data.get('logs', {})

    tab_explainer("System Logs")

    sl("DISCOVERY RUNS")
    so("How many candidates were found and screened in each cycle — declining found-to-approved ratio means noisier discovery queries.")
    disc_runs = disc.get('discovery_runs', [])
    if disc_runs:
        ddf = pd.DataFrame(disc_runs)

        if 'approved' in ddf.columns and 'found' in ddf.columns:
            c = st.columns(4)
            total_found    = ddf['found'].sum()
            total_approved = ddf['approved'].sum()
            total_rejected = ddf['rejected'].sum() if 'rejected' in ddf.columns else 0
            with c[0]: mc("TOTAL RUNS", len(ddf))
            with c[1]: mc("TOTAL FOUND", int(total_found))
            with c[2]: mc("TOTAL APPROVED", int(total_approved))
            with c[3]: mc("TOTAL REJECTED", int(total_rejected), "r")

            sl("DISCOVERY YIELD OVER TIME")
            so("Approval rate trend across runs — a rising green-to-red ratio means your discovery queries are becoming more targeted.")
            ddf['timestamp'] = pd.to_datetime(ddf['timestamp'])
            fig = go.Figure()
            fig.add_trace(go.Bar(x=ddf['timestamp'], y=ddf['approved'],
                                 name='Approved', marker_color='#59a14f'))
            fig.add_trace(go.Bar(x=ddf['timestamp'], y=ddf['rejected'],
                                 name='Rejected', marker_color='#e15759'))
            fig.update_layout(**_lay(barmode='stack', height=300,
                                     legend=dict(),
                                     xaxis=dict(title=''), yaxis=dict(title='Candidates')))
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

        st.dataframe(ddf, use_container_width=True, hide_index=True)
    else:
        st.caption("No discovery run data available.")

    ingestion = disc.get('ingestion_logs', [])
    if ingestion:
        sl("INGESTION LOGS")
        so("Raw record of what entered the pipeline — use this to audit unexpected content or trace a specific document's origin.")
        st.dataframe(pd.DataFrame(ingestion), use_container_width=True, hide_index=True)

    st.divider()

    sl("RECENT PIPELINE RUNS")
    so("Last N pipeline executions — repeated errors in the same stage indicate a systemic issue worth investigating.")
    runs = logs_data.get('runs', [])
    if runs:
        rdf = pd.DataFrame(runs)
        if 'errors' in rdf.columns:
            rdf['status'] = rdf['errors'].apply(lambda x: 'ERRORS' if x > 0 else 'OK')
        st.dataframe(rdf, use_container_width=True, hide_index=True)

    sl("RECENT SYSTEM LOGS")
    so("Full log stream filterable by severity — ERROR and WARNING entries are the most actionable starting points.")
    recent = logs_data.get('recent_logs', [])
    if recent:
        ldf = pd.DataFrame(recent)
        levels = (['All'] + sorted(ldf['level'].dropna().unique().tolist())
                  if 'level' in ldf.columns else ['All'])
        sel_level = st.selectbox("Log Level", levels)
        if sel_level != 'All':
            ldf = ldf[ldf['level'] == sel_level]

        st.caption(f"Showing {len(ldf):,} log entries")
        display_cols = ['timestamp', 'level', 'source', 'action', 'message']
        display_cols = [c for c in display_cols if c in ldf.columns]
        st.dataframe(ldf[display_cols].head(100), use_container_width=True, hide_index=True)
    else:
        st.caption("No system logs available.")


# ══════════════════════════════════════════════════════════════════════════
# TAB: Retrieval Quality  (v5 corrected + HyDE — hardcoded from Notebook 3,
# updated after the chunk-vs-document aggregation bug fix and the HyDE/pool-size
# experiment series; recall curve, query-type, and chunking ablation below are
# still the earlier v2 snapshot and have not been recomputed on the corrected
# pipeline yet)
# ══════════════════════════════════════════════════════════════════════════

# v5 corrected metrics (Notebook 3, 31 labeled queries — 27 resolved, nomic-embed-text-v1.5, 92,705 chunks)
_RQ_CONFIGS = [
    {'config': 'Dense',                    'mrr': 0.399, 'recall5': 0.317, 'recall10': 0.403},
    {'config': 'Hybrid (BM25+RRF)',        'mrr': 0.442, 'recall5': 0.350, 'recall10': 0.453},
    {'config': 'Reranked',                 'mrr': 0.426, 'recall5': 0.337, 'recall10': 0.473},
    {'config': 'HyDE + Dense',             'mrr': 0.416, 'recall5': 0.367, 'recall10': 0.475},
    {'config': 'HyDE + Hybrid',            'mrr': 0.489, 'recall5': 0.335, 'recall10': 0.438},
    {'config': 'HyDE + Hybrid + Reranked', 'mrr': 0.381, 'recall5': 0.293, 'recall10': 0.420},
]

# MRR at each project milestone — for the progression chart below.
# Only MRR is tracked across every milestone (Recall@5 wasn't recorded pre-v5).
_RQ_TIMELINE = [
    {'label': 'Dense Baseline',         'mrr': 0.307, 'milestone': None},
    {'label': '+ Hybrid + Reranker',    'mrr': 0.374, 'milestone': True},
    {'label': '+ Contextual Chunking',  'mrr': 0.412, 'milestone': True},
    {'label': '+ HyDE',                 'mrr': 0.489, 'milestone': True},
]
_RQ_RECALL_CURVE = [
    {'k': 1,  'recall': 0.110},
    {'k': 3,  'recall': 0.210},
    {'k': 5,  'recall': 0.300},
    {'k': 10, 'recall': 0.448},
    {'k': 20, 'recall': 0.511},
]
_RQ_BY_TYPE = [
    {'type': 'temporal',        'recall10': 0.00},
    {'type': 'broad_topical',   'recall10': 0.25},
    {'type': 'cross_domain',    'recall10': 0.40},
    {'type': 'entity_specific', 'recall10': 0.45},
    {'type': 'comparative',     'recall10': 0.50},
    {'type': 'factual_lookup',  'recall10': 0.65},
    {'type': 'niche_specific',  'recall10': 0.75},
]
_ABLATION = [
    {'max_chars': 300,  'n_chunks': 89203, 'mean_chars': 400,  'std_chars': 698,  'recall5': 0.167, 'top_sim': 0.779},
    {'max_chars': 600,  'n_chunks': 44135, 'mean_chars': 673,  'std_chars': 946,  'recall5': 0.210, 'top_sim': 0.762},
    {'max_chars': 900,  'n_chunks': 27289, 'mean_chars': 982,  'std_chars': 1154, 'recall5': 0.210, 'top_sim': 0.744},
    {'max_chars': 1200, 'n_chunks': 19649, 'mean_chars': 1286, 'std_chars': 1313, 'recall5': 0.143, 'top_sim': 0.726},
]


def tab_retrieval_quality(data):
    tab_explainer("Retrieval Quality")

    # Corpus doc count comes from the same source as the Dataset Quality tab
    # (unique docs in the LanceDB vector index) so the two can't drift apart.
    corpus_docs = len(data.get('topic_map', {}).get('docs', [])) or "N/A"

    # ── KPI cards ─────────────────────────────────────────────────────────
    c = st.columns(5)
    with c[0]: mc("CORPUS DOCS", f"{corpus_docs:,}" if isinstance(corpus_docs, int) else corpus_docs)
    with c[1]: mc("CHUNKS", "92,705")
    with c[2]: mc("LABELED QUERIES", "31")
    with c[3]: mc("BEST MRR", "0.489")
    with c[4]: mc("BEST RECALL@10", "0.475")

    # ── Config comparison bar chart ────────────────────────────────────────
    sl("RETRIEVAL CONFIGURATION COMPARISON")
    so("HyDE + Hybrid wins MRR (0.489, best overall); HyDE + Dense wins Recall@10 (0.475). Reranking on top of HyDE hurts both — a query/candidate phrasing mismatch.")

    cfg_df = pd.DataFrame(_RQ_CONFIGS)
    labels = cfg_df['config'].tolist()

    best_mrr_idx = int(cfg_df['mrr'].idxmax())

    fig2 = go.Figure()
    colors = ['#4C72B0', '#55A868', '#C44E52']
    metrics_list = [('mrr', 'MRR'), ('recall5', 'Recall@5'), ('recall10', 'Recall@10')]
    for col, (metric, name) in zip(colors, metrics_list):
        vals = cfg_df[metric].tolist()
        is_mrr = metric == 'mrr'
        fig2.add_trace(go.Bar(
            name=name, x=labels, y=vals,
            marker=dict(
                color=col, opacity=0.88,
                line=dict(
                    color=['#000000' if (is_mrr and i == best_mrr_idx) else 'rgba(0,0,0,0)' for i in range(len(vals))],
                    width=[3 if (is_mrr and i == best_mrr_idx) else 0 for i in range(len(vals))],
                ),
            ),
            text=[f'{v:.3f}' for v in vals],
            textposition='outside',
            textfont=dict(
                color=['#000000'] * len(vals) if not is_mrr else
                      ['#000000' if i != best_mrr_idx else '#000000' for i in range(len(vals))],
                size=[10 if not (is_mrr and i == best_mrr_idx) else 12 for i in range(len(vals))],
            ),
        ))
    fig2.add_annotation(
        x=labels[best_mrr_idx], y=cfg_df['mrr'].iloc[best_mrr_idx],
        text='★ best MRR', showarrow=True, arrowhead=0, ax=0, ay=-32,
        font=dict(color='#000000', size=13), bgcolor='#FFFFFF', bordercolor='#000000', borderwidth=1,
    )
    fig2.update_layout(**_lay(
        height=460, barmode='group',
        xaxis=dict(tickangle=-15),
        yaxis=dict(range=[0, 0.65], title='Score'),
        margin=dict(l=40, r=20, t=50, b=110),
    ))
    st.plotly_chart(fig2, use_container_width=True, config={'staticPlot': True})

    # ── MRR progression over project milestones ────────────────────────────
    sl("MRR PROGRESSION AS FEATURES WERE ADDED")
    so("Best recorded MRR at each stage — each label is cumulative (each stage includes everything to its left).")

    tl_df = pd.DataFrame(_RQ_TIMELINE)
    fig_tl = go.Figure()
    fig_tl.add_trace(go.Scatter(
        x=tl_df['label'], y=tl_df['mrr'],
        mode='lines+markers+text',
        line=dict(color='#4C72B0', width=2.5),
        marker=dict(size=10, color='#4C72B0', line=dict(color='#FFFFFF', width=2)),
        text=[f'{v:.3f}' for v in tl_df['mrr']],
        textposition='top center',
        textfont=dict(color='#000000', size=13),
    ))
    _TL_YMIN, _TL_YMAX = 0.28, 0.54
    for row in _RQ_TIMELINE:  # raw list, not the DataFrame — avoids None→NaN coercion in a mixed-type column
        if row['milestone']:
            fig_tl.add_shape(
                type='line', x0=row['label'], x1=row['label'], y0=_TL_YMIN, y1=_TL_YMAX,
                line=dict(color='#C44E52', width=1.5, dash='dash'),
            )
    fig_tl.update_layout(**_lay(
        height=380,
        xaxis=dict(title=''),
        yaxis=dict(title='MRR', range=[_TL_YMIN, _TL_YMAX]),
        showlegend=False,
    ))
    st.plotly_chart(fig_tl, use_container_width=True, config={'staticPlot': True})

    # ── Recall curve + by query type ──────────────────────────────────────
    cl, cr = st.columns(2)

    with cl:
        sl("RECALL CURVE (DENSE)")
        so("Still rising at k=20 — relevant docs exist in the corpus but rank outside top 10. Ranking is the dominant failure mode.")
        rc_df = pd.DataFrame(_RQ_RECALL_CURVE)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=rc_df['k'], y=rc_df['recall'],
            mode='lines+markers+text',
            line=dict(color='#4C72B0', width=2.5),
            marker=dict(size=9, color='#4C72B0'),
            text=[f'{v:.3f}' for v in rc_df['recall']],
            textposition='top center',
            textfont=dict(color='#000000', size=12),
        ))
        fig.update_layout(**_lay(
            height=320,
            xaxis=dict(title='k', tickvals=[1, 3, 5, 10, 20]),
            yaxis=dict(title='Mean Recall@k', range=[0, 0.65]),
        ))
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    with cr:
        sl("RECALL@10 BY QUERY TYPE")
        so("Temporal queries score 0 — no recency signal in dense embeddings. Niche and factual queries perform best.")
        qt_df = pd.DataFrame(_RQ_BY_TYPE).sort_values('recall10')
        fig = go.Figure(go.Bar(
            x=qt_df['recall10'], y=qt_df['type'],
            orientation='h',
            marker_color=['#e15759' if v == 0 else '#4e79a7' for v in qt_df['recall10']],
            text=[f'{v:.2f}' for v in qt_df['recall10']],
            textposition='outside',
            textfont=dict(color='#000000', size=12),
        ))
        fig.update_layout(**_lay(
            height=320,
            xaxis=dict(title='Mean Recall@10', range=[0, 1]),
            yaxis=dict(title=''),
        ))
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    # ── Chunking ablation ─────────────────────────────────────────────────
    st.divider()
    sl("CHUNKING ABLATION — max_chars GRID")
    so("max_chars=600 wins on Recall@5. Smaller chunks raise similarity but split context; larger chunks dilute the signal.")

    abl_df = pd.DataFrame(_ABLATION)
    cl2, cr2 = st.columns(2)

    with cl2:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[str(r['max_chars']) for r in _ABLATION],
            y=[r['top_sim'] for r in _ABLATION],
            name='top_sim', marker_color='#4C72B0', opacity=0.85,
            text=[f"{r['top_sim']:.3f}" for r in _ABLATION],
            textposition='outside', textfont=dict(color='#000000', size=12),
        ))
        fig.add_trace(go.Bar(
            x=[str(r['max_chars']) for r in _ABLATION],
            y=[r['recall5'] for r in _ABLATION],
            name='Recall@5', marker_color='#55A868', opacity=0.85,
            text=[f"{r['recall5']:.3f}" for r in _ABLATION],
            textposition='outside', textfont=dict(color='#000000', size=12),
        ))
        fig.update_layout(**_lay(
            height=320, barmode='group',
            xaxis=dict(title='max_chars'),
            yaxis=dict(title='Score', range=[0, 1]),
            legend=dict(),
        ))
        # Winner = highest Recall@5, tie-broken by lowest std_chars (more consistent chunk size).
        # Uses the string category label directly with add_shape/add_annotation — add_vline's
        # numeric x on a categorical (string-labeled) axis doesn't reliably land on the right bar.
        _winner_row = max(_ABLATION, key=lambda r: (r['recall5'], -r['std_chars']))
        _winner_label = str(_winner_row['max_chars'])
        fig.add_shape(type='line', x0=_winner_label, x1=_winner_label, y0=0, y1=1,
                      line=dict(color='#C44E52', width=2, dash='dash'))
        fig.add_annotation(x=_winner_label, y=1, text='winner', showarrow=False,
                           yshift=10, font=dict(color='#C44E52', size=13))
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    with cr2:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[str(r['max_chars']) for r in _ABLATION],
            y=[r['n_chunks'] for r in _ABLATION],
            marker_color=TABLEAU[:4], opacity=0.85,
            text=[f"{r['n_chunks']:,}" for r in _ABLATION],
            textposition='outside', textfont=dict(color='#000000', size=12),
        ))
        fig.update_layout(**_lay(
            height=320,
            xaxis=dict(title='max_chars'),
            yaxis=dict(title='Total Chunks'),
        ))
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    st.dataframe(
        abl_df.rename(columns={
            'max_chars': 'max_chars', 'n_chunks': 'Chunks',
            'mean_chars': 'Mean Chars', 'std_chars': 'Std Chars',
            'recall5': 'Recall@5', 'top_sim': 'Top Sim',
        }),
        use_container_width=True, hide_index=True,
    )


# ══════════════════════════════════════════════════════════════════════════
# TAB: Answer Quality  (NB4, real agent-graph pipeline — hardcoded from
# notebooks/eval_snapshots.json, updated after each versioned eval run. Same
# pattern as Retrieval Quality above: numbers are point-in-time snapshots,
# not live-queried, so this file is the source of truth until the next run.)
# ══════════════════════════════════════════════════════════════════════════

# v1 -> v3 -> v4 progression on the same 15-query, human-authored test set.
# v2 has no nb4_answer_quality entry (corpus-only snapshot). v3 rewired NB4
# to call the real agent graph (router -> retrieval -> generation) instead
# of a notebook-only bypass; v4 is one scoped prompt fix on top of v3 (see
# "Chat Agent System" in the README for what the fix was and why).
_AQ_TIMELINE = [
    {'label': 'Original Baseline',        'fact_recall': 0.583, 'groundedness': 0.913, 'milestone': None},
    {'label': 'Connected to Real System', 'fact_recall': 0.693, 'groundedness': 0.821, 'milestone': True},
    {'label': 'General-Knowledge Fix',    'fact_recall': 0.833, 'groundedness': 0.947, 'milestone': True},
]

# Config comparison — same 15 queries, four retrieval/generation configs,
# scored the same way (fact-recall + groundedness/hallucination LLM judge).
# C_agent_graph is the real production pipeline; A/B/D are raw retrieval
# ablations that bypass routing and resolved-query resolution entirely.
_AQ_CONFIGS = [
    {'config': 'A: Dense Only',                 'fact_recall': 0.457, 'hallucination_rate': 0.067},
    {'config': 'B: Hybrid',                     'fact_recall': 0.503, 'hallucination_rate': 0.067},
    {'config': 'D: Signals + Hybrid + Rerank',  'fact_recall': 0.537, 'hallucination_rate': 0.000},
    {'config': 'C: Real Agent Graph (prod)',    'fact_recall': 0.693, 'hallucination_rate': 0.000},
]

# Per-query manual groundedness vs. RAGAS faithfulness, v3 snapshot (before
# the general-knowledge fix, same run RAGAS was scored against). Included
# deliberately even though it doesn't flatter either metric — the divergence
# itself is the finding: q13 scores 0.0 manual / 1.0 RAGAS, q21 scores 0.92
# manual / 0.03 RAGAS. Two "rigorous-looking" numbers that actively
# disagree, not just noisy around each other.
_AQ_MANUAL_VS_RAGAS = [
    {'id': 'q01', 'manual': 0.90, 'ragas': 0.062, 'query': 'What is IEC 62443 and what does it cover?'},
    {'id': 'q02', 'manual': 1.00, 'ragas': 0.182, 'query': 'What is OPC Unified Architecture?'},
    {'id': 'q04', 'manual': 0.90, 'ragas': 0.524, 'query': 'What are ISA/IEC 62443 security levels and conduit zones?'},
    {'id': 'q06', 'manual': 1.00, 'ragas': 0.160, 'query': 'What is SCADA and how is it used in industrial process control?'},
    {'id': 'q07', 'manual': 1.00, 'ragas': 0.500, 'query': 'What is Siemens doing in industrial edge AI?'},
    {'id': 'q08', 'manual': 0.90, 'ragas': 0.500, 'query': 'Rockwell Automation cybersecurity products and documentation'},
    {'id': 'q13', 'manual': 0.00, 'ragas': 1.000, 'query': 'How does Modbus compare to OPC UA for industrial communication?'},
    {'id': 'q14', 'manual': 0.90, 'ragas': 0.412, 'query': 'HMI vs SCADA: architecture differences and use cases'},
    {'id': 'q15', 'manual': 0.90, 'ragas': 0.478, 'query': 'Edge computing vs cloud computing for manufacturing analytics'},
    {'id': 'q20', 'manual': 0.90, 'ragas': 0.711, 'query': 'Predictive maintenance using AI in industrial settings'},
    {'id': 'q21', 'manual': 0.92, 'ragas': 0.032, 'query': 'Industrial cybersecurity best practices for OT networks'},
    {'id': 'q23', 'manual': 1.00, 'ragas': 0.333, 'query': 'Security vulnerabilities and attack surface of industrial robots'},
    {'id': 'q24', 'manual': 0.90, 'ragas': 0.000, 'query': 'Object-oriented programming patterns for industrial PLC code'},
    {'id': 'q26', 'manual': 0.90, 'ragas': 0.200, 'query': 'How does zero trust network security apply to industrial OT environments?'},
    {'id': 'q27', 'manual': 0.20, 'ragas': 0.714, 'query': 'Convergence of edge AI and Industry 5.0 in smart manufacturing'},
]


def tab_answer_quality(data):
    tab_explainer("Answer Quality")

    # ── KPI cards ─────────────────────────────────────────────────────────
    c = st.columns(5)
    with c[0]: mc("TEST QUERIES", "15")
    with c[1]: mc("FACT RECALL", "0.833")
    with c[2]: mc("GROUNDEDNESS", "0.947")
    with c[3]: mc("HALLUCINATION RATE", "0%")
    with c[4]: mc("COMPLETENESS", "0.887")

    # ── Fact recall / groundedness progression ──────────────────────────────
    sl("ANSWER QUALITY ACROSS THREE REAL PIPELINE CHANGES")
    so("Same 15 hand-authored test queries at every point. The jump on the right is one scoped fix, found by reading real system traces and verified against the exact failing case before this re-run confirmed it.")

    tl_df = pd.DataFrame(_AQ_TIMELINE)
    fig_aq = go.Figure()
    fig_aq.add_trace(go.Scatter(
        x=tl_df['label'], y=tl_df['fact_recall'],
        mode='lines+markers+text', name='Fact Recall',
        line=dict(color='#4C72B0', width=2.5),
        marker=dict(size=10, color='#4C72B0', line=dict(color='#FFFFFF', width=2)),
        text=[f'{v:.3f}' for v in tl_df['fact_recall']],
        textposition='top center', textfont=dict(color='#000000', size=13),
    ))
    fig_aq.add_trace(go.Scatter(
        x=tl_df['label'], y=tl_df['groundedness'],
        mode='lines+markers+text', name='Groundedness',
        line=dict(color='#55A868', width=2.5, dash='dot'),
        marker=dict(size=10, color='#55A868', line=dict(color='#FFFFFF', width=2)),
        text=[f'{v:.3f}' for v in tl_df['groundedness']],
        textposition='bottom center', textfont=dict(color='#000000', size=13),
    ))
    for row in _AQ_TIMELINE:
        if row['milestone']:
            fig_aq.add_shape(
                type='line', x0=row['label'], x1=row['label'], y0=0.55, y1=1.0,
                line=dict(color='#C44E52', width=1.5, dash='dash'),
            )
    fig_aq.update_layout(**_lay(
        height=380,
        xaxis=dict(title='', fixedrange=True),
        yaxis=dict(title='Score', range=[0.55, 1.0], fixedrange=True),
        dragmode=False,
        showlegend=True,
    ))
    st.plotly_chart(fig_aq, use_container_width=True, config=PLOTLY_CONFIG)

    # ── Config comparison ────────────────────────────────────────────────────
    sl("REAL PIPELINE VS. RAW RETRIEVAL CONFIGS")
    so("The production pipeline (router + resolved-query + corrective RAG) beats every raw retrieval ablation on fact recall, and matches the best config's 0% hallucination rate — the routing/resolution overhead isn't just latency, it measurably improves correctness.")

    cfg_df = pd.DataFrame(_AQ_CONFIGS)
    fig_cfg = go.Figure()
    fig_cfg.add_trace(go.Bar(
        name='Fact Recall', x=cfg_df['config'], y=cfg_df['fact_recall'],
        marker_color='#4C72B0', opacity=0.88,
        text=[f'{v:.3f}' for v in cfg_df['fact_recall']],
        textposition='outside', textfont=dict(color='#000000', size=12),
    ))
    fig_cfg.add_trace(go.Bar(
        name='Hallucination Rate', x=cfg_df['config'], y=cfg_df['hallucination_rate'],
        marker_color='#C44E52', opacity=0.88,
        text=[f'{v:.1%}' for v in cfg_df['hallucination_rate']],
        textposition='outside', textfont=dict(color='#000000', size=12),
    ))
    fig_cfg.update_layout(**_lay(
        height=420, barmode='group',
        xaxis=dict(tickangle=-10),
        yaxis=dict(title='Score', range=[0, 0.85]),
    ))
    st.plotly_chart(fig_cfg, use_container_width=True, config={'staticPlot': True})

    # ── Manual vs RAGAS divergence ───────────────────────────────────────────
    sl("MANUAL JUDGE VS. RAGAS — WHERE THE TWO DISAGREE")
    so("Points on the diagonal would mean the two methods agree. Most don't — q13 and q21 are near-total inversions. Treated as an open methodology gap (an eval reference-ID review flagged but not yet done), not resolved by picking whichever number looks better.")

    rv_df = pd.DataFrame(_AQ_MANUAL_VS_RAGAS)
    fig_rv = go.Figure()
    fig_rv.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode='lines',
        line=dict(color='#999999', width=1.5, dash='dash'),
        name='Perfect agreement', showlegend=True, hoverinfo='skip',
    ))
    fig_rv.add_trace(go.Scatter(
        x=rv_df['manual'], y=rv_df['ragas'], mode='markers+text',
        text=rv_df['id'], textposition='top center',
        textfont=dict(color='#000000', size=10),
        marker=dict(size=12, color='#4C72B0', opacity=0.85, line=dict(color='#FFFFFF', width=1)),
        name='Query', showlegend=False,
        customdata=rv_df['query'],
        hovertemplate='<b>%{text}</b>: %{customdata}<br>Manual: %{x:.2f}  RAGAS: %{y:.2f}<extra></extra>',
    ))
    fig_rv.update_layout(**_lay(
        height=420,
        xaxis=dict(title='Manual Groundedness', range=[-0.05, 1.05]),
        yaxis=dict(title='RAGAS Faithfulness', range=[-0.05, 1.05]),
        showlegend=True,
    ))
    st.plotly_chart(fig_rv, use_container_width=True, config=PLOTLY_CONFIG)


# ══════════════════════════════════════════════════════════════════════════
# TAB: Behavior Demo  (every response below is a REAL cached answer from
# an actual graph.invoke() run against the live agent system, captured
# once and replayed — not scripted text, not a live call. See
# tools/resilient_batch.py's docstring for why demos in this project don't
# depend on a network call succeeding live.)
# ══════════════════════════════════════════════════════════════════════════

_RD_SCENARIOS = [
    {
        'id': 'modbus_before_after', 'title': 'Wrong Answer -> Fixed',
        'sub': 'Watch the same question fail, then get answered, after one real fix.',
        'takeaway': "The score going from 0.69 to 0.83 tells you something got better. It doesn't show what — this does: a flat \"I don't know\" turning into a real, sourced answer, from one targeted fix.",
    },
    {
        'id': 'web_search_fallback', 'title': "Honest About What It Knows",
        'sub': "The corpus barely mentions this topic — it says so, then still helps.",
        'takeaway': "The system notices the corpus barely mentions this topic, says so honestly, and still answers using general knowledge instead of just giving up. It only cites the corpus when it actually relied on it.",
    },
    {
        'id': 'multi_intent', 'title': 'One Question, Two Experts',
        'sub': 'A question with two parts in it gets both parts answered.',
        'takeaway': "This question really has two separate questions inside it. Instead of answering just one and dropping the other, the system hands each half to the right specialist at the same time and combines their answers.",
    },
    {
        'id': 'memory_followup', 'title': 'Remembers the Conversation',
        'sub': 'A follow-up question that only makes sense with context.',
        'takeaway': "\"Does that also cover conduit zones?\" means nothing on its own. The system remembers what was just discussed and figures out what \"that\" refers to before it goes looking for an answer — the same way a person would.",
    },
]


@st.cache_data(ttl=300)
def load_reliability_scenarios() -> dict:
    scenarios = {}
    for fname in ('reliability_demo_modbus.json', 'reliability_demo_fresh.json'):
        loaded = load_json(fname)
        if not loaded:
            continue
        if 'id' in loaded:
            scenarios[loaded['id']] = loaded
        else:
            scenarios.update(loaded)
    return scenarios


def _rd_render_turn(turn: dict, animate: bool):
    if turn['role'] == 'user':
        st.markdown(
            f'<div class="rd-turn user"><div class="rd-bubble-user">{turn["content"]}</div></div>',
            unsafe_allow_html=True,
        )
        return

    personas = turn.get('personas') or []
    badges = ''.join(f'<span class="rd-persona-badge">{p}</span>' for p in personas)
    latency = turn.get('latency_s')
    latency_html = f'<span class="rd-latency">{latency:.1f}s</span>' if latency else ''
    st.markdown(f'<div class="rd-assistant-meta">{badges}{latency_html}</div>', unsafe_allow_html=True)

    full_text = turn['content']
    if animate:
        placeholder = st.empty()
        words = full_text.split(' ')
        shown = ''
        for i in range(0, len(words), 3):
            shown += (' ' if shown else '') + ' '.join(words[i:i + 3])
            placeholder.markdown(shown + ' ▌')
            time.sleep(0.02)
        placeholder.markdown(shown)
    else:
        st.markdown(full_text)

    sources = turn.get('sources') or []
    if sources:
        chips = ''.join(f'<span class="rd-source-chip">{s}</span>' for s in sources)
        st.markdown(chips, unsafe_allow_html=True)


def _rd_render_conversation(turns: list, captured_at: str, key: str):
    st.markdown(
        f'<div class="rd-cached-note"><span class="rd-badge">CACHED</span> {captured_at}</div>',
        unsafe_allow_html=True,
    )
    # Animate only the first time this exact conversation is shown in this
    # session — flipping tabs or re-selecting the same card afterward just
    # renders it instantly instead of replaying the reveal every time.
    seen_key = f'rd_seen_{key}'
    animate = seen_key not in st.session_state
    st.session_state[seen_key] = True

    with st.container(height=360, border=True):
        for turn in turns:
            _rd_render_turn(turn, animate)


def tab_reliability_demo(data):
    tab_explainer("Behavior Demo")

    scenarios = load_reliability_scenarios()

    if 'rd_selected' not in st.session_state:
        st.session_state.rd_selected = next(
            (s['id'] for s in _RD_SCENARIOS if s['id'] in scenarios), None
        )
    if 'rd_variant' not in st.session_state:
        st.session_state.rd_variant = 'before'

    # Pick a card -> see the chat -> see what it means — laid out
    # left-to-right so nothing has to be scrolled past to reach the next
    # step, instead of stacking picker / chat / explanation vertically.
    col_pick, col_chat, col_mean = st.columns([1.0, 1.7, 1.1], gap="medium")

    with col_pick:
        sl("SCENARIOS")
        for s in _RD_SCENARIOS:
            available = s['id'] in scenarios
            is_selected = st.session_state.rd_selected == s['id']
            if st.button(
                s['title'], key=f"rd_card_{s['id']}", disabled=not available,
                use_container_width=True, type='primary' if is_selected else 'secondary',
            ):
                st.session_state.rd_selected = s['id']
                st.session_state.rd_variant = 'before'
                # Without this, the button's own primary/secondary styling
                # for THIS click is computed from is_selected above — set
                # BEFORE this click's state update — so it draws with the
                # old (unselected) style on the click that actually
                # selected it, and only shows highlighted after whatever
                # click happens next. Forcing an immediate rerun makes the
                # highlight land on the same click that caused it.
                st.rerun()
            st.caption(s['sub'] if available else f"{s['sub']} (not generated)")

    sel = st.session_state.rd_selected
    scenario = scenarios.get(sel)
    meta = next((s for s in _RD_SCENARIOS if s['id'] == sel), None)

    # WHAT THIS MEANS renders BEFORE the chat on purpose — Streamlit executes
    # top-to-bottom regardless of visual column order, and the chat's
    # streaming reveal below is full of real time.sleep() calls. Rendering
    # it second meant this column sat blank until the whole reveal
    # finished, even though it's laid out beside it, not after it.
    with col_mean:
        sl("WHAT THIS MEANS")
        if meta:
            st.markdown(
                f'<div style="font-size:0.86rem; line-height:1.55; color:{TXT_DARK};">{meta["takeaway"]}</div>',
                unsafe_allow_html=True,
            )

    with col_chat:
        if scenario is None:
            st.info("Pick a scenario.")
        else:
            # meta['title'], not scenario['title'] — the latter is baked
            # into the cached JSON at generation time and has no reason to
            # stay in sync with _RD_SCENARIOS when a title gets renamed
            # (confirmed real: it didn't, and the chat header showed a
            # stale name after a rename here while the card was correct).
            sl((meta['title'] if meta else scenario['title']).upper())
            if scenario['kind'] == 'before_after':
                variant = st.radio(
                    'variant', ['before', 'after'],
                    index=1 if st.session_state.rd_variant == 'after' else 0,
                    format_func=lambda v: 'Before the fix' if v == 'before' else 'After the fix',
                    horizontal=True, label_visibility='collapsed', key=f"rd_radio_{sel}",
                )
                st.session_state.rd_variant = variant
                captured_at = scenario['captured_at_after'] if variant == 'after' else scenario['captured_at_before']
                _rd_render_conversation(scenario[variant], captured_at, key=f"{sel}_{variant}")
            else:
                _rd_render_conversation(scenario['turns'], scenario['captured_at'], key=sel)


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    st.markdown(f"""
    <div class="dash-inner">
      <div class="dash-hero-wrap" style="text-align:center; padding:26px 40px 20px 40px; background:{INNER_BG}; margin:0;">
        <h1 class="dash-hero-title" style="color:{TXT_DARK}; margin:0 0 9px 0; font-size:2.3rem; font-weight:800; letter-spacing:-0.03em;">
          Private RAG Evaluation Dashboard
        </h1>
        <p class="dash-hero-text" style="color:{TXT_DARK}; margin:0 0 5px 0; font-size:1rem; line-height:1.45; white-space:nowrap;">
          An end-to-end view of how well a private document intelligence pipeline ingests, prepares, retrieves, and validates evidence from internal knowledge.
        </p>
        <p class="dash-hero-text dash-hero-note" style="color:{TXT2}; margin:0; font-size:0.86rem; line-height:1.4; white-space:nowrap;">
          Manufacturing is used as the demo domain because it reflects a common enterprise problem: legacy PDFs, manuals, SOPs, and engineering documents that often need to stay inside private or controlled environments.
        </p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    data = load_all()
    exported_at = data.get('pipeline', {}).get('exported_at', '')

    tabs = st.tabs([
        "Pipeline Overview",
        "Dataset Quality",
        "Retrieval Quality",
        "Answer Quality",
        "Behavior Demo",
    ])

    with tabs[0]: tab_overview(data)
    with tabs[1]:
        tab_corpus_map(data)
        st.divider()
        tab_sources(data)
    with tabs[2]: tab_retrieval_quality(data)
    with tabs[3]: tab_answer_quality(data)
    with tabs[4]: tab_reliability_demo(data)

    if exported_at:
        st.caption(f"Data snapshot: {exported_at[:19]}")


if __name__ == '__main__':
    main()
