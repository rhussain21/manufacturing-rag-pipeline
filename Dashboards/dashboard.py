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
import requests
from collections import Counter

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Config ────────────────────────────────────────────────────────────────

LOCAL_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dashboard_data')
REMOTE_DATA_URL = os.getenv('DATA_URL', '')

st.set_page_config(
    page_title="Corpus Intelligence Dashboard",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="collapsed",
)

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
    /* collapse the gap Streamlit adds after the markdown block */
    div[data-testid="stMarkdownContainer"]:has(.dash-inner),
    div[data-testid="element-container"]:has(.dash-inner) {{
        margin-bottom: -4px !important;
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
        font-size: 0.88rem;
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
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        color: {TXT_DARK} !important;
        border-bottom: 2px solid {GRN_LT};
        padding-bottom: 2px;
        margin: 18px 0 2px 0;
    }}
    .section-so {{
        font-size: 0.75rem;
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
    }}
    .metric-card .metric-value {{
        font-size: 1.7rem; font-weight: 700; color: {TXT_DARK} !important;
    }}
    .metric-card .metric-label {{
        font-size: 0.72rem; font-weight: 600; color: {TXT2} !important;
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
        'signals':   load_json('signal_stats.json'),
        'temporal':  load_json('temporal_data.json'),
        'explorer':  load_json('signal_explorer.json'),
        'logs':      load_json('system_logs.json'),
        'discovery': load_json('discovery_stats.json'),
        'corpus':    load_json('corpus_quality.json'),
    }


# ── Helpers ───────────────────────────────────────────────────────────────

def sl(text):
    """Section label — bold uppercase heading."""
    st.markdown(f'<div class="section-label">{text}</div>', unsafe_allow_html=True)


def so(text):
    """Section subtitle — the 'so what' in one line."""
    st.markdown(f'<div class="section-so">{text}</div>', unsafe_allow_html=True)


def mc(label, value, color="g"):
    bg = GRN_PALE if color == "g" else ('#fce4ec' if color == "r" else '#fff3e0')
    st.markdown(f"""<div class="metric-card" style="background:{bg};">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>""", unsafe_allow_html=True)


def _lay(height=400, **kw):
    """Build a Plotly layout dict with black fonts and transparent background.
    Pass xaxis/yaxis/coloraxis inside kw — they will be merged, not duplicated."""
    _ax = dict(
        tickfont=dict(color='#000000'),
        title=dict(font=dict(color='#000000')),
        gridcolor=GRID,
        zerolinecolor=GRID,
    )
    _colorax = dict(
        colorbar=dict(
            tickfont=dict(color='#000000'),
            title=dict(font=dict(color='#000000')),
        )
    )
    base = dict(
        font=dict(family='Inter, system-ui, sans-serif', size=13, color='#000000'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=20, t=30, b=80),
        height=height,
        legend=dict(
            orientation='h',
            x=0.5, xanchor='center',
            y=-0.22, yanchor='top',
            font=dict(color='#000000', size=12),
            bgcolor='rgba(0,0,0,0)',
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
# TAB: Overview  (merged with Pipeline)
# ══════════════════════════════════════════════════════════════════════════

def tab_overview(data):
    ps = data.get('pipeline', {})
    t = ps.get('totals', {})

    # ── Top-level KPIs ────────────────────────────────────────────────────
    c = st.columns(6)
    with c[0]: mc("DOCUMENTS", t.get('content', 0))
    with c[1]: mc("APPROVED", t.get('approved', 0))
    with c[2]: mc("ACCEPTANCE RATE", f"{t.get('acceptance_rate', 0):.0f}%")
    with c[3]: mc("SIGNALS EXTRACTED", f"{t.get('total_signals', 0):,}")
    with c[4]: mc("AVG SIGNALS / DOC", f"{t.get('avg_signals_per_doc', 0):.1f}")
    with c[5]: mc("EXTRACTION SUCCESS", f"{t.get('extraction_success_rate', 0):.0f}%")

    # ── Funnel + Corpus composition ───────────────────────────────────────
    cl, cr = st.columns(2)
    with cl:
        sl("ACCEPTANCE FUNNEL")
        so("How many discovered documents survive each pipeline stage — drop-offs reveal where quality gates are working.")
        funnel = ps.get('funnel', [])
        if funnel:
            fig = go.Figure(go.Funnel(
                y=[s['stage'] for s in funnel],
                x=[s['count'] for s in funnel],
                textinfo='value+percent initial',
                textfont=dict(color='#000000'),
                marker=dict(color=['#4e79a7', '#59a14f', '#76b7b2', '#f28e2b', '#b07aa1']),
                connector=dict(line=dict(width=1, color=BDR)),
            ))
            fig.update_layout(**_lay(height=340))
            st.plotly_chart(fig, use_container_width=True)

    with cr:
        sl("CORPUS COMPOSITION")
        so("Where approved content is coming from — a balanced mix reduces over-reliance on any single source type.")
        by_type = ps.get('by_source_type', [])
        if by_type:
            agg = pd.DataFrame(by_type).groupby('source_type')['count'].sum().reset_index()
            fig = px.pie(agg, values='count', names='source_type',
                         color_discrete_sequence=MULTI, hole=0.45)
            fig.update_layout(**_lay(height=340, showlegend=True,
                                     legend=dict(orientation='h', y=-0.15, font=dict(size=11))))
            fig.update_traces(textinfo='percent+label', textposition='outside', textfont_size=11)
            st.plotly_chart(fig, use_container_width=True)

    # ── Source type detail table (under Acceptance Funnel) ───────────────
    sl("SOURCE TYPE BREAKDOWN")
    so("Volume and quality split by content format and origin — identifies which content types yield the most signals.")
    by_type = ps.get('by_source_type', [])
    if by_type:
        btdf = pd.DataFrame(by_type)
        display = btdf.rename(columns={
            'source_type': 'Source', 'content_type': 'Format',
            'count': 'Total', 'approved': 'Approved',
            'rejected': 'Rejected', 'signals_extracted': 'Signals Done',
        })
        st.dataframe(display, use_container_width=True, hide_index=True)

    # ── Pipeline status row ───────────────────────────────────────────────
    sl("PIPELINE STATUS")
    so("Current processing state across the full document set — shows how far each document has progressed.")
    c2 = st.columns(6)
    with c2[0]: mc("EXTRACTED", t.get('extracted', 0))
    with c2[1]: mc("VECTORIZED", t.get('vectorized', 0))
    with c2[2]: mc("SIGNALS DONE", t.get('signals_done', 0))
    with c2[3]: mc("PENDING", t.get('pending', 0))
    with c2[4]: mc("FAILED EXTRACTION", t.get('failed_extraction', 0), "r")
    with c2[5]: mc("REJECTED", t.get('rejected', 0), "r")

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
            st.plotly_chart(fig, use_container_width=True)
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
                                     xaxis=dict(title='Error Count')))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("No errors in the last 7 days.")

    # ── Raw rejection reasons table (full width) ──────────────────────────
    if reasons:
        sl("RAW REJECTION REASONS")
        so("All individual rejection reason strings from the screener — useful for diagnosing gate calibration issues.")
        raw_df = pd.DataFrame(reasons).rename(
            columns={'screening_reason': 'Reason', 'count': 'Count'})
        st.dataframe(raw_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════
# TAB: Sources  (improved legibility and usefulness)
# ══════════════════════════════════════════════════════════════════════════

def tab_sources(data):
    sd = data.get('signals', {})
    cq = data.get('corpus', {})

    sy = sd.get('source_yield', [])
    if sy:
        sdf = pd.DataFrame(sy)
        sdf['signals_per_doc'] = (sdf['signal_count'] / sdf['doc_count']).round(1)
        sdf['avg_confidence'] = sdf['avg_confidence'].round(2)

        c = st.columns(3)
        with c[0]: mc("HIGHEST YIELD", sdf.iloc[0]['source_name'] if len(sdf) else "N/A")
        with c[1]:
            bq = sdf.loc[sdf['avg_confidence'].idxmax()] if len(sdf) else {}
            mc("HIGHEST QUALITY", bq.get('source_name', 'N/A'))
        with c[2]:
            bd = sdf.loc[sdf['signals_per_doc'].idxmax()] if len(sdf) else {}
            mc("HIGHEST SIGNAL DENSITY", bd.get('source_name', 'N/A'))

        sl("SIGNAL YIELD BY SOURCE")
        so("Which sources produce the most actionable signals — high bars with high confidence dots are your most valuable data assets.")
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(
            x=sdf['source_name'], y=sdf['signal_count'],
            name='Total Signals', marker_color='#4e79a7', opacity=0.9,
            text=sdf['signal_count'], textposition='outside',
            textfont=dict(color='#000000', size=11),
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=sdf['source_name'], y=sdf['avg_confidence'],
            name='Avg Confidence', mode='markers+lines',
            marker=dict(color='#e15759', size=10, symbol='diamond'),
            line=dict(color='#e15759', width=2, dash='dot'),
        ), secondary_y=True)
        fig.update_layout(**_lay(height=460,
                                  legend=dict(font=dict(size=12))))
        fig.update_xaxes(tickangle=-35, tickfont=dict(size=12))
        fig.update_yaxes(title_text="Signal Count", tickfont=dict(color='#000000', size=12),
                         title_font=dict(color='#000000'), secondary_y=False)
        fig.update_yaxes(title_text="Avg Confidence", range=[0, 1],
                         tickfont=dict(color='#000000', size=12),
                         title_font=dict(color='#000000'), secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            sdf[['source_name', 'source_type', 'doc_count', 'signal_count',
                 'signals_per_doc', 'avg_confidence']].rename(columns={
                'source_name': 'Source', 'source_type': 'Type', 'doc_count': 'Docs',
                'signal_count': 'Signals', 'signals_per_doc': 'Signals/Doc',
                'avg_confidence': 'Avg Conf.',
            }),
            use_container_width=True, hide_index=True,
        )

    # Top sources — approved vs rejected stacked + acceptance rate
    sl("TOP SOURCES — VOLUME & ACCEPTANCE")
    so("How selective the pipeline is about each source — a low acceptance rate may mean the source is noisy or off-domain.")
    ts = cq.get('top_sources', [])
    if ts:
        tsdf = pd.DataFrame(ts).head(20)
        tsdf['accept_pct'] = (tsdf['approved'] / tsdf['count'] * 100).round(1)
        tsdf['rejected_count'] = tsdf['count'] - tsdf['approved']
        tsdf = tsdf.sort_values('count', ascending=True)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=tsdf['source_name'], x=tsdf['approved'],
            name='Approved', orientation='h',
            marker_color='#59a14f', opacity=0.9,
            text=tsdf['approved'], textposition='inside',
            textfont=dict(color='#ffffff', size=11),
        ))
        fig.add_trace(go.Bar(
            y=tsdf['source_name'], x=tsdf['rejected_count'],
            name='Rejected', orientation='h',
            marker_color='#e15759', opacity=0.9,
            text=[f"{p}% acc." for p in tsdf['accept_pct']],
            textposition='outside',
            textfont=dict(color='#000000', size=10),
        ))
        fig.update_layout(**_lay(
            height=max(480, len(tsdf) * 36),
            barmode='stack',
            legend=dict(),
            xaxis=dict(title='Documents', tickfont=dict(color='#000000', size=12)),
            yaxis=dict(title='', tickfont=dict(color='#000000', size=11)),
        ))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Green = approved docs, Red = rejected. Accept % shown after each bar.")

        sl("SOURCE ACCEPTANCE RATE RANKING")
        so("Sources sorted by selectivity — high-volume, high-acceptance sources are your most reliable data feeds.")
        rank_df = tsdf[['source_name', 'source_type', 'count', 'approved',
                         'rejected_count', 'accept_pct']].sort_values(
            'accept_pct', ascending=False).rename(columns={
            'source_name': 'Source', 'source_type': 'Type', 'count': 'Total',
            'approved': 'Approved', 'rejected_count': 'Rejected',
            'accept_pct': 'Accept %',
        })
        st.dataframe(rank_df, use_container_width=True, hide_index=True)
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
            fig.update_layout(**_lay(height=310, xaxis=dict(title=''), yaxis=dict(title='Documents')))
            st.plotly_chart(fig, use_container_width=True)

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
            st.plotly_chart(fig, use_container_width=True)

    cl2, cr2 = st.columns(2)
    with cl2:
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
            fig.update_layout(**_lay(barmode='stack', height=270,
                                     legend=dict(),
                                     xaxis=dict(title=''), yaxis=dict(title='Count')))
            st.plotly_chart(fig, use_container_width=True)

    with cr2:
        sl("SIGNAL TYPES OVER TIME")
        so("Which signal categories are growing — reflects what the domain is actually publishing about.")
        st_data = tmp.get('signal_timeline', [])
        if st_data:
            stdf = pd.DataFrame(st_data)
            stdf['date'] = pd.to_datetime(stdf['date'])
            fig = px.area(stdf, x='date', y='count', color='signal_type',
                          color_discrete_sequence=MULTI)
            fig.update_layout(**_lay(height=270, xaxis=dict(title=''),
                                     yaxis=dict(title='Signals')))
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# TAB: View Data  (was Explorer)
# ══════════════════════════════════════════════════════════════════════════

def tab_view_data(data):
    sl("SIGNAL EXPLORER")
    so("Browse and filter the raw signal extraction output — use this to spot patterns, gaps, or suspect extractions.")
    exp = data.get('explorer', {}).get('signals', [])
    if exp:
        edf = pd.DataFrame(exp)

        all_types = sorted(edf['signal_type'].dropna().unique().tolist())
        all_inds  = sorted(edf['industry'].dropna().unique().tolist())

        c1, c2, c3 = st.columns(3)
        with c1:
            sel_type = st.multiselect(
                "Signal Type", all_types, default=all_types,
                help="Deselect to filter. All selected = show all.",
            )
        with c2:
            sel_ind = st.multiselect(
                "Industry", all_inds, default=all_inds,
                help="Deselect to filter. All selected = show all.",
            )
        with c3:
            min_conf = st.slider("Min Confidence", 0.0, 1.0, 0.0, 0.05)

        filtered = edf.copy()
        # treat empty selection same as "all selected"
        if sel_type and set(sel_type) != set(all_types):
            filtered = filtered[filtered['signal_type'].isin(sel_type)]
        if sel_ind and set(sel_ind) != set(all_inds):
            filtered = filtered[filtered['industry'].isin(sel_ind)]
        filtered = filtered[filtered['confidence'] >= min_conf]

        st.caption(f"Showing {len(filtered):,} of {len(edf):,} signals")

        cols = ['signal_type', 'entity', 'description', 'industry', 'impact_level',
                'confidence', 'source_title', 'source_type']
        display_cols = [c for c in cols if c in filtered.columns]
        # fixed height so the container doesn't resize with the table
        st.dataframe(
            filtered[display_cols].head(300),
            use_container_width=True,
            hide_index=True,
            height=520,
        )
    else:
        st.caption("No signal data available.")


# ══════════════════════════════════════════════════════════════════════════
# TAB: Signal Analysis  (Intelligence + 3D Analytics merged)
# ══════════════════════════════════════════════════════════════════════════

def tab_signal_analysis(data):
    sd = data.get('signals', {})
    exp = data.get('explorer', {}).get('signals', [])

    # ── Extraction Quality EDA ────────────────────────────────────────────
    sl("EXTRACTION QUALITY OVERVIEW")
    so("Diagnostic metrics on whether signal extraction is producing meaningful, specific output — or just generic noise.")

    if exp:
        sdf = pd.DataFrame(exp)
        total_signals = len(sdf)
        unique_entities = sdf['entity'].nunique() if 'entity' in sdf.columns else 0
        entity_ratio = round(unique_entities / total_signals * 100, 1) if total_signals else 0

        short_ents = 0
        multi_type_count = 0
        multi_type_ents = pd.DataFrame()
        top5_pct = 0
        dominant_type_pct = 0
        dominant_type_name = 'N/A'

        if 'entity' in sdf.columns:
            short_ents = sdf[sdf['entity'].str.len() <= 3]['entity'].nunique()
        if 'entity' in sdf.columns and 'signal_type' in sdf.columns:
            ent_types = sdf.groupby('entity')['signal_type'].nunique().reset_index()
            ent_types.columns = ['entity', 'type_count']
            multi_type_ents = ent_types[ent_types['type_count'] > 1]
            multi_type_count = len(multi_type_ents)
        if 'entity' in sdf.columns:
            top5_counts = sdf['entity'].value_counts().head(5).sum()
            top5_pct = round(top5_counts / total_signals * 100, 1) if total_signals else 0
        if 'signal_type' in sdf.columns:
            dominant_type_pct = round(
                sdf['signal_type'].value_counts().iloc[0] / total_signals * 100, 1
            ) if total_signals else 0
            dominant_type_name = sdf['signal_type'].value_counts().index[0] if total_signals else 'N/A'

        c = st.columns(6)
        with c[0]: mc("UNIQUE ENTITIES", f"{unique_entities}")
        with c[1]: mc("ENTITY RATIO", f"{entity_ratio}%")
        with c[2]: mc("MULTI-TYPE ENTITIES", f"{multi_type_count}")
        with c[3]: mc("SHORT ENTITIES (≤3)", f"{short_ents}", "r" if short_ents > 5 else "g")
        with c[4]: mc("TOP-5 DOMINANCE", f"{top5_pct}%", "r" if top5_pct > 40 else "g")
        with c[5]: mc("DOMINANT TYPE %", f"{dominant_type_pct}%")

        st.caption(
            f"Entity Ratio = unique entities ÷ total signals. "
            f"Low ratio = heavy repetition. "
            f"Dominant type: {dominant_type_name} ({dominant_type_pct}%)."
        )

        cl_eda, cr_eda = st.columns(2)
        with cl_eda:
            sl("ENTITIES SPANNING MULTIPLE SIGNAL TYPES")
            so("Entities appearing across multiple categories are either genuinely cross-domain actors or signs of over-extraction.")
            if len(multi_type_ents) > 0:
                ent_detail = sdf.groupby(['entity', 'signal_type']).size().reset_index(name='count')
                multi_detail = ent_detail[ent_detail['entity'].isin(multi_type_ents['entity'])]
                pivot_multi = multi_detail.pivot_table(
                    index='entity', columns='signal_type', values='count', fill_value=0)
                pivot_multi['total'] = pivot_multi.sum(axis=1)
                pivot_multi = pivot_multi.sort_values('total', ascending=False).head(20)
                display_pivot = pivot_multi.drop('total', axis=1)
                fig = px.imshow(display_pivot, color_continuous_scale=VIBRANT_SCALE, aspect='auto')
                fig.update_layout(**_lay(height=min(400, max(200, len(display_pivot) * 25))))
                st.plotly_chart(fig, use_container_width=True)
                st.caption(f"{multi_type_count} entities appear under 2+ signal types.")
            else:
                st.caption("No entities span multiple signal types.")

        with cr_eda:
            sl("ENTITY LENGTH DISTRIBUTION")
            so("Short entity names (1–3 chars) are usually too generic to be useful — most entities should be 4+ characters.")
            if 'entity' in sdf.columns:
                ent_lens = sdf.drop_duplicates('entity')['entity'].str.len()
                len_bins = pd.cut(ent_lens, bins=[0, 3, 8, 15, 30, 200],
                                  labels=['1-3 (too short?)', '4-8 (single word)',
                                          '9-15 (typical)', '16-30 (detailed)', '30+ (very long)'])
                len_dist = len_bins.value_counts().reset_index()
                len_dist.columns = ['length_range', 'count']
                fig = px.bar(len_dist, x='length_range', y='count',
                             color_discrete_sequence=['#4e79a7'])
                fig.update_layout(**_lay(height=270, xaxis=dict(title='Entity Name Length'),
                                         yaxis=dict(title='Unique Entities')))
                st.plotly_chart(fig, use_container_width=True)

        sl("MOST REPEATED ENTITIES")
        so("High mention counts from few unique sources may indicate over-extraction from a single document or press release.")
        if 'entity' in sdf.columns:
            repeat = sdf.groupby('entity').agg(
                mentions=('entity', 'size'),
                signal_types=('signal_type', lambda x: ', '.join(sorted(x.unique()))),
                avg_confidence=('confidence', 'mean'),
                unique_sources=('source_title', 'nunique'),
            ).reset_index().sort_values('mentions', ascending=False).head(30)
            repeat['avg_confidence'] = repeat['avg_confidence'].round(2)
            repeat.columns = ['Entity', 'Mentions', 'Signal Types', 'Avg Conf.', 'Unique Sources']
            st.dataframe(repeat, use_container_width=True, hide_index=True)

        sl("ENTITY COMPLEXITY BY SIGNAL TYPE")
        so("Multi-word entities carry more specific meaning — a high proportion of single-word entities signals a prompt tuning opportunity.")
        if 'entity' in sdf.columns and 'signal_type' in sdf.columns:
            sdf_ent = sdf.copy()
            sdf_ent['word_count'] = sdf_ent['entity'].str.split().str.len()
            sdf_ent['complexity'] = pd.cut(
                sdf_ent['word_count'], bins=[0, 1, 2, 100],
                labels=['Single word', '2 words', '3+ words'])
            complexity_by_type = sdf_ent.groupby(
                ['signal_type', 'complexity']).size().reset_index(name='count')
            fig = px.bar(complexity_by_type, x='signal_type', y='count', color='complexity',
                         barmode='group',
                         color_discrete_sequence=['#e15759', '#f28e2b', '#59a14f'])
            fig.update_layout(**_lay(height=300, xaxis=dict(title='Signal Type'),
                                     yaxis=dict(title='Count'),
                                     legend=dict()))
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("No signal explorer data available for EDA analysis.")

    st.divider()

    # ── Signal type metrics + distributions ───────────────────────────────
    st_types = sd.get('signal_types', [])
    if st_types:
        c = st.columns(min(len(st_types), 6))
        for i, s in enumerate(st_types[:6]):
            with c[i]:
                mc(s['signal_type'], f"{s['count']:,}")

    a, b = st.columns(2)
    with a:
        sl("CONFIDENCE DISTRIBUTION")
        so("How certain the model is about its own extractions — a left-skewed distribution suggests the extraction prompt needs tightening.")
        cd = sd.get('confidence_distribution', [])
        if cd:
            fig = px.bar(pd.DataFrame(cd), x='bucket', y='count',
                         color_discrete_sequence=['#4e79a7'])
            fig.update_layout(**_lay(height=250, xaxis=dict(title='Confidence Bucket'),
                                     yaxis=dict(title='Count')))
            st.plotly_chart(fig, use_container_width=True)

    with b:
        sl("IMPACT LEVEL")
        so("Proportion of signals tagged as high, medium, or low business impact — too many 'low' signals may indicate weak domain coverage.")
        imp = sd.get('impact_distribution', [])
        if imp:
            fig = px.pie(pd.DataFrame(imp), values='count', names='impact_level',
                         color_discrete_sequence=MULTI, hole=0.4)
            fig.update_layout(**_lay(height=330))
            st.plotly_chart(fig, use_container_width=True)

    cl, cr = st.columns(2)
    with cl:
        sl("TOP ENTITIES")
        so("The most frequently mentioned entities across the corpus — these are the key players and technologies in your domain.")
        te = sd.get('top_entities', [])
        if te:
            edf2 = pd.DataFrame(te[:25])
            fig = px.bar(edf2, x='count', y='entity', orientation='h',
                         color='signal_type', color_discrete_sequence=MULTI,
                         hover_data=['avg_confidence'])
            fig.update_layout(**_lay(height=max(340, min(len(edf2) * 20, 560)),
                                     yaxis=dict(autorange='reversed', title='')))
            st.plotly_chart(fig, use_container_width=True)

    with cr:
        sl("BY INDUSTRY")
        so("Signal volume and quality broken down by industrial vertical — reveals which sub-sectors your corpus covers most deeply.")
        ind = sd.get('industry_distribution', [])
        if ind:
            idf = pd.DataFrame(ind)
            fig = px.bar(idf.head(15), x='count', y='industry', orientation='h',
                         color='avg_confidence', color_continuous_scale=VIBRANT_SCALE)
            fig.update_layout(**_lay(height=max(270, min(len(idf.head(15)) * 30, 460)),
                                     yaxis=dict(autorange='reversed', title=''),
                                     coloraxis_colorbar=dict(title='Conf.')))
            st.plotly_chart(fig, use_container_width=True)

    te = sd.get('top_entities', [])
    if te and len(te) >= 5:
        sl("ENTITY × SIGNAL TYPE HEATMAP")
        so("Which entities appear across which signal categories — bright rows indicate cross-domain actors worth investigating.")
        edf3 = pd.DataFrame(te[:50])
        pivot = edf3.pivot_table(index='entity', columns='signal_type',
                                 values='count', fill_value=0)
        pivot['_t'] = pivot.sum(axis=1)
        pivot = pivot.sort_values('_t', ascending=False).drop('_t', axis=1).head(20)
        fig = px.imshow(pivot, color_continuous_scale=VIBRANT_SCALE, aspect='auto')
        fig.update_layout(**_lay(height=470))
        st.plotly_chart(fig, use_container_width=True)

    if te and len(te) >= 3:
        sl("ENTITY TREEMAP")
        so("Visual proportion of signal volume by type and entity — large tiles dominate the corpus and may warrant closer scrutiny.")
        edf4 = pd.DataFrame(te[:40])
        fig = px.treemap(edf4, path=['signal_type', 'entity'], values='count',
                         color='avg_confidence', color_continuous_scale=VIBRANT_SCALE)
        fig.update_layout(**_lay(height=470))
        st.plotly_chart(fig, use_container_width=True)

    # ── Entity Co-occurrence Network (3D) ─────────────────────────────────
    sl("ENTITY CO-OCCURRENCE NETWORK (3D)")
    so("Which entities appear together in the same documents — clusters reveal partnerships, shared tech stacks, and market relationships.")
    cooc = sd.get('entity_cooccurrence', [])
    if cooc and len(cooc) >= 3:
        ents = set()
        for edge in cooc[:60]:
            ents.add(edge['entity_a']); ents.add(edge['entity_b'])
        ents = list(ents)
        np.random.seed(42)
        coords = np.random.randn(len(ents), 3) * 2.5
        pos = {e: coords[i] for i, e in enumerate(ents)}

        ex, ey, ez = [], [], []
        for edge in cooc[:60]:
            a, b = edge['entity_a'], edge['entity_b']
            if a in pos and b in pos:
                p1, p2 = pos[a], pos[b]
                ex.extend([p1[0], p2[0], None])
                ey.extend([p1[1], p2[1], None])
                ez.extend([p1[2], p2[2], None])

        mention_cnt = Counter()
        te_raw = sd.get('top_entities', [])
        if te_raw:
            for row in te_raw:
                mention_cnt[row['entity']] += row['count']
        ecnt = Counter()
        for edge in cooc[:60]:
            ecnt[edge['entity_a']] += edge['count']
            ecnt[edge['entity_b']] += edge['count']

        max_mentions = max(mention_cnt.values()) if mention_cnt else 1
        nsz = [max(5, min(35, (mention_cnt.get(e, ecnt.get(e, 1)) / max_mentions) * 35))
               for e in ents]
        nclr = [mention_cnt.get(e, ecnt.get(e, 1)) for e in ents]

        fig = go.Figure()
        fig.add_trace(go.Scatter3d(x=ex, y=ey, z=ez, mode='lines',
                                   line=dict(width=1, color='rgba(0,0,0,0.35)'),
                                   hoverinfo='none', showlegend=False))
        fig.add_trace(go.Scatter3d(
            x=[pos[e][0] for e in ents], y=[pos[e][1] for e in ents],
            z=[pos[e][2] for e in ents], mode='markers+text',
            marker=dict(size=nsz, color=nclr, colorscale=WARM_COOL, opacity=0.9,
                        line=dict(width=0.5, color='#333333'),
                        colorbar=dict(
                            title=dict(text='Mentions', font=dict(color='#000000')),
                            tickfont=dict(color='#000000'))),
            text=ents, textposition='top center', textfont=dict(size=9, color='#000000'),
            hovertext=[f"{e}: {mention_cnt.get(e, ecnt.get(e, 0))} mentions" for e in ents],
            hoverinfo='text', showlegend=False))
        fig.update_layout(**_lay(height=600,
            scene=dict(xaxis=dict(showgrid=False, zeroline=False, visible=False),
                       yaxis=dict(showgrid=False, zeroline=False, visible=False),
                       zaxis=dict(showgrid=False, zeroline=False, visible=False),
                       bgcolor='rgba(0,0,0,0)')))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Drag to rotate. Node size and color reflect total mention count.")
    else:
        st.caption("Not enough co-occurrence data for 3D network.")

    st.divider()

    # ── Source × Confidence × Volume (3D scatter) ─────────────────────────
    sl("SOURCE × CONFIDENCE × SIGNAL VOLUME (3D)")
    so("Which sources are both prolific and high-confidence — top-right-back corner is where your best data assets live.")
    sy = sd.get('source_yield', [])
    if sy:
        sdf2 = pd.DataFrame(sy)
        fig = go.Figure(go.Scatter3d(
            x=sdf2['doc_count'], y=sdf2['avg_confidence'], z=sdf2['signal_count'],
            mode='markers+text',
            marker=dict(size=8, color=sdf2['avg_confidence'],
                        colorscale=WARM_COOL, opacity=0.9,
                        line=dict(width=0.5, color='#333333'),
                        colorbar=dict(
                            title=dict(text='Confidence', font=dict(color='#000000')),
                            tickfont=dict(color='#000000'))),
            text=sdf2['source_name'], textposition='top center',
            textfont=dict(size=8, color='#000000'),
            hovertext=[f"{r['source_name']}: {r['signal_count']} signals, {r['doc_count']} docs"
                       for _, r in sdf2.iterrows()],
            hoverinfo='text',
        ))
        fig.update_layout(**_lay(height=600,
            scene=dict(
                xaxis=dict(title='Documents', showgrid=True, gridcolor='#ddd'),
                yaxis=dict(title='Avg Confidence', showgrid=True, gridcolor='#ddd'),
                zaxis=dict(title='Signal Count', showgrid=True, gridcolor='#ddd'),
                bgcolor='rgba(0,0,0,0)',
            )))
        st.plotly_chart(fig, use_container_width=True)

    sl("SIGNAL DENSITY (SIGNALS PER 1,000 WORDS)")
    so("Documents with high signal density pack more actionable content per word — low-density documents may be too verbose or off-topic.")
    density = sd.get('signal_density', [])
    if density:
        ddf = pd.DataFrame(density)
        if 'word_count' in ddf.columns and 'signal_density' in ddf.columns:
            fig = px.scatter(ddf, x='word_count', y='signal_density', color='source_type',
                             size='signal_count', hover_data=['title', 'source_name'],
                             color_discrete_sequence=MULTI)
            fig.update_layout(**_lay(height=370, xaxis=dict(title='Word Count'),
                                     yaxis=dict(title='Signals / 1,000 words')))
            st.plotly_chart(fig, use_container_width=True)

    sl("ENTITY CLUSTER (3D)")
    so("Entity distribution by signal volume and extraction confidence — isolated high-confidence nodes are your most reliably extracted facts.")
    te2 = sd.get('top_entities', [])
    if te2 and len(te2) >= 5:
        edf5 = pd.DataFrame(te2[:40])
        np.random.seed(99)
        n = len(edf5)
        edf5['x'] = np.random.randn(n) * 3
        edf5['y'] = np.random.randn(n) * 3
        edf5['z'] = np.random.randn(n) * 3

        fig = go.Figure(go.Scatter3d(
            x=edf5['x'], y=edf5['y'], z=edf5['z'],
            mode='markers+text',
            marker=dict(size=np.clip(edf5['count'] / edf5['count'].max() * 25, 5, 30),
                        color=edf5['avg_confidence'],
                        colorscale=WARM_COOL, opacity=0.85,
                        line=dict(width=0.5, color='#333333'),
                        colorbar=dict(
                            title=dict(text='Confidence', font=dict(color='#000000')),
                            tickfont=dict(color='#000000'))),
            text=edf5['entity'], textposition='top center',
            textfont=dict(size=8, color='#000000'),
            hovertext=[f"{r['entity']} ({r['signal_type']}): {r['count']} mentions"
                       for _, r in edf5.iterrows()],
            hoverinfo='text',
        ))
        fig.update_layout(**_lay(height=600,
            scene=dict(
                xaxis=dict(showgrid=False, zeroline=False, visible=False),
                yaxis=dict(showgrid=False, zeroline=False, visible=False),
                zaxis=dict(showgrid=False, zeroline=False, visible=False),
                bgcolor='rgba(0,0,0,0)',
            )))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Entity positions are randomized. Size = mention count, color = avg confidence.")


# ══════════════════════════════════════════════════════════════════════════
# TAB: System Logs  (Discovery + Logs merged)
# ══════════════════════════════════════════════════════════════════════════

def tab_system_logs(data):
    disc = data.get('discovery', {})
    logs_data = data.get('logs', {})

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
            st.plotly_chart(fig, use_container_width=True)

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
# TAB: Retrieval Quality  (v2 evaluation snapshot — hardcoded from Notebook 3)
# ══════════════════════════════════════════════════════════════════════════

# v2 final metrics (Notebook 3, 31 labeled queries, nomic-embed-text-v1.5, 78,467 chunks)
_RQ_CONFIGS = [
    {'config': 'Dense Only',        'mrr': 0.307, 'recall5': 0.257, 'recall10': 0.385},
    {'config': 'Hybrid (RRF)',      'mrr': 0.363, 'recall5': 0.253, 'recall10': 0.291},
    {'config': 'Hybrid + Reranker', 'mrr': 0.374, 'recall5': 0.220, 'recall10': 0.273},
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


def tab_retrieval_quality(_data):
    st.caption("v2 evaluation snapshot · 31 labeled queries · nomic-embed-text-v1.5 · 78,467 chunks @ max_chars=600")

    # ── KPI cards ─────────────────────────────────────────────────────────
    c = st.columns(5)
    with c[0]: mc("CORPUS DOCS", "736")
    with c[1]: mc("CHUNKS", "78,467")
    with c[2]: mc("LABELED QUERIES", "31")
    with c[3]: mc("BEST MRR", "0.374")
    with c[4]: mc("BEST RECALL@10", "0.385")

    # ── Config comparison bar chart ────────────────────────────────────────
    sl("RETRIEVAL CONFIGURATION COMPARISON")
    so("Dense retrieval wins on recall; Hybrid+Reranker wins on MRR. BM25 hybrid hurts recall on narrow-domain vocabulary.")

    cfg_df = pd.DataFrame(_RQ_CONFIGS)
    labels = cfg_df['config'].tolist()
    x = np.arange(len(labels))
    w = 0.25

    fig = go.Figure()
    for i, (col, metric, name) in enumerate([
        ('#4C72B0', 'mrr',      'MRR'),
        ('#55A868', 'recall5',  'Recall@5'),
        ('#C44E52', 'recall10', 'Recall@10'),
    ]):
        vals = cfg_df[metric].tolist()
        fig.add_trace(go.Bar(
            name=name,
            x=[f"{l}<br><sub>{name}</sub>" for l in labels],
            y=vals,
            marker_color=col, opacity=0.88,
            text=[f'{v:.3f}' for v in vals],
            textposition='outside',
            textfont=dict(color='#000000', size=10),
            width=0.22,
            offset=(i - 1) * 0.23,
        ))

    # Rebuild as grouped
    fig2 = go.Figure()
    colors = ['#4C72B0', '#55A868', '#C44E52']
    metrics_list = [('mrr', 'MRR'), ('recall5', 'Recall@5'), ('recall10', 'Recall@10')]
    for col, (metric, name) in zip(colors, metrics_list):
        vals = cfg_df[metric].tolist()
        fig2.add_trace(go.Bar(
            name=name, x=labels, y=vals,
            marker_color=col, opacity=0.88,
            text=[f'{v:.3f}' for v in vals],
            textposition='outside',
            textfont=dict(color='#000000', size=10),
        ))
    fig2.update_layout(**_lay(
        height=420, barmode='group',
        yaxis=dict(range=[0, 0.65], title='Score'),
        legend=dict(font=dict(size=12)),
    ))
    st.plotly_chart(fig2, use_container_width=True)

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
            textfont=dict(color='#000000', size=10),
        ))
        fig.update_layout(**_lay(
            height=320,
            xaxis=dict(title='k', tickvals=[1, 3, 5, 10, 20]),
            yaxis=dict(title='Mean Recall@k', range=[0, 0.65]),
        ))
        st.plotly_chart(fig, use_container_width=True)

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
            textfont=dict(color='#000000', size=10),
        ))
        fig.update_layout(**_lay(
            height=320,
            xaxis=dict(title='Mean Recall@10', range=[0, 1]),
            yaxis=dict(title=''),
        ))
        st.plotly_chart(fig, use_container_width=True)

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
            textposition='outside', textfont=dict(color='#000000', size=10),
        ))
        fig.add_trace(go.Bar(
            x=[str(r['max_chars']) for r in _ABLATION],
            y=[r['recall5'] for r in _ABLATION],
            name='Recall@5', marker_color='#55A868', opacity=0.85,
            text=[f"{r['recall5']:.3f}" for r in _ABLATION],
            textposition='outside', textfont=dict(color='#000000', size=10),
        ))
        fig.update_layout(**_lay(
            height=320, barmode='group',
            xaxis=dict(title='max_chars'),
            yaxis=dict(title='Score', range=[0, 1]),
            legend=dict(),
        ))
        fig.add_vline(x=1, line_dash='dash', line_color='#C44E52', line_width=2,
                      annotation_text='winner', annotation_font_color='#C44E52')
        st.plotly_chart(fig, use_container_width=True)

    with cr2:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[str(r['max_chars']) for r in _ABLATION],
            y=[r['n_chunks'] for r in _ABLATION],
            marker_color=TABLEAU[:4], opacity=0.85,
            text=[f"{r['n_chunks']:,}" for r in _ABLATION],
            textposition='outside', textfont=dict(color='#000000', size=10),
        ))
        fig.update_layout(**_lay(
            height=320,
            xaxis=dict(title='max_chars'),
            yaxis=dict(title='Total Chunks'),
        ))
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        abl_df.rename(columns={
            'max_chars': 'max_chars', 'n_chunks': 'Chunks',
            'mean_chars': 'Mean Chars', 'std_chars': 'Std Chars',
            'recall5': 'Recall@5', 'top_sim': 'Top Sim',
        }),
        use_container_width=True, hide_index=True,
    )


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    st.markdown(f"""
    <div class="dash-inner">
      <div style="text-align:center; padding:26px 52px 20px 52px; background:{INNER_BG}; margin:0;">
        <h1 style="color:{TXT_DARK}; margin:0 0 6px 0; font-size:1.65rem; font-weight:700; letter-spacing:-0.02em;">
          Corpus Intelligence Dashboard
        </h1>
        <p style="color:{TXT2}; margin:0; font-size:0.84rem; line-height:1.5;">
          System health and retrieval quality dashboard — tracks pipeline throughput, corpus composition, and how well the knowledge base supports RAG applications.
        </p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    data = load_all()
    exported_at = data.get('pipeline', {}).get('exported_at', '')

    tabs = st.tabs([
        "Overview",
        "Sources",
        "Signal Analysis",
        "Retrieval Quality",
        "View Data",
        "System Logs",
    ])

    with tabs[0]: tab_overview(data)
    with tabs[1]: tab_sources(data)
    with tabs[2]: tab_signal_analysis(data)
    with tabs[3]: tab_retrieval_quality(data)
    with tabs[4]: tab_view_data(data)
    with tabs[5]: tab_system_logs(data)

    if exported_at:
        st.caption(f"Data snapshot: {exported_at[:19]}")


if __name__ == '__main__':
    main()
