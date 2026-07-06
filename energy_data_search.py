"""
Query layer over the synthetic energy telemetry data for the Diagnosis Agent.

synthetic_data/energy_data.csv — 3 fictional sites (edge_id 0/1/2), one row
per site per synthetic minute, ground-truth anomaly labels (is_anomaly /
anomaly_type) injected at export time for scoring rather than guessing.
`timestamp` is the synthetic calendar time to reason about (starts
2026-01-01); `source_time` is just real capture provenance from the
Jetson/InfluxDB pull, not meaningful for analysis. This is a dev-time
snapshot — the production version reads InfluxDB directly on Jetson; CSV is
just easier for building/testing the agent.
"""

from pathlib import Path

import pandas as pd

_CSV_PATH = Path(__file__).resolve().parent / "synthetic_data" / "energy_data.csv"

_df = None


def _load():
    global _df
    if _df is None:
        _df = pd.read_csv(_CSV_PATH, parse_dates=["timestamp"])
    return _df


def get_energy_data_overview() -> str:
    """Deterministic, accurate summary of the dataset — real counts computed
    from the data, not something the model should guess or count itself."""
    df = _load()
    lines = [
        f"Total rows: {len(df)}",
        f"Fields: timestamp (synthetic calendar time, starts {df['timestamp'].min()}), "
        f"site, edge_id, battery_soc_pct, battery_power_w, grid_power_w, "
        f"production_power_w, consumption_power_w, is_anomaly, anomaly_type",
        "",
        "Per site:",
    ]
    for site in sorted(df["site"].unique()):
        site_df = df[df["site"] == site]
        n_anom = int(site_df["is_anomaly"].sum())
        lines.append(f"  {site}: {len(site_df)} rows, {n_anom} anomalous ({n_anom / len(site_df):.1%})")

    lines.append("")
    lines.append("Anomaly type breakdown (across all sites):")
    anomaly_counts = df[df["is_anomaly"]]["anomaly_type"].value_counts()
    for atype, count in anomaly_counts.items():
        lines.append(f"  {atype}: {count} rows")

    return "\n".join(lines)


def find_site_and_anomaly_mentions(query: str) -> tuple:
    """Which sites/anomaly types (if any) a piece of text names explicitly —
    shared by both the live query and, when the query itself names none,
    the conversation history fallback in diagnosis_agent.py."""
    df = _load()
    query_lower = query.lower()
    site_matches = [s for s in df["site"].unique() if s.split()[0].lower() in query_lower]
    anomaly_types = df["anomaly_type"].dropna().unique()
    anomaly_matches = [
        a for a in anomaly_types
        if a.lower() in query_lower or a.lower().replace("_", " ") in query_lower
    ]
    return site_matches, anomaly_matches


def filter_energy_data(site_matches: list, anomaly_matches: list, max_sample_rows: int = 5) -> str:
    """Real filtered rows for an already-resolved site/anomaly-type
    selection — the actual data lookup, separated from mention-detection so
    a history-derived filter can reuse this without re-parsing any text."""
    df = _load()
    filtered = df
    if site_matches:
        filtered = filtered[filtered["site"].isin(site_matches)]
    if anomaly_matches:
        filtered = filtered[filtered["anomaly_type"].isin(anomaly_matches)]

    anomalous = filtered[filtered["is_anomaly"]]
    parts = [f"Filtered to {len(filtered)} rows"]
    if site_matches:
        parts[0] += f" for site(s): {site_matches}"
    if anomaly_matches:
        parts[0] += f" for anomaly type(s): {list(anomaly_matches)}"
    parts.append(f"{len(anomalous)} of those rows are anomalous.")
    parts.append("\nSample rows:\n" + filtered.head(max_sample_rows).to_string(index=False))
    if not anomalous.empty:
        parts.append("\nSample anomalous rows:\n" + anomalous.head(max_sample_rows).to_string(index=False))

    return "\n".join(parts)


def search_energy_data(query: str, max_sample_rows: int = 5) -> str:
    """Keyword-based filter: site names or anomaly type names mentioned in
    the query narrow the data down to real matching rows (with a sample of
    both normal and anomalous rows) — this is small, structured, tabular
    data, so exact name matching is more reliable here than the
    keyword-overlap scoring used for the PLC text corpus."""
    site_matches, anomaly_matches = find_site_and_anomaly_mentions(query)
    if not site_matches and not anomaly_matches:
        return "No specific site or anomaly type named in this query."
    return filter_energy_data(site_matches, anomaly_matches, max_sample_rows)
