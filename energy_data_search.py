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


# Confirmed against the actual data (not assumed): every row is exactly 1
# minute apart. Needed to convert instantaneous power (W) into real energy
# (kWh) via proper time-integration — summing/averaging raw watts across
# rows isn't a meaningful "total energy used" figure on its own, and asking
# the LLM to eyeball that from a handful of sample rows is exactly the
# failure mode this function exists to replace.
_MINUTES_PER_ROW = 1


def compute_energy_summary(site: str = None) -> str:
    """Real, computed power/energy statistics per site — not an LLM
    impression from sample rows. Covers the full available time range,
    since this is a static snapshot, not a live feed — there's no
    meaningful "right now" to report on otherwise."""
    df = _load()
    sites = [site] if site else sorted(df["site"].unique())

    lines = []
    for s in sites:
        site_df = df[df["site"] == s]
        if site_df.empty:
            continue
        lines.append(
            f"{s} ({len(site_df)} minutes, "
            f"{site_df['timestamp'].min()} to {site_df['timestamp'].max()}):"
        )
        for col, label in [
            ("production_power_w", "Production"),
            ("consumption_power_w", "Consumption"),
            ("grid_power_w", "Grid import(-)/export(+)"),
        ]:
            avg_w = site_df[col].mean()
            total_kwh = site_df[col].sum() * _MINUTES_PER_ROW / 60 / 1000
            lines.append(f"  {label}: avg {avg_w:.0f} W, total {total_kwh:.1f} kWh over the period")
        lines.append(f"  Average battery state of charge: {site_df['battery_soc_pct'].mean():.1f}%")

    return "\n".join(lines)


def compute_anomaly_trends(site: str = None) -> str:
    """Real, computed anomaly-rate trend per site — splits each site's data
    in half by timestamp and compares anomaly rates, giving an actual
    direction (increasing/decreasing/stable) rather than an LLM impression.
    Also surfaces each site's single most common anomaly type with a real
    count, not a guess."""
    df = _load().sort_values("timestamp")
    sites = [site] if site else sorted(df["site"].unique())

    lines = []
    for s in sites:
        site_df = df[df["site"] == s]
        if site_df.empty:
            continue
        midpoint = len(site_df) // 2
        first_rate = site_df.iloc[:midpoint]["is_anomaly"].mean()
        second_rate = site_df.iloc[midpoint:]["is_anomaly"].mean()
        if second_rate > first_rate * 1.1:
            direction = "increasing"
        elif second_rate < first_rate * 0.9:
            direction = "decreasing"
        else:
            direction = "stable"
        lines.append(
            f"{s}: anomaly rate {first_rate:.1%} (first half of period) -> "
            f"{second_rate:.1%} (second half) — {direction}"
        )
        anomalies = site_df[site_df["is_anomaly"]]
        if not anomalies.empty:
            top_type = anomalies["anomaly_type"].value_counts().idxmax()
            top_count = int((anomalies["anomaly_type"] == top_type).sum())
            lines.append(f"  Most common anomaly type: {top_type} ({top_count} occurrences)")

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
