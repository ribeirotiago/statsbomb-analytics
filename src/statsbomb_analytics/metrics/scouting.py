"""Percentile profiles for the scouting pizza (metrics grouped in four dimensions).

Assumptions and limitations:
- Percentiles are rank-based (0-100) inside a pool chosen by the caller (a league or all five,
  above the minutes threshold). With `by_group=True` each player is ranked only against players
  of his own position group (GK/DEF/MID/FWD): volume metrics depend strongly on the role.
- A percentile says where a player sits, not by how much. Per-90 rates are noisy with few
  minutes, so the app defaults to a 300-minute minimum.
- Volume metrics reflect team style (possession teams accumulate more passes and xT).
- The position group comes from the position with most minutes, which blurs hybrid roles.
- Goalkeepers are ranked on the same metrics but event data barely describes their work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

CATEGORY_COLORS: dict[str, str] = {
    "Attack": "#E65100",
    "Creation": "#1565C0",
    "Defense": "#2E7D32",
    "Possession": "#6A1B9A",
}


@dataclass(frozen=True)
class RadarMetric:
    """One slice of the pizza: a per-90 (or percentage) column and how to show its total."""

    column: str
    label: str
    category: str
    total: str | None = None  # column with the season total (None for percentages)
    ratio: tuple[str, str] | None = None  # (numerator, denominator) columns for percentages
    unit: str = ""  # "%" for percentages


METRICS: tuple[RadarMetric, ...] = (
    RadarMetric("goals_p90", "Goals", "Attack", total="goals"),
    RadarMetric("npxg_p90", "npxG", "Attack", total="npxg"),
    RadarMetric("shots_p90", "Shots", "Attack", total="shots"),
    RadarMetric("vaep_p90", "VAEP", "Attack", total="vaep"),
    RadarMetric("key_passes_p90", "Key passes", "Creation", total="key_passes"),
    RadarMetric("assists_p90", "Assists", "Creation", total="assists"),
    RadarMetric("xt_p90", "Singh xT", "Creation", total="xt"),
    RadarMetric("axt_p90", "Action xT", "Creation", total="axt"),
    RadarMetric("xgchain_p90", "xGChain", "Creation", total="xgchain"),
    RadarMetric("tackles_p90", "Tackles", "Defense", total="tackles"),
    RadarMetric("interceptions_p90", "Interceptions", "Defense", total="interceptions"),
    RadarMetric("clearances_p90", "Clearances", "Defense", total="clearances"),
    RadarMetric("pressures_p90", "Pressures", "Defense", total="pressures"),
    RadarMetric("ball_recoveries_p90", "Recoveries", "Defense", total="ball_recoveries"),
    RadarMetric("passes_p90", "Passes", "Possession", total="passes"),
    RadarMetric("pass_pct", "Pass %", "Possession", ratio=("passes_completed", "passes"), unit="%"),
    RadarMetric("dribbles_completed_p90", "Dribbles", "Possession", total="dribbles_completed"),
    RadarMetric(
        "dribble_pct", "Dribble %", "Possession", ratio=("dribbles_completed", "dribbles"), unit="%"
    ),
    RadarMetric("xgbuildup_p90", "xGBuildup", "Possession", total="xgbuildup"),
)

PCTILE_SUFFIX = "_pctile"


def percentile_ranks(pool: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Percentile (0-100) of each row on each column inside `pool`; NaN stays NaN."""
    return pool[columns].rank(pct=True) * 100.0


def add_percentiles(pool: pd.DataFrame, by_group: bool = False) -> pd.DataFrame:
    """Copy of `pool` with a `<column>_pctile` (0-100) per radar metric.

    With `by_group=True` the rank is computed inside each `position_group`.
    """
    out = pool.copy()
    for metric in METRICS:
        if by_group:
            ranks = out.groupby("position_group")[metric.column].rank(pct=True)
        else:
            ranks = out[metric.column].rank(pct=True)
        out[f"{metric.column}{PCTILE_SUFFIX}"] = ranks * 100.0
    return out


def format_total(row: pd.Series, metric: RadarMetric) -> str:
    """Season total as text: `completed/attempts` for percentages, one decimal otherwise."""
    if metric.ratio:
        num, den = (row.get(c) for c in metric.ratio)
        if pd.isna(num) or pd.isna(den):
            return "—"
        return f"{int(num)}/{int(den)}"
    value = row.get(metric.total)
    if value is None or pd.isna(value):
        return "—"
    value = float(value)
    return f"{int(round(value))}" if value.is_integer() else f"{value:.1f}"


def format_p90(row: pd.Series, metric: RadarMetric) -> str:
    """Per-90 value (or percentage) as text."""
    value = row.get(metric.column)
    if value is None or pd.isna(value):
        return "—"
    return f"{value:.1f}%" if metric.unit == "%" else f"{value:.2f}"


def available_metrics(row: pd.Series) -> list[RadarMetric]:
    """Metrics with a percentile for this player (a missing percentile hides the slice)."""
    return [m for m in METRICS if pd.notna(row.get(f"{m.column}{PCTILE_SUFFIX}"))]


def pizza_data(row: pd.Series) -> dict[str, Any]:
    """Everything the plot needs for one player (`row` from `add_percentiles`)."""
    metrics = available_metrics(row)
    return {
        "params": [m.label for m in metrics],
        "metric_names": [m.label for m in metrics],
        "categories": [m.category for m in metrics],
        "colors": [CATEGORY_COLORS[m.category] for m in metrics],
        "values": [int(round(row[f"{m.column}{PCTILE_SUFFIX}"])) for m in metrics],
        "p90_labels": [format_p90(row, m) for m in metrics],
        "raw_totals": [format_total(row, m) for m in metrics],
        "player_name": row.get("player_name"),
        "team_name": row.get("team_name"),
        "competition_name": row.get("competition_name"),
        "position": row.get("primary_position"),
        "position_group": row.get("position_group"),
        "minutes": float(row.get("minutes", 0.0)),
    }


def breakdown_table(row: pd.Series) -> pd.DataFrame:
    """Rows `Dimension, Metric, Total, Per 90, Percentile` for the detail table."""
    return pd.DataFrame(
        [
            {
                "Dimension": m.category,
                "Metric": m.label,
                "Total": format_total(row, m),
                "Per 90": format_p90(row, m),
                "Percentile": int(round(row[f"{m.column}{PCTILE_SUFFIX}"])),
            }
            for m in available_metrics(row)
        ]
    )
