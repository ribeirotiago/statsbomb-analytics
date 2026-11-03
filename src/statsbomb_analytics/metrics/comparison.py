"""Comparison of the action-valuation methods (decision D9).

All methods are compared on the same players, using per-90 values and ranks only: scales and
meanings differ (Singh xT is a gain, action-based xT a state value, VAEP a probability change),
so values are never compared directly. Rank agreement is measured with Spearman's correlation.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from statsbomb_analytics.metrics.action_xt import build_player_action_xt
from statsbomb_analytics.metrics.chains import build_player_chains
from statsbomb_analytics.metrics.minutes import MATCH_END_SQL
from statsbomb_analytics.metrics.player_stats import primary_positions
from statsbomb_analytics.metrics.vaep import build_player_vaep
from statsbomb_analytics.metrics.xt import build_player_xt
from statsbomb_analytics.storage.duckdb_store import query

METHODS: dict[str, str] = {
    "Singh xT": "xt_p90",
    "xGChain": "xgchain_p90",
    "xGBuildup": "xgbuildup_p90",
    "Action xT": "axt_p90",
    "VAEP": "vaep_p90",
}
KEY = ["team_id", "player_id"]
MIN_MINUTES = 300.0
MIN_GROUP_SIZE = 20


def filter_minutes(table: pd.DataFrame, min_minutes: float = MIN_MINUTES) -> pd.DataFrame:
    """Keep players with at least `min_minutes` played (small-sample guard, D2)."""
    return table[table["minutes"] >= min_minutes].reset_index(drop=True)


def outfield_players(table: pd.DataFrame) -> pd.DataFrame:
    """Drop goalkeepers (and players without a known position group)."""
    return table[table["position_group"].isin(["DEF", "MID", "FWD"])].reset_index(drop=True)


def spearman_matrix(table: pd.DataFrame, methods: dict[str, str] = METHODS) -> pd.DataFrame:
    """Spearman rank correlation between the per-90 columns of each method."""
    values = table[list(methods.values())].rename(columns={v: k for k, v in methods.items()})
    return values.corr(method="spearman")


def spearman_by_group(
    table: pd.DataFrame,
    group_col: str = "position_group",
    methods: dict[str, str] = METHODS,
    min_size: int = MIN_GROUP_SIZE,
) -> pd.DataFrame:
    """Spearman correlation of every method pair inside each group (skips small groups)."""
    names = list(methods)
    rows = []
    for group, part in table.groupby(group_col):
        if len(part) < min_size:
            continue
        corr = spearman_matrix(part, methods)
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                rows.append({group_col: group, "pair": f"{a} / {b}", "spearman": corr.loc[a, b]})
    return pd.DataFrame(rows, columns=[group_col, "pair", "spearman"])


def top_players(
    table: pd.DataFrame, methods: dict[str, str] = METHODS, n: int = 10
) -> dict[str, pd.DataFrame]:
    """Top `n` players per method by per-90 value."""
    cols = ["player_name", "primary_position", "minutes"]
    return {
        name: table.nlargest(n, col)[[*cols, col]].reset_index(drop=True)
        for name, col in methods.items()
    }


def rank_shifts(
    table: pd.DataFrame,
    method_a: str,
    method_b: str,
    methods: dict[str, str] = METHODS,
    n: int = 10,
) -> pd.DataFrame:
    """Players ranked much higher by `method_a` than by `method_b` (percentile difference)."""
    pct_a = table[methods[method_a]].rank(pct=True)
    pct_b = table[methods[method_b]].rank(pct=True)
    out = table[["player_name", "primary_position", "minutes"]].assign(
        **{f"pct_{method_a}": pct_a, f"pct_{method_b}": pct_b, "gap": pct_a - pct_b}
    )
    return out.nlargest(n, "gap").reset_index(drop=True)


def mean_percentile_by_group(
    table: pd.DataFrame, group_col: str = "position_group", methods: dict[str, str] = METHODS
) -> pd.DataFrame:
    """Mean percentile of each method per group: shows which positions a method favours."""
    pct = table[list(methods.values())].rank(pct=True)
    pct.columns = list(methods)
    return pct.groupby(table[group_col]).mean()


def build_comparison_table(processed_dir: Path) -> pd.DataFrame:
    """One row per player with the per-90 value of every method, position and name."""
    xt_table, _ = build_player_xt(processed_dir)
    chains = build_player_chains(processed_dir)
    axt_table, _ = build_player_action_xt(processed_dir)
    vaep_table, _, _ = build_player_vaep(processed_dir)

    table = xt_table[[*KEY, "minutes", "matches", "xt", "xt_p90"]]
    for part, cols in (
        (chains, ["xgchain", "xgchain_p90", "xgbuildup", "xgbuildup_p90"]),
        (axt_table, ["axt", "axt_p90"]),
        (vaep_table, ["vaep", "vaep_p90"]),
    ):
        table = table.merge(part[[*KEY, *cols]], on=KEY, how="left")

    positions = primary_positions(
        query(processed_dir, "SELECT * FROM lineup_positions"),
        query(processed_dir, MATCH_END_SQL),
    )
    players = query(processed_dir, "SELECT player_id, player_name FROM players")
    table = table.merge(positions, on=KEY, how="left").merge(players, on="player_id", how="left")
    return table
