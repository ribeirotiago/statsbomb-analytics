"""Single player table for the app: counts, per-90, the five valuation methods and labels.

Assumptions and limitations:
- The league of a player is the league of his team (Champions League matches do not define it:
  the competition has a single match in the open data, see docs/phase0_coverage.md).
- Percentages (pass and dribble completion) are NaN when the denominator is zero.
- No minimum-minutes threshold is applied here; the UI filters on `minutes` (default 300).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from statsbomb_analytics.metrics.comparison import METHODS, build_comparison_table
from statsbomb_analytics.metrics.player_stats import build_player_table
from statsbomb_analytics.storage.duckdb_store import query

KEY = ["team_id", "player_id"]
EXCLUDED_COMPETITIONS = ("Champions League",)
PLAYER_METRICS_FILE = "player_metrics.parquet"


def team_competitions(matches: pd.DataFrame) -> pd.DataFrame:
    """League per team: the competition (excluding `EXCLUDED_COMPETITIONS`) with most matches."""
    long = pd.concat(
        [
            matches[["home_team_id", "competition_name"]].set_axis(
                ["team_id", "competition_name"], axis=1
            ),
            matches[["away_team_id", "competition_name"]].set_axis(
                ["team_id", "competition_name"], axis=1
            ),
        ]
    )
    long = long[~long["competition_name"].isin(EXCLUDED_COMPETITIONS)]
    counts = long.groupby(["team_id", "competition_name"], as_index=False).size()
    counts = counts.sort_values(["size", "competition_name"], ascending=[False, True])
    return counts.drop_duplicates("team_id")[["team_id", "competition_name"]].reset_index(drop=True)


def add_completion_rates(table: pd.DataFrame) -> pd.DataFrame:
    """Append `pass_pct` and `dribble_pct` (0-100, NaN without attempts)."""
    out = table.copy()
    out["pass_pct"] = out["passes_completed"] / out["passes"].where(out["passes"] > 0) * 100
    out["dribble_pct"] = (
        out["dribbles_completed"] / out["dribbles"].where(out["dribbles"] > 0) * 100
    )
    return out


def build_player_metrics(processed_dir: Path) -> pd.DataFrame:
    """Players with minutes, position, counts, per-90, method values and team/league labels."""
    base = build_player_table(processed_dir)
    method_cols = [c for c in (*METHODS.values(), "xt", "xgchain", "xgbuildup", "axt", "vaep")]
    comparison = build_comparison_table(processed_dir)[[*KEY, *dict.fromkeys(method_cols)]]
    table = base.merge(comparison, on=KEY, how="left")
    players = query(processed_dir, "SELECT player_id, player_name FROM players")
    teams = query(processed_dir, "SELECT team_id, team_name FROM teams")
    leagues = team_competitions(query(processed_dir, "SELECT * FROM matches"))
    table = table.merge(players, on="player_id", how="left").merge(teams, on="team_id", how="left")
    table = table.merge(leagues, on="team_id", how="left")
    return add_completion_rates(table)
