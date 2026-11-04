"""Single player table for the app: counts, per-90, the five valuation methods and labels.

Assumptions and limitations:
- The league of a player is the league of his team (Champions League matches do not define it:
  the competition has a single match in the open data, see docs/phase0_coverage.md).
- Percentages (pass and dribble completion) are NaN when the denominator is zero.
- No minimum-minutes threshold is applied here; the UI filters on `minutes` (default 300).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from statsbomb_analytics.metrics.action_xt import (
    ACTION_EVENTS_SQL,
    fit_action_xt,
    label_actions,
    player_action_xt,
    score_actions,
)
from statsbomb_analytics.metrics.chains import build_player_chains
from statsbomb_analytics.metrics.comparison import METHODS, build_comparison_table
from statsbomb_analytics.metrics.player_match import (
    PLAYER_MATCH_METRICS_FILE,
    XT_GRID_FILE,
    build_player_match_metrics,
)
from statsbomb_analytics.metrics.player_stats import build_player_table
from statsbomb_analytics.metrics.vaep import VAEP_EVENTS_SQL, fit_vaep, player_vaep
from statsbomb_analytics.metrics.xt import MOVES_SQL, SHOTS_XT_SQL, fit_xt, player_xt
from statsbomb_analytics.storage.duckdb_store import query

KEY = ["team_id", "player_id"]
EXCLUDED_COMPETITIONS = ("Champions League",)
PLAYER_METRICS_FILE = "player_metrics.parquet"
SHOTS_FILE = "shots.parquet"

SHOTS_APP_SQL = """
SELECT e.match_id, e.team_id, e.player_id, p.player_name, t.team_name, e.minute, e.x, e.y,
       e.shot_xg, e.shot_outcome, e.shot_type, e.shot_body_part, e.shot_technique
FROM events e
LEFT JOIN players p ON p.player_id = e.player_id
LEFT JOIN teams t ON t.team_id = e.team_id
WHERE e.type = 'Shot' AND e.period < 5
"""

__all__ = [
    "EXCLUDED_COMPETITIONS",
    "KEY",
    "PLAYER_MATCH_METRICS_FILE",
    "PLAYER_METRICS_FILE",
    "SHOTS_APP_SQL",
    "SHOTS_FILE",
    "XT_GRID_FILE",
    "add_completion_rates",
    "build_all_app_artifacts",
    "build_player_metrics",
    "build_shots_table",
    "team_competitions",
]


def build_shots_table(processed_dir: Path) -> pd.DataFrame:
    """All season shots (excluding period 5 penalty shootouts) for the Shot Map page."""
    return query(processed_dir, SHOTS_APP_SQL)


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


def build_all_app_artifacts(
    processed_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Build season `player_metrics`, `player_match_metrics` and `xt_grid` in one pass."""
    base = build_player_table(processed_dir)

    moves = query(processed_dir, MOVES_SQL)
    shots_xt = query(processed_dir, SHOTS_XT_SQL)
    grid = fit_xt(moves, shots_xt)
    xt_df = player_xt(moves, grid)

    chains_df = build_player_chains(processed_dir)[
        [*KEY, "xgchain", "xgchain_p90", "xgbuildup", "xgbuildup_p90"]
    ]

    axt_events = query(processed_dir, ACTION_EVENTS_SQL)
    axt_actions = label_actions(axt_events)
    axt_model = fit_action_xt(axt_actions)
    scored_axt = score_actions(axt_actions, axt_model)
    axt_df = player_action_xt(scored_axt)

    vaep_events = query(processed_dir, VAEP_EVENTS_SQL)
    vaep_values, _, _ = fit_vaep(vaep_events)
    vaep_df = player_vaep(vaep_values)

    table = base.merge(xt_df, on=KEY, how="left")
    table = table.merge(chains_df, on=KEY, how="left")
    table = table.merge(axt_df, on=KEY, how="left")
    table = table.merge(vaep_df, on=KEY, how="left")

    minutes_pos = table["minutes"].where(table["minutes"] > 0)
    for col in (
        "xt",
        "xt_pass",
        "xt_carry",
        "axt",
        "axt_pass",
        "axt_carry",
        "vaep",
        "vaep_offensive",
        "vaep_defensive",
    ):
        table[col] = table[col].fillna(0.0)
        table[f"{col}_p90"] = table[col] / minutes_pos * 90.0
    for col in ("xgchain", "xgbuildup"):
        table[col] = table[col].fillna(0.0)
        table[f"{col}_p90"] = table[f"{col}_p90"].fillna(0.0)

    players = query(processed_dir, "SELECT player_id, player_name FROM players")
    teams = query(processed_dir, "SELECT team_id, team_name FROM teams")
    leagues = team_competitions(query(processed_dir, "SELECT * FROM matches"))
    table = table.merge(players, on="player_id", how="left").merge(teams, on="team_id", how="left")
    table = table.merge(leagues, on="team_id", how="left")
    season_table = add_completion_rates(table)

    match_table = build_player_match_metrics(
        processed_dir, grid=grid, scored_axt=scored_axt, vaep_values=vaep_values
    )
    return season_table, match_table, grid
