"""Expected Threat (xT) and PPDA.

xT follows Karun Singh's method (also used in `atletico-analytics`), re-implemented with
vectorised numpy on StatsBomb coordinates (120 x 80, attack towards increasing x):

    xT[z] = P(shot | z) * xG(z) + P(move | z) * sum_z' T[z -> z'] * xT[z']

solved by value iteration until convergence.

Assumptions and limitations:
- The grid is fitted on the processed season itself: values are league/season specific and
  noisy in rarely visited zones (no smoothing is applied; the empty-cell share is reported).
- Failed moves enter the grid fit (they lower P(transition), as in Singh's method), but only
  successful passes and carries earn per-action xT (`xT[end] - xT[start]`). Failed passes are
  not penalised per action, so volume passers look better than in models with explicit
  turnover costs.
- xT per 90 is biased by role and team style (deep-lying and possession teams accumulate more
  moves) and inflated by small minutes samples; the UI filters on minutes.
- PPDA = opponent passes in their own 60% of the pitch / team defensive actions in the
  opposite 60% (tackles, interceptions, fouls committed, dribbled past). Per-match counts are
  summed over the season. Low PPDA means a more intense press; it ignores pressure events
  and is sensitive to game state and opponent style.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from statsbomb_analytics.metrics.minutes import load_player_minutes
from statsbomb_analytics.storage.duckdb_store import query

FIELD_LENGTH = 120.0
FIELD_WIDTH = 80.0
N_ROWS = 12  # along the width (y)
N_COLS = 16  # along the length (x)

# PPDA zone: share of the pitch (from the pressing team's attacking direction).
PPDA_ZONE_START_X = 0.4 * FIELD_LENGTH  # defensive actions at x >= 48
PPDA_OPPONENT_MAX_X = 0.6 * FIELD_LENGTH  # opponent passes at x <= 72 (their own frame)

MOVES_SQL = """
SELECT match_id, team_id, player_id, type, x, y,
       COALESCE(pass_end_x, carry_end_x) AS end_x,
       COALESCE(pass_end_y, carry_end_y) AS end_y,
       (type = 'Carry' OR pass_outcome IS NULL) AS success
FROM events
WHERE type IN ('Pass', 'Carry') AND x IS NOT NULL AND player_id IS NOT NULL
"""

SHOTS_XT_SQL = "SELECT x, y, shot_xg FROM events WHERE type = 'Shot' AND shot_xg IS NOT NULL"

PPDA_SQL = """
SELECT match_id, team_id, type, duel_type, x
FROM events
WHERE x IS NOT NULL
  AND (type IN ('Pass', 'Interception', 'Foul Committed', 'Dribbled Past')
       OR (type = 'Duel' AND duel_type = 'Tackle'))
"""


def zone_index(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Flat zone index (`row * N_COLS + col`), clipping coordinates into the pitch."""
    col = np.clip((np.asarray(x, dtype=float) / FIELD_LENGTH * N_COLS).astype(int), 0, N_COLS - 1)
    row = np.clip((np.asarray(y, dtype=float) / FIELD_WIDTH * N_ROWS).astype(int), 0, N_ROWS - 1)
    return row * N_COLS + col


def fit_xt(
    moves: pd.DataFrame,
    shots: pd.DataFrame,
    tolerance: float = 1e-9,
    max_iterations: int = 200,
) -> np.ndarray:
    """Fit the xT grid (shape `N_ROWS x N_COLS`).

    `moves`: all passes/carries with `x, y, end_x, end_y` and an optional boolean `success`
    (default: all successful). As in Singh's original method, failed moves count in the
    denominator of P(move) but create no transition, so their probability mass is lost
    (turnover cost). Fitting on successes only would inflate the value of deep zones.
    `shots`: `x, y, shot_xg`.
    """
    n = N_ROWS * N_COLS
    start = zone_index(moves["x"].to_numpy(), moves["y"].to_numpy())
    end = zone_index(moves["end_x"].to_numpy(), moves["end_y"].to_numpy())
    success = (
        moves["success"].to_numpy(dtype=bool) if "success" in moves else np.ones(len(moves), bool)
    )
    move_count = np.bincount(start, minlength=n).astype(float)
    transition = np.zeros((n, n))
    np.add.at(transition, (start[success], end[success]), 1.0)

    shot_zone = zone_index(shots["x"].to_numpy(), shots["y"].to_numpy())
    shot_count = np.bincount(shot_zone, minlength=n).astype(float)
    shot_xg_sum = np.bincount(shot_zone, weights=shots["shot_xg"].to_numpy(), minlength=n)

    total = shot_count + move_count
    safe_total = np.where(total > 0, total, 1.0)
    shot_prob = shot_count / safe_total
    move_prob = move_count / safe_total
    shot_value = shot_xg_sum / np.where(shot_count > 0, shot_count, 1.0)
    transition = transition / np.where(move_count > 0, move_count, 1.0)[:, None]

    base = shot_prob * shot_value
    xt = np.zeros(n)
    for _ in range(max_iterations):
        new = base + move_prob * (transition @ xt)
        converged = np.max(np.abs(new - xt)) < tolerance
        xt = new
        if converged:
            break
    return xt.reshape(N_ROWS, N_COLS)


def action_xt(moves: pd.DataFrame, grid: np.ndarray) -> pd.Series:
    """xT gained by each move: `grid[end] - grid[start]` for successes, NaN otherwise."""
    flat = grid.ravel()
    start = zone_index(moves["x"].to_numpy(), moves["y"].to_numpy())
    end = zone_index(moves["end_x"].to_numpy(), moves["end_y"].to_numpy())
    gain = flat[end] - flat[start]
    return pd.Series(np.where(moves["success"].to_numpy(dtype=bool), gain, np.nan), moves.index)


def player_xt(moves: pd.DataFrame, grid: np.ndarray) -> pd.DataFrame:
    """Season xT per (team, player), split into passes and carries (successful moves only)."""
    df = moves.assign(xt=action_xt(moves, grid)).dropna(subset=["xt"])
    df["xt_pass"] = df["xt"].where(df["type"] == "Pass", 0.0)
    df["xt_carry"] = df["xt"].where(df["type"] == "Carry", 0.0)
    return df.groupby(["team_id", "player_id"], as_index=False)[["xt", "xt_pass", "xt_carry"]].sum()


def ppda(events: pd.DataFrame) -> pd.DataFrame:
    """Season PPDA per team from `PPDA_SQL` events (`match_id, team_id, type, duel_type, x`)."""
    is_pass = events["type"] == "Pass"
    is_defensive = ~is_pass & (events["x"] >= PPDA_ZONE_START_X)
    passes = events[is_pass & (events["x"] <= PPDA_OPPONENT_MAX_X)]
    defensive = events[is_defensive]
    per_match = passes.groupby(["match_id", "team_id"]).size().rename("passes").reset_index()
    allowed = per_match.merge(per_match, on="match_id", suffixes=("", "_opp"))
    allowed = allowed[allowed["team_id"] != allowed["team_id_opp"]]
    allowed = allowed[["match_id", "team_id_opp", "passes"]].rename(
        columns={"team_id_opp": "team_id", "passes": "opp_passes"}
    )
    actions = defensive.groupby(["match_id", "team_id"]).size().rename("def_actions").reset_index()
    merged = allowed.merge(actions, on=["match_id", "team_id"], how="left").fillna(
        {"def_actions": 0}
    )
    totals = merged.groupby("team_id", as_index=False).agg(
        matches=("match_id", "nunique"),
        opp_passes=("opp_passes", "sum"),
        def_actions=("def_actions", "sum"),
    )
    totals["ppda"] = totals["opp_passes"] / totals["def_actions"].where(totals["def_actions"] > 0)
    return totals


def build_xt_grid(processed_dir: Path) -> np.ndarray:
    """Fit the xT grid from the processed season."""
    moves = query(processed_dir, MOVES_SQL)
    shots = query(processed_dir, SHOTS_XT_SQL)
    return fit_xt(moves, shots)


def build_player_xt(processed_dir: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """Player xT table with minutes and per-90 (no minimum-minutes threshold) and the grid."""
    moves = query(processed_dir, MOVES_SQL)
    shots = query(processed_dir, SHOTS_XT_SQL)
    grid = fit_xt(moves, shots)
    table = load_player_minutes(processed_dir).merge(
        player_xt(moves, grid), on=["team_id", "player_id"], how="left"
    )
    for col in ("xt", "xt_pass", "xt_carry"):
        table[col] = table[col].fillna(0.0)
        table[f"{col}_p90"] = table[col] / table["minutes"].where(table["minutes"] > 0) * 90.0
    return table, grid


def build_ppda(processed_dir: Path) -> pd.DataFrame:
    """Team PPDA over the processed season."""
    return ppda(query(processed_dir, PPDA_SQL))
