"""Per-match player metrics, action-valuation match logs and single-match tactical breakdowns.

Assumptions and limitations:
- Single-match action-valuation numbers (Singh xT, xGChain, xGBuildup, Action xT, VAEP) are
  summed over the player's actions in that match (no per-90 scaling inside a single match, as
  a 10-minute substitute appearance would explode per-90 values).
- In the tactical pitch maps of the Player Match Report, successful passes/carries receive
  `grid[end] - grid[start]` (which can be positive -> xT Gain, or non-positive -> xT Loss).
  Incomplete passes receive the turnover cost `-grid[mirror(start)]` (threat handed to the
  opponent at that location), and defensive actions receive `+grid[mirror(start)]` (`xTP`,
  Expected Threat Prevented). Note that season-level Singh xT (`xt`) follows Singh's canonical
  definition (positive/negative gains of successful moves only, without turnover subtraction).
- Progressive passes/carries follow the standard geometric criterion: the action moves the ball
  at least 10 yards closer to the centre of the opponent goal `(120, 40)` and does not start and
  end inside the team's own defensive third (`x < 40`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from statsbomb_analytics.metrics.action_xt import (
    ACTION_EVENTS_SQL,
    fit_action_xt,
    label_actions,
    score_actions,
)
from statsbomb_analytics.metrics.chains import CHAIN_EVENTS_SQL, possession_values
from statsbomb_analytics.metrics.minutes import (
    MATCH_END_SQL,
    player_match_minutes,
    stint_minutes,
)
from statsbomb_analytics.metrics.player_stats import position_group
from statsbomb_analytics.metrics.vaep import VAEP_EVENTS_SQL, fit_vaep
from statsbomb_analytics.metrics.xt import (
    FIELD_LENGTH,
    FIELD_WIDTH,
    MOVES_SQL,
    N_COLS,
    N_ROWS,
    SHOTS_XT_SQL,
    action_xt,
    fit_xt,
    zone_index,
)
from statsbomb_analytics.storage.duckdb_store import query

MATCH_KEY = ["match_id", "team_id", "player_id"]
PLAYER_MATCH_METRICS_FILE = "player_match_metrics.parquet"
XT_GRID_FILE = "xt_grid.npy"

MATCH_METHODS: dict[str, str] = {
    "Singh xT": "xt",
    "xGChain": "xgchain",
    "xGBuildup": "xgbuildup",
    "Action xT": "axt",
    "VAEP": "vaep",
}

MATCH_COUNTS_SQL = """
SELECT match_id, team_id, player_id, type, shot_xg, shot_outcome, shot_type,
       pass_outcome, pass_shot_assist, pass_goal_assist
FROM events
WHERE player_id IS NOT NULL
  AND period <= 4
  AND type IN ('Shot', 'Pass')
"""

_ON_TARGET_OUTCOMES = ("Goal", "Saved", "Saved to Post")
_TOUCH_TYPES = (
    "Pass",
    "Ball Receipt*",
    "Carry",
    "Dribble",
    "Shot",
    "Miscontrol",
    "Dispossessed",
    "Clearance",
    "Interception",
    "Ball Recovery",
    "Block",
    "Duel",
)


def match_positions(positions: pd.DataFrame, match_end: pd.DataFrame) -> pd.DataFrame:
    """Primary position (most minutes in that match) and group per (match, team, player)."""
    stints = stint_minutes(positions, match_end)
    by_pos = stints.groupby([*MATCH_KEY, "position"], as_index=False)["minutes"].sum()
    by_pos = by_pos.sort_values(["minutes", "position"], ascending=[False, True])
    top = by_pos.drop_duplicates(MATCH_KEY).drop(columns="minutes")
    top["position_group"] = top["position"].map(position_group)
    return top.reset_index(drop=True)


def match_event_counts(events: pd.DataFrame) -> pd.DataFrame:
    """Per (match, team, player) basic counts and xG from `MATCH_COUNTS_SQL` events."""
    if events.empty:
        return pd.DataFrame(
            columns=[
                *MATCH_KEY,
                "goals",
                "assists",
                "shots",
                "xg",
                "npxg",
                "key_passes",
                "passes",
                "passes_completed",
            ]
        )
    e = events
    is_shot = (e["type"] == "Shot").fillna(False).astype(bool)
    is_pass = (e["type"] == "Pass").fillna(False).astype(bool)
    is_penalty = (e["shot_type"] == "Penalty").fillna(False).astype(bool)
    xg = e["shot_xg"].where(is_shot, 0.0).fillna(0.0)
    goal_assist = e["pass_goal_assist"].fillna(False).astype(bool)
    shot_assist = e["pass_shot_assist"].fillna(False).astype(bool)
    flags = pd.DataFrame(
        {
            "match_id": e["match_id"],
            "team_id": e["team_id"],
            "player_id": e["player_id"],
            "goals": (is_shot & (e["shot_outcome"] == "Goal").fillna(False)).astype(int),
            "assists": (is_pass & goal_assist).astype(int),
            "shots": is_shot.astype(int),
            "xg": xg.astype(float),
            "npxg": xg.where(~is_penalty, 0.0).astype(float),
            "key_passes": (is_pass & shot_assist).astype(int),
            "passes": is_pass.astype(int),
            "passes_completed": (is_pass & e["pass_outcome"].isna()).astype(int),
        }
    )
    return flags.groupby(MATCH_KEY, as_index=False).sum()


def player_match_xt(moves: pd.DataFrame, grid: np.ndarray) -> pd.DataFrame:
    """Per (match, team, player) Singh xT (`xt`, `xt_pass`, `xt_carry`)."""
    df = moves.assign(xt=action_xt(moves, grid)).dropna(subset=["xt"])
    df["xt_pass"] = df["xt"].where(df["type"] == "Pass", 0.0)
    df["xt_carry"] = df["xt"].where(df["type"] == "Carry", 0.0)
    return df.groupby(MATCH_KEY, as_index=False)[["xt", "xt_pass", "xt_carry"]].sum()


def player_match_chains(events: pd.DataFrame) -> pd.DataFrame:
    """Per (match, team, player) `xgchain` and `xgbuildup`."""
    poss_key = ["match_id", "possession", "team_id"]
    values = possession_values(events)
    key = [*poss_key, "player_id"]
    participants = events[key].drop_duplicates().merge(values, on=poss_key)

    finishers = events[
        (events["type"] == "Shot") | (events["pass_shot_assist"].fillna(False).astype(bool))
    ][key].drop_duplicates()
    finishers["is_finisher"] = True
    participants = participants.merge(finishers, on=key, how="left")
    participants["is_finisher"] = participants["is_finisher"].fillna(False).astype(bool)
    participants["buildup_xg"] = participants["chain_xg"].where(~participants["is_finisher"], 0.0)

    return participants.groupby(MATCH_KEY, as_index=False).agg(
        xgchain=("chain_xg", "sum"), xgbuildup=("buildup_xg", "sum")
    )


def player_match_action_xt(scored: pd.DataFrame) -> pd.DataFrame:
    """Per (match, team, player) action-based xT (`axt`, `axt_pass`, `axt_carry`)."""
    df = scored[scored["success"].astype(bool)].copy()
    df["axt_pass"] = df["axt"].where(df["type"] == "Pass", 0.0)
    df["axt_carry"] = df["axt"].where(df["type"] == "Carry", 0.0)
    return df.groupby(MATCH_KEY, as_index=False)[["axt", "axt_pass", "axt_carry"]].sum()


def player_match_vaep(values: pd.DataFrame) -> pd.DataFrame:
    """Per (match, team, player) VAEP (`vaep`, `vaep_offensive`, `vaep_defensive`)."""
    return values.groupby(MATCH_KEY, as_index=False).agg(
        vaep=("vaep", "sum"),
        vaep_offensive=("offensive", "sum"),
        vaep_defensive=("defensive", "sum"),
    )


def build_player_match_metrics(
    processed_dir: Path,
    grid: np.ndarray | None = None,
    scored_axt: pd.DataFrame | None = None,
    vaep_values: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the per-match player table with minutes, basic stats and all 5 valuation methods."""
    positions = query(processed_dir, "SELECT * FROM lineup_positions")
    match_end = query(processed_dir, MATCH_END_SQL)
    minutes = player_match_minutes(positions, match_end)
    minutes = minutes[minutes["minutes"] > 0].reset_index(drop=True)
    pos = match_positions(positions, match_end)

    moves = query(processed_dir, MOVES_SQL)
    if grid is None:
        shots_xt = query(processed_dir, SHOTS_XT_SQL)
        grid = fit_xt(moves, shots_xt)
    xt_df = player_match_xt(moves, grid)

    chain_events = query(processed_dir, CHAIN_EVENTS_SQL)
    chains_df = player_match_chains(chain_events)

    if scored_axt is None:
        axt_events = query(processed_dir, ACTION_EVENTS_SQL)
        axt_actions = label_actions(axt_events)
        axt_model = fit_action_xt(axt_actions)
        scored_axt = score_actions(axt_actions, axt_model)
    axt_df = player_match_action_xt(scored_axt)

    if vaep_values is None:
        vaep_events = query(processed_dir, VAEP_EVENTS_SQL)
        vaep_values, _, _ = fit_vaep(vaep_events)
    vaep_df = player_match_vaep(vaep_values)

    counts_df = match_event_counts(query(processed_dir, MATCH_COUNTS_SQL))

    table = minutes.merge(pos, on=MATCH_KEY, how="left")
    for part in (counts_df, xt_df, chains_df, axt_df, vaep_df):
        table = table.merge(part, on=MATCH_KEY, how="left")

    int_cols = ("goals", "assists", "shots", "key_passes", "passes", "passes_completed")
    float_cols = (
        "xg",
        "npxg",
        "xt",
        "xt_pass",
        "xt_carry",
        "xgchain",
        "xgbuildup",
        "axt",
        "axt_pass",
        "axt_carry",
        "vaep",
        "vaep_offensive",
        "vaep_defensive",
    )
    for col in int_cols:
        table[col] = table[col].fillna(0).astype(int)
    for col in float_cols:
        table[col] = table[col].fillna(0.0).astype(float)

    players = query(processed_dir, "SELECT player_id, player_name FROM players")
    teams = query(processed_dir, "SELECT team_id, team_name FROM teams")
    matches = query(
        processed_dir,
        """
        SELECT m.match_id, m.competition_name, m.match_date,
               m.home_team_id, h.team_name AS home_team, m.home_score,
               m.away_team_id, a.team_name AS away_team, m.away_score
        FROM matches m
        JOIN teams h ON h.team_id = m.home_team_id
        JOIN teams a ON a.team_id = m.away_team_id
        """,
    )
    table = (
        table.merge(players, on="player_id", how="left")
        .merge(teams, on="team_id", how="left")
        .merge(matches, on="match_id", how="left")
    )
    return table.sort_values(["match_date", "match_id", "team_id", "player_name"]).reset_index(
        drop=True
    )


def match_top_contributors(
    match_df: pd.DataFrame, method_col: str, n: int = 3
) -> dict[int, pd.DataFrame]:
    """Top `n` players per `team_id` in one match for `method_col`."""
    cols = ["player_id", "player_name", "position", "minutes", method_col]
    available = [c for c in cols if c in match_df.columns]
    result: dict[int, pd.DataFrame] = {}
    for team_id, part in match_df.groupby("team_id"):
        ranked = part.sort_values([method_col, "minutes"], ascending=[False, False]).head(n)
        result[int(team_id)] = ranked[available].reset_index(drop=True)
    return result


def _is_progressive(
    x0: pd.Series, y0: pd.Series, x1: pd.Series, y1: pd.Series, min_gain: float = 10.0
) -> pd.Series:
    """True when an action moves at least `min_gain` yards closer to `(120, 40)` outside own 3rd."""
    d0 = np.hypot(FIELD_LENGTH - x0.astype(float), FIELD_WIDTH / 2.0 - y0.astype(float))
    d1 = np.hypot(FIELD_LENGTH - x1.astype(float), FIELD_WIDTH / 2.0 - y1.astype(float))
    not_deep = ~((x0.astype(float) < 40.0) & (x1.astype(float) < 40.0))
    return ((d0 - d1) >= min_gain) & not_deep


def match_player_detailed_stats(
    events: pd.DataFrame, player_id: int
) -> dict[str, dict[str, float | int]]:
    """Compute the 4 categories of single-match stats for `player_id` from match `events`."""
    mine = events[(events["player_id"] == player_id) & (events["period"] <= 4)].copy()
    shots = mine[mine["type"] == "Shot"]
    passes = mine[mine["type"] == "Pass"]
    carries = mine[mine["type"] == "Carry"]
    dribbles = mine[mine["type"] == "Dribble"]

    is_pen = (shots["shot_type"] == "Penalty").fillna(False).astype(bool)
    xg = float(shots["shot_xg"].fillna(0.0).sum())
    npxg = float(shots.loc[~is_pen, "shot_xg"].fillna(0.0).sum())
    shots_tot = len(shots)
    shots_on = int(shots["shot_outcome"].isin(_ON_TARGET_OUTCOMES).sum())
    goals = int((shots["shot_outcome"] == "Goal").fillna(False).sum())
    key_passes = int(passes["pass_shot_assist"].fillna(False).astype(bool).sum())
    assists = int(passes["pass_goal_assist"].fillna(False).astype(bool).sum())
    drib_ok = int((dribbles["dribble_outcome"] == "Complete").fillna(False).sum())
    drib_tot = len(dribbles)

    pass_ok = passes[passes["pass_outcome"].isna()]
    pass_tot = len(passes)
    pass_ok_n = len(pass_ok)
    pass_pct = (pass_ok_n / pass_tot * 100.0) if pass_tot > 0 else 0.0
    prog_passes = int(
        _is_progressive(
            pass_ok["x"], pass_ok["y"], pass_ok["pass_end_x"], pass_ok["pass_end_y"]
        ).sum()
        if not pass_ok.empty
        else 0
    )
    crosses = passes[passes["pass_cross"].fillna(False).astype(bool)]
    cross_ok = int(crosses["pass_outcome"].isna().sum())
    switches = int(passes["pass_switch"].fillna(False).astype(bool).sum())
    through_balls = int(passes["pass_through_ball"].fillna(False).astype(bool).sum())
    touches = int(mine["type"].isin(_TOUCH_TYPES).sum())

    tackles = mine[(mine["type"] == "Duel") & (mine["duel_type"] == "Tackle")]
    tackles_won = int(
        tackles["duel_outcome"].isin(["Won", "Success", "Success In Play", "Success Out"]).sum()
    )
    interceptions = int((mine["type"] == "Interception").sum())
    clearances = int((mine["type"] == "Clearance").sum())
    recoveries = int(
        (
            (mine["type"] == "Ball Recovery")
            & (~mine["ball_recovery_failure"].fillna(False).astype(bool))
        ).sum()
    )
    blocks = int((mine["type"] == "Block").sum())
    pressures = int((mine["type"] == "Pressure").sum())
    counterpressures = int(mine["counterpress"].fillna(False).astype(bool).sum())
    fouls = int((mine["type"] == "Foul Committed").sum())

    valid_carries = carries.dropna(subset=["x", "y", "carry_end_x", "carry_end_y"])
    if not valid_carries.empty:
        carry_dist = float(
            np.hypot(
                valid_carries["carry_end_x"].astype(float) - valid_carries["x"].astype(float),
                valid_carries["carry_end_y"].astype(float) - valid_carries["y"].astype(float),
            ).sum()
        )
        prog_carries = int(
            _is_progressive(
                valid_carries["x"],
                valid_carries["y"],
                valid_carries["carry_end_x"],
                valid_carries["carry_end_y"],
            ).sum()
        )
    else:
        carry_dist = 0.0
        prog_carries = 0
    under_pressure = int(mine["under_pressure"].fillna(False).astype(bool).sum())

    return {
        "attack": {
            "xg": xg,
            "npxg": npxg,
            "shots": shots_tot,
            "shots_on_target": shots_on,
            "goals": goals,
            "key_passes": key_passes,
            "assists": assists,
            "dribbles_completed": drib_ok,
            "dribbles": drib_tot,
        },
        "passes": {
            "passes_completed": pass_ok_n,
            "passes": pass_tot,
            "pass_pct": pass_pct,
            "progressive_passes": prog_passes,
            "crosses_completed": cross_ok,
            "crosses": len(crosses),
            "switches": switches,
            "through_balls": through_balls,
            "touches": touches,
        },
        "defense": {
            "tackles_won": tackles_won,
            "tackles": len(tackles),
            "interceptions": interceptions,
            "clearances": clearances,
            "ball_recoveries": recoveries,
            "blocks": blocks,
            "pressures": pressures,
            "counterpressures": counterpressures,
            "fouls_committed": fouls,
        },
        "progression": {
            "carries": len(carries),
            "carry_distance": carry_dist,
            "progressive_carries": prog_carries,
            "under_pressure": under_pressure,
        },
    }


def _zone_values(x: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    flat = np.asarray(grid, dtype=float).reshape(N_ROWS * N_COLS)
    idx = zone_index(x, y)
    return flat[idx]


def score_match_spatial_events(
    events: pd.DataFrame, player_id: int, grid: np.ndarray | None = None
) -> dict[str, pd.DataFrame]:
    """Extract and score spatial events of `player_id` for the 5 tactical pitch maps."""
    mine = events[
        (events["player_id"] == player_id) & (events["period"] <= 4) & events["x"].notna()
    ].copy()

    # 1. Passes & Carries (moves)
    raw_moves = mine[mine["type"].isin(["Pass", "Carry"])].copy()
    raw_moves["end_x"] = raw_moves["pass_end_x"].combine_first(raw_moves["carry_end_x"])
    raw_moves["end_y"] = raw_moves["pass_end_y"].combine_first(raw_moves["carry_end_y"])
    raw_moves = raw_moves.dropna(subset=["x", "y", "end_x", "end_y"]).reset_index(drop=True)

    if not raw_moves.empty:
        is_pass = (raw_moves["type"] == "Pass").fillna(False).astype(bool)
        success = (~is_pass) | raw_moves["pass_outcome"].isna()
        is_cross = is_pass & raw_moves["pass_cross"].fillna(False).astype(bool)
        x0 = raw_moves["x"].to_numpy(dtype=float)
        y0 = raw_moves["y"].to_numpy(dtype=float)
        x1 = raw_moves["end_x"].to_numpy(dtype=float)
        y1 = raw_moves["end_y"].to_numpy(dtype=float)
        if grid is not None:
            start_v = _zone_values(x0, y0, grid)
            end_v = _zone_values(x1, y1, grid)
            opp_v = _zone_values(FIELD_LENGTH - x0, FIELD_WIDTH - y0, grid)
            xt_val = np.where(success.to_numpy(dtype=bool), end_v - start_v, -opp_v)
        else:
            xt_val = np.zeros(len(raw_moves))
        moves = pd.DataFrame(
            {
                "minute": raw_moves["minute"].astype(int),
                "event_type": np.where(is_pass, "pass", "carry"),
                "outcome": success.astype(bool),
                "is_cross": is_cross.astype(bool),
                "x_start": x0,
                "y_start": y0,
                "x_end": x1,
                "y_end": y1,
                "xt_value": xt_val,
            }
        )
    else:
        moves = pd.DataFrame(
            columns=[
                "minute",
                "event_type",
                "outcome",
                "is_cross",
                "x_start",
                "y_start",
                "x_end",
                "y_end",
                "xt_value",
            ]
        )

    # 2. Defensive actions
    is_tackle = (mine["type"] == "Duel") & (mine["duel_type"] == "Tackle")
    is_rec = (mine["type"] == "Ball Recovery") & (
        ~mine["ball_recovery_failure"].fillna(False).astype(bool)
    )
    def_mask = (
        is_tackle | is_rec | mine["type"].isin(["Interception", "Clearance", "Block", "Pressure"])
    )
    raw_def = mine[def_mask].reset_index(drop=True)
    if not raw_def.empty:
        subtypes = np.select(
            [
                raw_def["type"] == "Duel",
                raw_def["type"] == "Interception",
                raw_def["type"] == "Clearance",
                raw_def["type"] == "Ball Recovery",
                raw_def["type"] == "Block",
                raw_def["type"] == "Pressure",
            ],
            ["tackle", "interception", "clearance", "ball-recovery", "block", "pressure"],
            default="other",
        )
        dx = raw_def["x"].to_numpy(dtype=float)
        dy = raw_def["y"].to_numpy(dtype=float)
        xtp = (
            _zone_values(FIELD_LENGTH - dx, FIELD_WIDTH - dy, grid)
            if grid is not None
            else np.zeros(len(raw_def))
        )
        defensive = pd.DataFrame(
            {
                "minute": raw_def["minute"].astype(int),
                "event_subtype": subtypes,
                "x_start": dx,
                "y_start": dy,
                "xt_value": xtp,
            }
        )
    else:
        defensive = pd.DataFrame(
            columns=["minute", "event_subtype", "x_start", "y_start", "xt_value"]
        )

    # 3. Shots
    raw_shots = mine[mine["type"] == "Shot"].reset_index(drop=True)
    if not raw_shots.empty:
        out_str = raw_shots["shot_outcome"].fillna("")
        shot_sub = np.select(
            [
                out_str == "Goal",
                out_str.isin(["Saved", "Saved to Post"]),
                out_str == "Blocked",
                out_str == "Post",
            ],
            ["goal", "save", "block", "post"],
            default="miss",
        )
        shots = pd.DataFrame(
            {
                "minute": raw_shots["minute"].astype(int),
                "event_subtype": shot_sub,
                "shot_outcome": out_str,
                "shot_body_part": raw_shots["shot_body_part"].fillna(""),
                "x_start": raw_shots["x"].to_numpy(dtype=float),
                "y_start": raw_shots["y"].to_numpy(dtype=float),
                "xg": raw_shots["shot_xg"].fillna(0.0).to_numpy(dtype=float),
            }
        )
    else:
        shots = pd.DataFrame(
            columns=[
                "minute",
                "event_subtype",
                "shot_outcome",
                "shot_body_part",
                "x_start",
                "y_start",
                "xg",
            ]
        )

    # 4. Touches for heatmap
    raw_touches = mine[mine["type"].isin(_TOUCH_TYPES)]
    touches = pd.DataFrame(
        {
            "x": raw_touches["x"].to_numpy(dtype=float),
            "y": raw_touches["y"].to_numpy(dtype=float),
        }
    )

    return {
        "moves": moves,
        "defensive": defensive,
        "shots": shots,
        "touches": touches,
    }


def normalize_player_match_log(
    df_log: pd.DataFrame, methods: dict[str, str] = MATCH_METHODS
) -> pd.DataFrame:
    """Append `<col>_pct` (0-100 within-player match percentile) and `<col>_z` (Z-score)."""
    out = df_log.copy()
    for col in methods.values():
        if col not in out.columns:
            continue
        series = out[col].astype(float)
        out[f"{col}_pct"] = series.rank(pct=True, method="average") * 100.0
        std = float(series.std(ddof=0))
        mean = float(series.mean())
        out[f"{col}_z"] = (series - mean) / std if std > 1e-12 else 0.0
    return out
