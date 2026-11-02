"""Season player metrics (counts, xG, per-90) and team xG for/against.

Assumptions and limitations:
- xG is the StatsBomb model value (`shot_xg`), not our own model. Penalties are included in
  `xg` and excluded from `npxg`.
- `primary_position` is the position with the most minutes for the player in a team; the
  position group (GK/DEF/MID/FWD) is derived from it. Players who change roles are
  summarised by a single position, which blurs hybrid profiles.
- Per-90 values are `count / minutes * 90`. This module applies **no minimum-minutes
  threshold**: small samples inflate per-90 rates, so the UI filters on `minutes` (slider,
  default 300).
- Team style biases per-90 volumes (e.g. possession-heavy teams accumulate more passes).
- Coverage is one season and uneven across competitions (see docs/phase0_coverage.md).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from statsbomb_analytics.metrics.minutes import MATCH_END_SQL, load_player_minutes, stint_minutes
from statsbomb_analytics.storage.duckdb_store import query

COUNT_COLUMNS = (
    "shots",
    "goals",
    "xg",
    "npxg",
    "passes",
    "passes_completed",
    "key_passes",
    "assists",
    "dribbles",
    "dribbles_completed",
    "tackles",
    "interceptions",
    "clearances",
    "pressures",
    "ball_recoveries",
)

PLAYER_EVENTS_SQL = """
SELECT player_id, team_id, type, shot_xg, shot_outcome, shot_type, pass_outcome,
       pass_shot_assist, pass_goal_assist, dribble_outcome, duel_type
FROM events
WHERE player_id IS NOT NULL
  AND type IN ('Shot', 'Pass', 'Dribble', 'Duel', 'Interception', 'Clearance',
               'Pressure', 'Ball Recovery')
"""

SHOTS_SQL = "SELECT match_id, team_id, shot_xg FROM events WHERE type = 'Shot'"

_GROUP_BY_POSITION = {
    "Goalkeeper": "GK",
    "Right Back": "DEF",
    "Left Back": "DEF",
    "Right Center Back": "DEF",
    "Left Center Back": "DEF",
    "Center Back": "DEF",
    "Right Wing Back": "DEF",
    "Left Wing Back": "DEF",
    "Center Defensive Midfield": "MID",
    "Left Defensive Midfield": "MID",
    "Right Defensive Midfield": "MID",
    "Center Midfield": "MID",
    "Left Center Midfield": "MID",
    "Right Center Midfield": "MID",
    "Left Midfield": "MID",
    "Right Midfield": "MID",
    "Center Attacking Midfield": "MID",
    "Left Attacking Midfield": "MID",
    "Right Attacking Midfield": "MID",
    "Left Wing": "FWD",
    "Right Wing": "FWD",
    "Center Forward": "FWD",
    "Left Center Forward": "FWD",
    "Right Center Forward": "FWD",
    "Secondary Striker": "FWD",
}


def position_group(position: str | None) -> str | None:
    """Map a StatsBomb position name to GK/DEF/MID/FWD (None if unknown)."""
    return _GROUP_BY_POSITION.get(position) if position else None


def player_event_counts(events: pd.DataFrame) -> pd.DataFrame:
    """Per (team, player) counts and xG from a slice of `events` (see `PLAYER_EVENTS_SQL`)."""
    e = events
    is_shot = e["type"] == "Shot"
    is_pass = e["type"] == "Pass"
    is_dribble = e["type"] == "Dribble"
    is_penalty = e["shot_type"] == "Penalty"
    xg = e["shot_xg"].where(is_shot, 0.0).fillna(0.0)
    flags = pd.DataFrame(
        {
            "player_id": e["player_id"],
            "team_id": e["team_id"],
            "shots": is_shot,
            "goals": is_shot & (e["shot_outcome"] == "Goal"),
            "xg": xg,
            "npxg": xg.where(~is_penalty, 0.0),
            "passes": is_pass,
            "passes_completed": is_pass & e["pass_outcome"].isna(),
            "key_passes": is_pass & (e["pass_shot_assist"] == True),  # noqa: E712
            "assists": is_pass & (e["pass_goal_assist"] == True),  # noqa: E712
            "dribbles": is_dribble,
            "dribbles_completed": is_dribble & (e["dribble_outcome"] == "Complete"),
            "tackles": (e["type"] == "Duel") & (e["duel_type"] == "Tackle"),
            "interceptions": e["type"] == "Interception",
            "clearances": e["type"] == "Clearance",
            "pressures": e["type"] == "Pressure",
            "ball_recoveries": e["type"] == "Ball Recovery",
        }
    )
    grouped = flags.groupby(["team_id", "player_id"], as_index=False).sum()
    grouped[list(COUNT_COLUMNS)] = grouped[list(COUNT_COLUMNS)].astype(float)
    int_cols = [c for c in COUNT_COLUMNS if c not in ("xg", "npxg")]
    grouped[int_cols] = grouped[int_cols].astype(int)
    return grouped


def primary_positions(positions: pd.DataFrame, match_end: pd.DataFrame) -> pd.DataFrame:
    """Position with the most minutes per (team, player), plus its group."""
    stints = stint_minutes(positions, match_end)
    by_pos = stints.groupby(["team_id", "player_id", "position"], as_index=False)["minutes"].sum()
    # Highest minutes first; name as a deterministic tie-breaker.
    by_pos = by_pos.sort_values(["minutes", "position"], ascending=[False, True])
    top = by_pos.drop_duplicates(["team_id", "player_id"]).drop(columns="minutes")
    top = top.rename(columns={"position": "primary_position"})
    top["position_group"] = top["primary_position"].map(position_group)
    return top.reset_index(drop=True)


def add_per90(df: pd.DataFrame, columns: tuple[str, ...] = COUNT_COLUMNS) -> pd.DataFrame:
    """Append `<col>_p90` for each count column. Zero minutes yields NaN, never inf."""
    out = df.copy()
    minutes = out["minutes"].where(out["minutes"] > 0)
    for col in columns:
        out[f"{col}_p90"] = out[col] / minutes * 90.0
    return out


def team_xg(shots: pd.DataFrame) -> pd.DataFrame:
    """xG for/against per team from shots (`match_id`, `team_id`, `shot_xg`).

    xG against is the xG of the opponent in the same matches (2 teams per match).
    """
    per_match = shots.groupby(["match_id", "team_id"], as_index=False)["shot_xg"].sum()
    totals = per_match.groupby("match_id")["shot_xg"].transform("sum")
    per_match["xg_for"] = per_match["shot_xg"]
    per_match["xg_against"] = totals - per_match["shot_xg"]
    return per_match.groupby("team_id", as_index=False).agg(
        matches=("match_id", "nunique"), xg_for=("xg_for", "sum"), xg_against=("xg_against", "sum")
    )


def build_player_table(processed_dir: Path) -> pd.DataFrame:
    """Players with minutes, position, counts and per-90 values (no minutes threshold)."""
    minutes = load_player_minutes(processed_dir)
    positions = query(processed_dir, "SELECT * FROM lineup_positions")
    match_end = query(processed_dir, MATCH_END_SQL)
    counts = player_event_counts(query(processed_dir, PLAYER_EVENTS_SQL))
    table = minutes.merge(primary_positions(positions, match_end), on=["team_id", "player_id"])
    table = table.merge(counts, on=["team_id", "player_id"], how="left")
    table[list(COUNT_COLUMNS)] = table[list(COUNT_COLUMNS)].fillna(0)
    return add_per90(table)


def build_team_xg(processed_dir: Path) -> pd.DataFrame:
    """Team xG for/against over the processed season."""
    return team_xg(query(processed_dir, SHOTS_SQL))
