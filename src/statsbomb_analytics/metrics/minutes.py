"""Minutes played, derived from `lineup_positions` stints.

Rule (documented assumptions):
- `from_time` / `to_time` are `mm:ss` on the cumulative match clock (period 2 starts at 45:00,
  extra time continues past 90:00), so a stint's duration is simply `to - from`.
- A null `to_time` means "until the final whistle". The match end is the latest event clock
  (`minute * 60 + second`) in periods 1-4; the penalty shootout (period 5) is excluded.
- Minutes include stoppage time and extra time. A player's minutes in a match are the sum of
  their stints (position changes split a stay into several stints, with no overlap).
- Players listed in a lineup without any stint (unused substitutes) get no row.
- Limitation: the match end is approximated by the last recorded event, so it can be a few
  seconds short of the real final whistle.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from statsbomb_analytics.storage.duckdb_store import query

MATCH_END_SQL = """
SELECT match_id, MAX(minute * 60 + second) AS match_end_seconds
FROM events
WHERE period <= 4
GROUP BY match_id
"""


def clock_to_seconds(value: object) -> float:
    """Convert a `mm:ss` string to seconds; null-like values become NaN."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return float("nan")
    minutes, seconds = str(value).split(":")[:2]
    return int(minutes) * 60 + int(seconds)


def stint_minutes(positions: pd.DataFrame, match_end: pd.DataFrame) -> pd.DataFrame:
    """Add `minutes` to each stint of `positions`.

    `match_end` has columns `match_id` and `match_end_seconds`.
    """
    df = positions.merge(match_end, on="match_id", how="left")
    start = df["from_time"].map(clock_to_seconds)
    end = df["to_time"].map(clock_to_seconds).fillna(df["match_end_seconds"])
    df["minutes"] = ((end - start).clip(lower=0) / 60.0).fillna(0.0)
    return df.drop(columns="match_end_seconds")


def player_match_minutes(positions: pd.DataFrame, match_end: pd.DataFrame) -> pd.DataFrame:
    """Minutes per (match, team, player)."""
    stints = stint_minutes(positions, match_end)
    return stints.groupby(["match_id", "team_id", "player_id"], as_index=False)["minutes"].sum()


def player_minutes(positions: pd.DataFrame, match_end: pd.DataFrame) -> pd.DataFrame:
    """Season totals per (team, player): `minutes` and `matches` (matches with > 0 minutes)."""
    per_match = player_match_minutes(positions, match_end)
    per_match = per_match[per_match["minutes"] > 0]
    return per_match.groupby(["team_id", "player_id"], as_index=False).agg(
        minutes=("minutes", "sum"), matches=("match_id", "nunique")
    )


def load_player_minutes(processed_dir: Path) -> pd.DataFrame:
    """Season totals per (team, player) straight from the processed data."""
    positions = query(processed_dir, "SELECT * FROM lineup_positions")
    match_end = query(processed_dir, MATCH_END_SQL)
    return player_minutes(positions, match_end)
