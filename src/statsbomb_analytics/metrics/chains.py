"""Possession-chain player valuation: xGChain and xGBuildup.

Definitions (StatsBomb `possession` id groups consecutive actions of one team):
- A possession's value is the sum of the xG of the possession team's non-penalty shots in it.
- A player *participates* in a possession if they have an on-ball action in it for the
  possession team (pass, carry, dribble or shot). Each player counts once per possession.
- **xGChain**: sum of the values of the possessions the player participated in.
- **xGBuildup**: like xGChain, but ignoring possessions where the player took a shot or made
  the shot-assisting pass (key pass), so it credits only the earlier build-up.

Assumptions and limitations:
- Credit is uniform: every participant gets the full possession value, regardless of the
  quality of their contribution. Players on teams that build long, shot-ending possessions
  are favoured, and forwards/creators dominate the ranking.
- Penalties are excluded from the possession value (the fouled player would otherwise
  receive the credit). Rebounds inside one possession add up.
- Players without chains (defenders who rarely touch shot-ending possessions) get zero.
- Per-90 rates are inflated by small minutes samples; the UI filters on minutes.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from statsbomb_analytics.metrics.minutes import load_player_minutes
from statsbomb_analytics.storage.duckdb_store import query

CHAIN_COLUMNS = ("xgchain", "xgbuildup")

CHAIN_EVENTS_SQL = """
SELECT match_id, possession, possession_team_id, team_id, player_id, type,
       shot_xg, shot_type, pass_shot_assist
FROM events
WHERE type IN ('Pass', 'Carry', 'Dribble', 'Shot')
  AND player_id IS NOT NULL
  AND team_id = possession_team_id
"""

_POSSESSION = ["match_id", "possession", "team_id"]


def possession_values(events: pd.DataFrame) -> pd.DataFrame:
    """Non-penalty xG per (match, possession, team) for possessions with shots."""
    shots = events[(events["type"] == "Shot") & (events["shot_type"] != "Penalty")]
    values = shots.groupby(_POSSESSION, as_index=False)["shot_xg"].sum()
    return values.rename(columns={"shot_xg": "chain_xg"})


def player_chains(events: pd.DataFrame) -> pd.DataFrame:
    """xGChain and xGBuildup per (team, player) from `CHAIN_EVENTS_SQL` events."""
    values = possession_values(events)
    key = _POSSESSION + ["player_id"]
    participants = events[key].drop_duplicates().merge(values, on=_POSSESSION)

    finishers = events[
        (events["type"] == "Shot") | (events["pass_shot_assist"].fillna(False).astype(bool))
    ][key].drop_duplicates()
    finishers["is_finisher"] = True
    participants = participants.merge(finishers, on=key, how="left")
    participants["is_finisher"] = participants["is_finisher"].fillna(False).astype(bool)
    participants["buildup_xg"] = participants["chain_xg"].where(~participants["is_finisher"], 0.0)

    return participants.groupby(["team_id", "player_id"], as_index=False).agg(
        xgchain=("chain_xg", "sum"), xgbuildup=("buildup_xg", "sum")
    )


def build_player_chains(processed_dir: Path) -> pd.DataFrame:
    """Players with minutes, xGChain, xGBuildup and per-90 values (no minutes threshold)."""
    table = load_player_minutes(processed_dir).merge(
        player_chains(query(processed_dir, CHAIN_EVENTS_SQL)),
        on=["team_id", "player_id"],
        how="left",
    )
    for col in CHAIN_COLUMNS:
        table[col] = table[col].fillna(0.0)
        table[f"{col}_p90"] = table[col] / table["minutes"].where(table["minutes"] > 0) * 90.0
    return table
