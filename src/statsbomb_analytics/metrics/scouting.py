"""Percentile profiles for the scouting radar.

Assumptions and limitations:
- Percentiles are computed inside a comparison pool chosen by the caller (typically players of
  the same position group above the minutes threshold, in one league or all five).
- Percentiles are rank-based (0-100): they say where a player sits, not by how much.
- Per-90 rates reflect team style (possession teams accumulate more passes and xT).
- Goalkeepers have no radar profile (event data barely describes their work).
"""

from __future__ import annotations

import pandas as pd

RADAR_PROFILES: dict[str, dict[str, str]] = {
    "FWD": {
        "npxg_p90": "npxG",
        "shots_p90": "Shots",
        "key_passes_p90": "Key passes",
        "dribbles_completed_p90": "Dribbles",
        "xt_p90": "xT (Singh)",
        "vaep_p90": "VAEP",
    },
    "MID": {
        "key_passes_p90": "Key passes",
        "passes_p90": "Passes",
        "pass_pct": "Pass %",
        "axt_p90": "Action xT",
        "xgbuildup_p90": "xGBuildup",
        "tackles_p90": "Tackles",
        "ball_recoveries_p90": "Recoveries",
    },
    "DEF": {
        "tackles_p90": "Tackles",
        "interceptions_p90": "Interceptions",
        "clearances_p90": "Clearances",
        "passes_p90": "Passes",
        "pass_pct": "Pass %",
        "xgbuildup_p90": "xGBuildup",
        "xt_p90": "xT (Singh)",
    },
}


def percentile_ranks(pool: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Percentile (0-100) of each row on each column inside `pool`; NaN stays NaN."""
    return pool[columns].rank(pct=True) * 100.0


def radar_percentiles(pool: pd.DataFrame, position_group: str) -> pd.DataFrame:
    """Radar percentiles for the profile of `position_group`, indexed like `pool`.

    Columns are renamed to the human labels of the profile. Unknown groups (e.g. GK) raise
    `KeyError`.
    """
    profile = RADAR_PROFILES[position_group]
    ranks = percentile_ranks(pool, list(profile))
    return ranks.rename(columns=profile)
