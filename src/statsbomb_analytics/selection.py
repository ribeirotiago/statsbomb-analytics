"""Pure helpers behind the app's cascading selectors (competition -> team -> match/player)."""

from __future__ import annotations

import pandas as pd


def team_names(matches: pd.DataFrame) -> list[str]:
    """Sorted unique team names (home or away) in `matches`."""
    names = pd.concat([matches["home_team"], matches["away_team"]])
    return sorted(names.dropna().unique().tolist())


def matches_with_team(matches: pd.DataFrame, team: str) -> pd.DataFrame:
    """Matches in which `team` played (home or away), keeping the original order."""
    mask = (matches["home_team"] == team) | (matches["away_team"] == team)
    return matches[mask].reset_index(drop=True)
