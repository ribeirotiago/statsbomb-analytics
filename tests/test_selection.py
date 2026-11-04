from __future__ import annotations

import pandas as pd

from statsbomb_analytics.selection import matches_with_team, team_names


def _matches() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "match_id": [1, 2, 3],
            "home_team": ["A", "B", "C"],
            "away_team": ["B", "C", "A"],
        }
    )


def test_team_names_sorted_unique() -> None:
    assert team_names(_matches()) == ["A", "B", "C"]


def test_matches_with_team_includes_home_and_away() -> None:
    assert matches_with_team(_matches(), "A")["match_id"].tolist() == [1, 3]


def test_matches_with_team_unknown_team_is_empty() -> None:
    assert matches_with_team(_matches(), "Z").empty
