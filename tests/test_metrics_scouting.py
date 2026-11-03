from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statsbomb_analytics.metrics.player_metrics import add_completion_rates, team_competitions
from statsbomb_analytics.metrics.scouting import (
    RADAR_PROFILES,
    percentile_ranks,
    radar_percentiles,
)


def test_percentile_ranks_span_zero_to_hundred() -> None:
    pool = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
    ranks = percentile_ranks(pool, ["a"])["a"]
    assert ranks.iloc[-1] == pytest.approx(100.0)
    assert ranks.iloc[0] == pytest.approx(25.0)


def test_percentile_ranks_keep_nan() -> None:
    pool = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
    assert np.isnan(percentile_ranks(pool, ["a"])["a"].iloc[1])


def test_radar_percentiles_use_profile_labels() -> None:
    profile = RADAR_PROFILES["FWD"]
    pool = pd.DataFrame({col: [1.0, 2.0, 3.0] for col in profile})
    out = radar_percentiles(pool, "FWD")
    assert list(out.columns) == list(profile.values())
    assert out.iloc[2].eq(100.0).all()


def test_radar_percentiles_reject_goalkeepers() -> None:
    with pytest.raises(KeyError):
        radar_percentiles(pd.DataFrame({"a": [1.0]}), "GK")


def test_team_competitions_ignores_champions_league() -> None:
    matches = pd.DataFrame(
        {
            "home_team_id": [1, 1, 1],
            "away_team_id": [2, 2, 2],
            "competition_name": ["La Liga", "La Liga", "Champions League"],
        }
    )
    out = team_competitions(matches).set_index("team_id")["competition_name"]
    assert out.to_dict() == {1: "La Liga", 2: "La Liga"}


def test_team_competitions_omits_teams_only_in_champions_league() -> None:
    matches = pd.DataFrame(
        {"home_team_id": [9], "away_team_id": [8], "competition_name": ["Champions League"]}
    )
    assert team_competitions(matches).empty


def test_completion_rates_nan_without_attempts() -> None:
    table = pd.DataFrame(
        {
            "passes": [10, 0],
            "passes_completed": [8, 0],
            "dribbles": [0, 4],
            "dribbles_completed": [0, 1],
        }
    )
    out = add_completion_rates(table)
    assert out["pass_pct"].iloc[0] == pytest.approx(80.0)
    assert np.isnan(out["pass_pct"].iloc[1])
    assert np.isnan(out["dribble_pct"].iloc[0])
    assert out["dribble_pct"].iloc[1] == pytest.approx(25.0)
