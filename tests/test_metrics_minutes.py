from __future__ import annotations

import math

import pandas as pd

from statsbomb_analytics.metrics.minutes import (
    clock_to_seconds,
    player_match_minutes,
    player_minutes,
    stint_minutes,
)

MATCH_END = pd.DataFrame({"match_id": [1, 2], "match_end_seconds": [95 * 60 + 10, 120 * 60]})


def _stints(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["match_id", "team_id", "player_id", "from_time", "to_time"])


def test_clock_to_seconds() -> None:
    assert clock_to_seconds("00:00") == 0
    assert clock_to_seconds("51:26") == 51 * 60 + 26
    assert clock_to_seconds("115:26") == 115 * 60 + 26
    assert math.isnan(clock_to_seconds(None))
    assert math.isnan(clock_to_seconds(float("nan")))


def test_full_match_uses_match_end_including_stoppage() -> None:
    out = stint_minutes(_stints([(1, 10, 100, "00:00", None)]), MATCH_END)
    assert out["minutes"].iloc[0] == (95 * 60 + 10) / 60


def test_substitution_second_half_cumulative_clock() -> None:
    # On at 51:26 (period 2 clock is cumulative), off at the final whistle.
    out = stint_minutes(_stints([(1, 10, 100, "51:26", None)]), MATCH_END)
    assert abs(out["minutes"].iloc[0] - ((95 * 60 + 10 - (51 * 60 + 26)) / 60)) < 1e-9


def test_extra_time_stint() -> None:
    out = stint_minutes(_stints([(2, 10, 100, "00:00", "115:26")]), MATCH_END)
    assert abs(out["minutes"].iloc[0] - (115 * 60 + 26) / 60) < 1e-9


def test_position_changes_are_summed_per_match() -> None:
    stints = _stints(
        [(1, 10, 100, "00:00", "62:25"), (1, 10, 100, "62:25", None)],
    )
    out = player_match_minutes(stints, MATCH_END)
    assert len(out) == 1
    assert abs(out["minutes"].iloc[0] - (95 * 60 + 10) / 60) < 1e-9


def test_season_totals_and_matches() -> None:
    stints = _stints(
        [
            (1, 10, 100, "00:00", "45:00"),
            (2, 10, 100, "00:00", "30:00"),
            (2, 10, 101, "30:00", "30:00"),  # zero-length stint: not counted as a match
        ]
    )
    out = player_minutes(stints, MATCH_END).set_index("player_id")
    assert out.loc[100, "minutes"] == 75
    assert out.loc[100, "matches"] == 2
    assert 101 not in out.index


def test_missing_match_end_gives_zero_not_nan() -> None:
    out = stint_minutes(_stints([(99, 10, 100, "10:00", None)]), MATCH_END)
    assert out["minutes"].iloc[0] == 0.0
