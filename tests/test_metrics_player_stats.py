from __future__ import annotations

import math

import pandas as pd

from statsbomb_analytics.metrics.player_stats import (
    add_per90,
    player_event_counts,
    position_group,
    primary_positions,
    team_xg,
)

EVENT_COLUMNS = [
    "player_id",
    "team_id",
    "type",
    "shot_xg",
    "shot_outcome",
    "shot_type",
    "pass_outcome",
    "pass_shot_assist",
    "pass_goal_assist",
    "dribble_outcome",
    "duel_type",
]


def _events(rows: list[dict]) -> pd.DataFrame:
    base = dict.fromkeys(EVENT_COLUMNS)
    return pd.DataFrame([{**base, **r} for r in rows], columns=EVENT_COLUMNS)


def test_position_group() -> None:
    assert position_group("Goalkeeper") == "GK"
    assert position_group("Left Wing Back") == "DEF"
    assert position_group("Center Attacking Midfield") == "MID"
    assert position_group("Right Wing") == "FWD"
    assert position_group("Nonexistent") is None
    assert position_group(None) is None


def test_event_counts_shots_xg_and_penalties() -> None:
    ev = _events(
        [
            {"player_id": 1, "team_id": 9, "type": "Shot", "shot_xg": 0.3, "shot_outcome": "Saved"},
            {
                "player_id": 1,
                "team_id": 9,
                "type": "Shot",
                "shot_xg": 0.76,
                "shot_outcome": "Goal",
                "shot_type": "Penalty",
            },
        ]
    )
    row = player_event_counts(ev).iloc[0]
    assert row["shots"] == 2 and row["goals"] == 1
    assert math.isclose(row["xg"], 1.06) and math.isclose(row["npxg"], 0.3)


def test_event_counts_passes_and_defensive_actions() -> None:
    ev = _events(
        [
            {"player_id": 1, "team_id": 9, "type": "Pass"},
            {"player_id": 1, "team_id": 9, "type": "Pass", "pass_outcome": "Incomplete"},
            {"player_id": 1, "team_id": 9, "type": "Pass", "pass_shot_assist": True},
            {"player_id": 1, "team_id": 9, "type": "Pass", "pass_goal_assist": True},
            {"player_id": 1, "team_id": 9, "type": "Dribble", "dribble_outcome": "Complete"},
            {"player_id": 1, "team_id": 9, "type": "Dribble", "dribble_outcome": "Incomplete"},
            {"player_id": 1, "team_id": 9, "type": "Duel", "duel_type": "Tackle"},
            {"player_id": 1, "team_id": 9, "type": "Duel", "duel_type": "Aerial Lost"},
            {"player_id": 1, "team_id": 9, "type": "Interception"},
            {"player_id": 1, "team_id": 9, "type": "Pressure"},
        ]
    )
    row = player_event_counts(ev).iloc[0]
    assert row["passes"] == 4 and row["passes_completed"] == 3
    assert row["key_passes"] == 1 and row["assists"] == 1
    assert row["dribbles"] == 2 and row["dribbles_completed"] == 1
    assert row["tackles"] == 1 and row["interceptions"] == 1 and row["pressures"] == 1


def test_event_counts_separates_players_and_teams() -> None:
    ev = _events(
        [
            {"player_id": 1, "team_id": 9, "type": "Pass"},
            {"player_id": 1, "team_id": 8, "type": "Pass"},
            {"player_id": 2, "team_id": 9, "type": "Pass"},
        ]
    )
    assert len(player_event_counts(ev)) == 3


def test_primary_position_is_most_minutes() -> None:
    positions = pd.DataFrame(
        {
            "match_id": [1, 1],
            "team_id": [9, 9],
            "player_id": [1, 1],
            "position": ["Left Wing", "Center Forward"],
            "from_time": ["00:00", "60:00"],
            "to_time": ["60:00", None],
        }
    )
    match_end = pd.DataFrame({"match_id": [1], "match_end_seconds": [90 * 60]})
    out = primary_positions(positions, match_end).iloc[0]
    assert out["primary_position"] == "Left Wing" and out["position_group"] == "FWD"


def test_per90_scales_and_handles_zero_minutes() -> None:
    df = pd.DataFrame({"minutes": [180.0, 0.0], "goals": [2, 1]})
    out = add_per90(df, ("goals",))
    assert math.isclose(out["goals_p90"].iloc[0], 1.0)
    assert math.isnan(out["goals_p90"].iloc[1])


def test_team_xg_for_and_against() -> None:
    shots = pd.DataFrame(
        {
            "match_id": [1, 1, 1],
            "team_id": [9, 9, 8],
            "shot_xg": [0.5, 0.25, 0.4],
        }
    )
    out = team_xg(shots).set_index("team_id")
    assert math.isclose(out.loc[9, "xg_for"], 0.75) and math.isclose(out.loc[9, "xg_against"], 0.4)
    assert math.isclose(out.loc[8, "xg_for"], 0.4) and math.isclose(out.loc[8, "xg_against"], 0.75)
