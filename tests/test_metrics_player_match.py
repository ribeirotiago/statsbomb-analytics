from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statsbomb_analytics.metrics.player_match import (
    match_event_counts,
    match_player_detailed_stats,
    match_positions,
    match_top_contributors,
    normalize_player_match_log,
    player_match_action_xt,
    player_match_chains,
    player_match_vaep,
    player_match_xt,
    score_match_spatial_events,
)
from statsbomb_analytics.metrics.xt import N_COLS, N_ROWS


def test_match_positions_picks_most_minutes_per_match() -> None:
    positions = pd.DataFrame(
        {
            "match_id": [1, 1],
            "team_id": [10, 10],
            "player_id": [100, 100],
            "position": ["Left Wing", "Center Forward"],
            "from_time": ["00:00", "30:00"],
            "to_time": ["30:00", None],
        }
    )
    match_end = pd.DataFrame({"match_id": [1], "match_end_seconds": [90 * 60]})
    out = match_positions(positions, match_end)
    assert len(out) == 1
    assert out.loc[0, "position"] == "Center Forward"
    assert out.loc[0, "position_group"] == "FWD"


def test_match_event_counts_handles_empty_and_populated() -> None:
    empty = match_event_counts(pd.DataFrame())
    assert empty.empty
    assert "goals" in empty.columns

    events = pd.DataFrame(
        {
            "match_id": [1, 1, 1],
            "team_id": [10, 10, 10],
            "player_id": [100, 100, 100],
            "type": ["Shot", "Shot", "Pass"],
            "shot_xg": [0.3, 0.76, np.nan],
            "shot_outcome": ["Goal", "Saved", None],
            "shot_type": ["Open Play", "Penalty", None],
            "pass_outcome": [None, None, None],
            "pass_shot_assist": [None, None, True],
            "pass_goal_assist": [None, None, True],
        }
    )
    out = match_event_counts(events)
    row = out.iloc[0]
    assert row["goals"] == 1
    assert row["shots"] == 2
    assert row["xg"] == pytest.approx(1.06)
    assert row["npxg"] == pytest.approx(0.3)
    assert row["assists"] == 1
    assert row["key_passes"] == 1
    assert row["passes"] == 1
    assert row["passes_completed"] == 1


def test_player_match_xt_and_action_xt_and_vaep() -> None:
    grid = np.linspace(0.0, 0.24, N_ROWS * N_COLS).reshape(N_ROWS, N_COLS)
    moves = pd.DataFrame(
        {
            "match_id": [1, 1],
            "team_id": [10, 10],
            "player_id": [100, 100],
            "type": ["Pass", "Carry"],
            "x": [30.0, 60.0],
            "y": [40.0, 40.0],
            "end_x": [80.0, 100.0],
            "end_y": [40.0, 40.0],
            "success": [True, True],
        }
    )
    xt_df = player_match_xt(moves, grid)
    assert len(xt_df) == 1
    assert xt_df.loc[0, "xt"] == pytest.approx(xt_df.loc[0, "xt_pass"] + xt_df.loc[0, "xt_carry"])

    scored = moves.assign(axt=[0.02, 0.03])
    axt_df = player_match_action_xt(scored)
    assert axt_df.loc[0, "axt"] == pytest.approx(0.05)
    assert axt_df.loc[0, "axt_pass"] == pytest.approx(0.02)
    assert axt_df.loc[0, "axt_carry"] == pytest.approx(0.03)

    vaep_vals = pd.DataFrame(
        {
            "match_id": [1, 1],
            "team_id": [10, 10],
            "player_id": [100, 100],
            "offensive": [0.1, 0.05],
            "defensive": [-0.01, 0.02],
            "vaep": [0.09, 0.07],
        }
    )
    v_df = player_match_vaep(vaep_vals)
    assert v_df.loc[0, "vaep"] == pytest.approx(0.16)
    assert v_df.loc[0, "vaep_offensive"] == pytest.approx(0.15)
    assert v_df.loc[0, "vaep_defensive"] == pytest.approx(0.01)


def test_player_match_chains_separates_buildup_and_finisher() -> None:
    events = pd.DataFrame(
        {
            "match_id": [1, 1, 1],
            "possession": [5, 5, 5],
            "possession_team_id": [10, 10, 10],
            "team_id": [10, 10, 10],
            "player_id": [101, 102, 103],
            "type": ["Pass", "Pass", "Shot"],
            "shot_xg": [np.nan, np.nan, 0.25],
            "shot_type": [None, None, "Open Play"],
            "pass_shot_assist": [False, True, None],
        }
    )
    chains = player_match_chains(events).set_index("player_id")
    assert chains.loc[101, "xgchain"] == pytest.approx(0.25)
    assert chains.loc[101, "xgbuildup"] == pytest.approx(0.25)
    assert chains.loc[102, "xgchain"] == pytest.approx(0.25)
    assert chains.loc[102, "xgbuildup"] == pytest.approx(0.0)
    assert chains.loc[103, "xgbuildup"] == pytest.approx(0.0)


def test_match_top_contributors_returns_top_n_per_team() -> None:
    df = pd.DataFrame(
        {
            "team_id": [1, 1, 1, 2, 2],
            "player_id": [11, 12, 13, 21, 22],
            "player_name": ["A", "B", "C", "D", "E"],
            "position": ["MID"] * 5,
            "minutes": [90.0] * 5,
            "vaep": [0.1, 0.5, 0.3, 0.2, 0.4],
        }
    )
    top = match_top_contributors(df, "vaep", n=2)
    assert top[1]["player_name"].tolist() == ["B", "C"]
    assert top[2]["player_name"].tolist() == ["E", "D"]


def test_match_player_detailed_stats_and_spatial_events() -> None:
    grid = np.zeros((N_ROWS, N_COLS))
    grid[:, -1] = 0.20  # high value near opponent goal (x near 120)
    grid[:, 0] = 0.01  # low value near own goal (x near 0)

    events = pd.DataFrame(
        {
            "player_id": [100, 100, 100, 100, 100],
            "period": [1, 1, 1, 2, 2],
            "minute": [10, 15, 22, 55, 70],
            "type": ["Pass", "Pass", "Carry", "Duel", "Shot"],
            "x": [60.0, 20.0, 70.0, 30.0, 108.0],
            "y": [40.0, 40.0, 40.0, 40.0, 40.0],
            "pass_end_x": [115.0, 35.0, np.nan, np.nan, np.nan],
            "pass_end_y": [40.0, 40.0, np.nan, np.nan, np.nan],
            "carry_end_x": [np.nan, np.nan, 115.0, np.nan, np.nan],
            "carry_end_y": [np.nan, np.nan, 40.0, np.nan, np.nan],
            "pass_outcome": [None, "Incomplete", None, None, None],
            "pass_shot_assist": [True, False, None, None, None],
            "pass_goal_assist": [True, False, None, None, None],
            "pass_cross": [True, False, None, None, None],
            "pass_switch": [False, False, None, None, None],
            "pass_through_ball": [True, False, None, None, None],
            "shot_xg": [np.nan, np.nan, np.nan, np.nan, 0.45],
            "shot_outcome": [None, None, None, None, "Goal"],
            "shot_type": [None, None, None, None, "Open Play"],
            "shot_body_part": [None, None, None, None, "Right Foot"],
            "dribble_outcome": [None, None, None, None, None],
            "duel_type": [None, None, None, "Tackle", None],
            "duel_outcome": [None, None, None, "Won", None],
            "ball_recovery_failure": [None, None, None, None, None],
            "counterpress": [None, None, None, True, None],
            "under_pressure": [True, False, True, False, False],
        }
    )

    stats = match_player_detailed_stats(events, 100)
    assert stats["attack"]["goals"] == 1
    assert stats["attack"]["xg"] == pytest.approx(0.45)
    assert stats["passes"]["passes_completed"] == 1
    assert stats["passes"]["passes"] == 2
    assert stats["passes"]["progressive_passes"] == 1
    assert stats["defense"]["tackles_won"] == 1
    assert stats["defense"]["counterpressures"] == 1
    assert stats["progression"]["carries"] == 1
    assert stats["progression"]["progressive_carries"] == 1
    assert stats["progression"]["carry_distance"] == pytest.approx(45.0)

    spatial = score_match_spatial_events(events, 100, grid=grid)
    moves = spatial["moves"]
    assert len(moves) == 3
    # First pass goes to x=115 (col 15, value 0.20) from x=60 (value 0.0) -> positive gain
    assert moves.loc[0, "xt_value"] == pytest.approx(0.20)
    # Second pass is incomplete at x=20 -> mirrored opp_x=100, non-positive turnover cost
    assert moves.loc[1, "xt_value"] <= 0.0
    assert len(spatial["defensive"]) == 1
    assert spatial["defensive"].loc[0, "event_subtype"] == "tackle"
    assert len(spatial["shots"]) == 1
    assert spatial["shots"].loc[0, "event_subtype"] == "goal"
    assert len(spatial["touches"]) == 5


def test_normalize_player_match_log_adds_percentiles_and_zscores() -> None:
    df = pd.DataFrame(
        {
            "match_id": [1, 2, 3],
            "xt": [0.1, 0.2, 0.3],
            "xgchain": [0.5, 0.5, 0.5],
        }
    )
    norm = normalize_player_match_log(df, methods={"Singh xT": "xt", "xGChain": "xgchain"})
    assert norm["xt_pct"].iloc[-1] == pytest.approx(100.0)
    assert norm["xt_z"].iloc[1] == pytest.approx(0.0)
    # Constant series gets 0.0 z-score instead of NaN
    assert (norm["xgchain_z"] == 0.0).all()
