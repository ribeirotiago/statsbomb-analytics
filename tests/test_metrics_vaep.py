from __future__ import annotations

import numpy as np
import pandas as pd

from statsbomb_analytics.metrics.vaep import (
    action_values,
    build_features,
    classifier_metrics,
    fit_vaep,
    label_actions,
    match_folds,
    player_vaep,
)


def _actions(rows: list[tuple]) -> pd.DataFrame:
    cols = ["match_id", "period", "team_id", "player_id", "type", "seconds", "x", "y"]
    df = pd.DataFrame(rows, columns=cols)
    df["end_x"] = df["x"] + 5
    df["end_y"] = df["y"]
    df["success"] = True
    df["is_goal"] = df["type"] == "Shot"
    return df


def test_label_actions_window_team_and_match_boundary() -> None:
    rows = [(1, 1, 10, 1, "Pass", i, 50, 40) for i in range(5)]
    rows.append((1, 1, 10, 2, "Shot", 5, 110, 40))  # team 10 scores in match 1
    rows += [(1, 1, 20, 3, "Pass", 6, 50, 40)]  # opponent action right after the goal
    rows += [(2, 1, 10, 1, "Pass", 0, 50, 40)]  # next match, no goals
    labelled = label_actions(_actions(rows), horizon=3)
    # Window of 3 includes the current action: rows 3, 4 and 5 (the shot) see the goal.
    assert labelled["scores"].tolist() == [0, 0, 0, 1, 1, 1, 0, 0]
    # The opponent (team 20) conceded in the window of its own action only if the goal is ahead.
    assert labelled["concedes"].tolist() == [0, 0, 0, 0, 0, 0, 0, 0]
    # From the opponent's point of view, a goal by team 10 in the window is a conceded goal.
    rows2 = [(1, 1, 20, 3, "Pass", 0, 50, 40), (1, 1, 10, 2, "Shot", 1, 110, 40)]
    assert label_actions(_actions(rows2), horizon=3)["concedes"].tolist() == [1, 0]


def test_build_features_mirrors_previous_team_and_masks_other_matches() -> None:
    df = _actions(
        [
            (1, 1, 10, 1, "Pass", 0, 100, 10),
            (1, 1, 20, 2, "Pass", 2, 30, 70),
            (2, 1, 10, 3, "Pass", 0, 60, 40),
        ]
    )
    f = build_features(df)
    assert f.loc[1, "x0_a1"] == 120 - 100 and f.loc[1, "y0_a1"] == 80 - 10
    assert f.loc[1, "same_team_a1"] == 0.0 and f.loc[1, "dt_a1"] == 2.0
    assert np.isnan(f.loc[2, "x0_a1"])  # previous action belongs to another match
    assert np.isnan(f.loc[0, "x0_a2"])
    assert f.loc[2, "dist_goal_start"] == 60.0


def test_match_folds_keep_matches_together() -> None:
    match_ids = np.repeat(np.arange(30), 4)
    folds = match_folds(match_ids, 5)
    for match in range(30):
        assert len(set(folds[match_ids == match])) == 1
    assert set(folds) == {0, 1, 2, 3, 4}


def test_action_values_offensive_defensive_and_resets() -> None:
    df = _actions(
        [
            (1, 1, 10, 1, "Pass", 0, 50, 40),
            (1, 1, 10, 2, "Pass", 1, 60, 40),
            (1, 1, 20, 3, "Pass", 2, 40, 40),  # turnover: team changes
            (1, 2, 20, 3, "Pass", 3, 40, 40),  # new period: previous state reset
        ]
    )
    p_score = np.array([0.02, 0.04, 0.03, 0.05])
    p_concede = np.array([0.01, 0.01, 0.06, 0.02])
    v = action_values(df, p_score, p_concede)
    assert np.isclose(v.loc[0, "vaep"], 0.02 - 0.0 - (0.01 - 0.0))
    assert np.isclose(v.loc[1, "offensive"], 0.04 - 0.02)
    assert np.isclose(v.loc[1, "defensive"], -(0.01 - 0.01))
    # Team change: previous state seen from team 20's side swaps scores and concedes.
    assert np.isclose(v.loc[2, "offensive"], 0.03 - 0.01)
    assert np.isclose(v.loc[2, "defensive"], -(0.06 - 0.04))
    assert np.isclose(v.loc[3, "offensive"], 0.05)  # period start: previous state is 0
    assert np.allclose(v["vaep"], v["offensive"] + v["defensive"])


def test_action_values_reset_after_goal() -> None:
    df = _actions(
        [
            (1, 1, 10, 1, "Shot", 0, 110, 40),  # goal
            (1, 1, 20, 2, "Pass", 1, 60, 40),  # kick-off
        ]
    )
    v = action_values(df, np.array([1.0, 0.01]), np.array([0.0, 0.01]))
    assert np.isclose(v.loc[1, "offensive"], 0.01)  # not 0.01 - P_concede(goal)


def test_classifier_metrics_and_player_vaep() -> None:
    y = np.array([0, 0, 0, 1])
    m = classifier_metrics(y, np.array([0.1, 0.1, 0.2, 0.9]))
    assert m["brier"] < m["brier_baseline"] and m["auc"] == 1.0
    values = pd.DataFrame(
        {
            "team_id": [1, 1, 1],
            "player_id": [5, 5, 6],
            "offensive": [0.1, 0.0, 0.2],
            "defensive": [0.0, 0.05, 0.0],
            "vaep": [0.1, 0.05, 0.2],
        }
    )
    out = player_vaep(values).set_index("player_id")
    assert np.isclose(out.loc[5, "vaep"], 0.15) and out.loc[5, "vaep_actions"] == 2


def test_fit_vaep_learns_shot_location_signal() -> None:
    rng = np.random.default_rng(1)
    n_matches, per_match = 30, 120
    n = n_matches * per_match
    x = rng.uniform(0, 120, n)
    team = rng.choice([10, 20], n)
    is_shot = rng.uniform(size=n) < 0.05
    goal = is_shot & (rng.uniform(size=n) < (x / 120) ** 2)
    df = pd.DataFrame(
        {
            "match_id": np.repeat(np.arange(n_matches), per_match),
            "period": 1,
            "team_id": team,
            "player_id": rng.integers(1, 8, n),
            "type": np.where(is_shot, "Shot", "Pass"),
            "seconds": np.tile(np.arange(per_match) * 5, n_matches),
            "x": x,
            "y": rng.uniform(0, 80, n),
            "end_x": x,
            "end_y": rng.uniform(0, 80, n),
            "success": goal,
            "is_goal": goal,
        }
    )
    values, metrics, calibration = fit_vaep(df, n_folds=3)
    assert len(values) == n and metrics["scores"]["auc"] > 0.6
    assert metrics["concedes"]["base_rate"] > 0
    assert set(calibration) == {"scores", "concedes"}
    assert np.allclose(values["vaep"], values["offensive"] + values["defensive"])
