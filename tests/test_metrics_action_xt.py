from __future__ import annotations

import numpy as np
import pandas as pd

from statsbomb_analytics.metrics.action_xt import (
    build_features,
    calibration_table,
    fit_action_xt,
    label_actions,
    player_action_xt,
    score_actions,
    split_by_match,
)

COLUMNS = [
    "match_id",
    "possession",
    "team_id",
    "player_id",
    "type",
    "x",
    "y",
    "end_x",
    "end_y",
    "success",
    "shot_xg",
    "shot_type",
]


def _events() -> pd.DataFrame:
    rows = [
        # Possession 1 ends in a 0.3 xG shot, possession 2 has no shot, possession 3 penalty.
        (1, 1, 9, 1, "Pass", 60, 40, 100, 40, True, None, None),
        (1, 1, 9, 2, "Carry", 100, 40, 110, 40, True, None, None),
        (1, 1, 9, 3, "Shot", 110, 40, None, None, True, 0.3, "Open Play"),
        (1, 2, 9, 1, "Pass", 30, 10, 40, 20, False, None, None),
        (1, 3, 9, 1, "Pass", 90, 40, 105, 40, True, None, None),
        (1, 3, 9, 4, "Shot", 108, 40, None, None, True, 0.76, "Penalty"),
    ]
    return pd.DataFrame(rows, columns=COLUMNS)


def test_label_actions_marks_shot_ending_possessions_and_skips_penalties() -> None:
    actions = label_actions(_events())
    assert len(actions) == 4  # shots are not actions
    by_poss = actions.groupby("possession")[["shot_end", "chain_xg"]].first()
    assert by_poss.loc[1, "shot_end"] == 1 and by_poss.loc[1, "chain_xg"] == 0.3
    assert by_poss.loc[2, "shot_end"] == 0
    assert by_poss.loc[3, "shot_end"] == 0  # penalty-only possession has no value


def test_build_features_scaled_and_symmetric() -> None:
    df = pd.DataFrame({"x": [0, 120], "y": [0, 40], "end_x": [60, 120], "end_y": [80, 40]})
    f = build_features(df)
    assert f["x0"].tolist() == [0.0, 1.0] and f["x1"].tolist() == [0.5, 1.0]
    assert f["c0"].tolist() == [1.0, 0.0] and f["c1"].tolist() == [1.0, 0.0]


def test_split_by_match_has_no_overlap() -> None:
    actions = pd.DataFrame({"match_id": np.repeat(np.arange(20), 5)})
    train, test = split_by_match(actions, test_fraction=0.2)
    assert set(actions.loc[train, "match_id"]).isdisjoint(actions.loc[test, "match_id"])
    assert len(train) + len(test) == len(actions)
    assert actions.loc[test, "match_id"].nunique() == 4


def _synthetic_actions(n_matches: int = 60, per_match: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = n_matches * per_match
    x = rng.uniform(0, 120, n)
    end_x = np.clip(x + rng.uniform(0, 15, n), 0, 120)
    shot_end = (rng.uniform(size=n) < (end_x / 120) ** 3).astype(int)
    return pd.DataFrame(
        {
            "match_id": np.repeat(np.arange(n_matches), per_match),
            "team_id": 1,
            "player_id": rng.integers(1, 6, n),
            "type": "Pass",
            "x": x,
            "y": rng.uniform(0, 80, n),
            "end_x": end_x,
            "end_y": rng.uniform(0, 80, n),
            "success": True,
            "shot_end": shot_end,
            "chain_xg": shot_end * (0.02 + 0.2 * end_x / 120),
        }
    )


def test_fit_and_score_learn_threat_gradient() -> None:
    actions = _synthetic_actions()
    model = fit_action_xt(actions)
    assert model.metrics["auc"] > 0.6
    assert model.metrics["brier"] <= model.metrics["brier_baseline"]
    scored = score_actions(actions, model)
    assert (scored["axt"] >= 0).all()
    deep = scored[scored["end_x"] < 30]["axt"].mean()
    advanced = scored[scored["end_x"] > 90]["axt"].mean()
    assert advanced > deep


def test_player_action_xt_sums_successful_actions_only() -> None:
    scored = pd.DataFrame(
        {
            "team_id": [1, 1, 1],
            "player_id": [7, 7, 7],
            "type": ["Pass", "Carry", "Pass"],
            "success": [True, True, False],
            "axt": [0.1, 0.2, 0.5],
        }
    )
    row = player_action_xt(scored).iloc[0]
    assert np.isclose(row["axt"], 0.3)
    assert np.isclose(row["axt_pass"], 0.1) and np.isclose(row["axt_carry"], 0.2)


def test_calibration_table_matches_means() -> None:
    p = np.linspace(0.01, 0.99, 100)
    table = calibration_table(np.round(p), p, bins=4)
    assert len(table) == 4
    assert table["predicted"].is_monotonic_increasing
