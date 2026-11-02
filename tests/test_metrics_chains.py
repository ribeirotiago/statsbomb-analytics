from __future__ import annotations

import math

import pandas as pd

from statsbomb_analytics.metrics.chains import player_chains, possession_values

COLUMNS = [
    "match_id",
    "possession",
    "team_id",
    "player_id",
    "type",
    "shot_xg",
    "shot_type",
    "pass_shot_assist",
]


def _events(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=COLUMNS)


def _chain() -> pd.DataFrame:
    # Possession 1 (team 9): player 1 builds up, player 2 key pass, player 3 shoots (xG 0.3).
    # Possession 2 (team 9): player 1 only, no shot (value 0).
    return _events(
        [
            (1, 1, 9, 1, "Pass", None, None, None),
            (1, 1, 9, 1, "Carry", None, None, None),
            (1, 1, 9, 2, "Pass", None, None, True),
            (1, 1, 9, 3, "Shot", 0.3, "Open Play", None),
            (1, 2, 9, 1, "Pass", None, None, None),
        ]
    )


def test_possession_values_sum_shots_and_skip_penalties() -> None:
    ev = _events(
        [
            (1, 1, 9, 3, "Shot", 0.3, "Open Play", None),
            (1, 1, 9, 3, "Shot", 0.1, "Open Play", None),
            (1, 2, 9, 3, "Shot", 0.76, "Penalty", None),
        ]
    )
    out = possession_values(ev)
    assert len(out) == 1 and math.isclose(out["chain_xg"].iloc[0], 0.4)


def test_xgchain_gives_full_value_once_per_possession() -> None:
    out = player_chains(_chain()).set_index("player_id")
    assert math.isclose(out.loc[1, "xgchain"], 0.3)  # two actions, one possession, counted once
    assert math.isclose(out.loc[2, "xgchain"], 0.3)
    assert math.isclose(out.loc[3, "xgchain"], 0.3)


def test_xgbuildup_excludes_shooter_and_key_passer() -> None:
    out = player_chains(_chain()).set_index("player_id")
    assert math.isclose(out.loc[1, "xgbuildup"], 0.3)
    assert math.isclose(out.loc[2, "xgbuildup"], 0.0)
    assert math.isclose(out.loc[3, "xgbuildup"], 0.0)


def test_possessions_without_shots_give_no_value() -> None:
    ev = _events([(1, 1, 9, 1, "Pass", None, None, None)])
    assert player_chains(ev).empty


def test_values_accumulate_across_matches() -> None:
    ev = pd.concat(
        [
            _chain(),
            _events(
                [
                    (2, 1, 9, 1, "Pass", None, None, None),
                    (2, 1, 9, 3, "Shot", 0.2, "Open Play", None),
                ]
            ),
        ]
    )
    out = player_chains(ev).set_index("player_id")
    assert math.isclose(out.loc[1, "xgchain"], 0.5)
