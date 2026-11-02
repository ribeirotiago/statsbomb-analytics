from __future__ import annotations

import math

import numpy as np
import pandas as pd

from statsbomb_analytics.metrics.xt import (
    N_COLS,
    N_ROWS,
    action_xt,
    fit_xt,
    player_xt,
    ppda,
    zone_index,
)


def _moves(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=["team_id", "player_id", "type", "x", "y", "end_x", "end_y", "success"],
    )


def test_zone_index_corners_and_clipping() -> None:
    idx = zone_index(np.array([0.0, 119.9, 125.0, -1.0]), np.array([0.0, 79.9, 85.0, -1.0]))
    assert idx[0] == 0
    assert idx[1] == N_ROWS * N_COLS - 1
    assert idx[2] == N_ROWS * N_COLS - 1
    assert idx[3] == 0


def test_fit_xt_single_zone_shot_only() -> None:
    # One zone with only shots: xT equals the mean xG there; every other zone is zero.
    moves = pd.DataFrame({"x": [], "y": [], "end_x": [], "end_y": []})
    shots = pd.DataFrame({"x": [110.0, 111.0], "y": [40.0, 40.0], "shot_xg": [0.2, 0.4]})
    grid = fit_xt(moves, shots)
    assert grid.shape == (N_ROWS, N_COLS)
    z = zone_index(np.array([110.0]), np.array([40.0]))[0]
    assert math.isclose(grid.ravel()[z], 0.3)
    assert math.isclose(grid.sum(), 0.3)


def test_fit_xt_propagates_value_backwards() -> None:
    # Zone A only passes to zone B; B only shoots (xG 0.5). xT[A] = xT[B] = 0.5.
    a, b = (10.0, 40.0), (110.0, 40.0)
    moves = pd.DataFrame({"x": [a[0]], "y": [a[1]], "end_x": [b[0]], "end_y": [b[1]]})
    shots = pd.DataFrame({"x": [b[0]], "y": [b[1]], "shot_xg": [0.5]})
    grid = fit_xt(moves, shots).ravel()
    za = zone_index(np.array([a[0]]), np.array([a[1]]))[0]
    zb = zone_index(np.array([b[0]]), np.array([b[1]]))[0]
    assert math.isclose(grid[zb], 0.5) and math.isclose(grid[za], 0.5)


def test_fit_xt_failed_moves_lower_value() -> None:
    # Zone A: one success to B and one failure. P(move)=1, P(A->B)=0.5, so xT[A] = 0.25.
    a, b = (10.0, 40.0), (110.0, 40.0)
    moves = pd.DataFrame(
        {
            "x": [a[0], a[0]],
            "y": [a[1], a[1]],
            "end_x": [b[0], b[0]],
            "end_y": [b[1], b[1]],
            "success": [True, False],
        }
    )
    shots = pd.DataFrame({"x": [b[0]], "y": [b[1]], "shot_xg": [0.5]})
    grid = fit_xt(moves, shots).ravel()
    za = zone_index(np.array([a[0]]), np.array([a[1]]))[0]
    assert math.isclose(grid[za], 0.25)


def test_action_xt_only_for_successful_moves() -> None:
    grid = np.zeros((N_ROWS, N_COLS))
    grid[N_ROWS // 2, N_COLS - 1] = 0.4
    moves = _moves(
        [
            (1, 1, "Pass", 10.0, 40.0, 118.0, 42.0, True),
            (1, 1, "Pass", 10.0, 40.0, 118.0, 42.0, False),
        ]
    )
    out = action_xt(moves, grid)
    assert math.isclose(out.iloc[0], 0.4) and math.isnan(out.iloc[1])


def test_player_xt_splits_pass_and_carry() -> None:
    grid = np.zeros((N_ROWS, N_COLS))
    grid[N_ROWS // 2, N_COLS - 1] = 0.4
    moves = _moves(
        [
            (1, 1, "Pass", 10.0, 40.0, 118.0, 42.0, True),
            (1, 1, "Carry", 10.0, 40.0, 118.0, 42.0, True),
            (1, 2, "Pass", 10.0, 40.0, 118.0, 42.0, False),
        ]
    )
    out = player_xt(moves, grid).set_index("player_id")
    assert math.isclose(out.loc[1, "xt"], 0.8)
    assert math.isclose(out.loc[1, "xt_pass"], 0.4) and math.isclose(out.loc[1, "xt_carry"], 0.4)
    assert 2 not in out.index


def test_ppda_zones_and_opponent_passes() -> None:
    rows = [
        # Team 1 passes: 3 in own 60% (x <= 72), 1 beyond (ignored).
        (1, 1, "Pass", None, 20.0),
        (1, 1, "Pass", None, 50.0),
        (1, 1, "Pass", None, 70.0),
        (1, 1, "Pass", None, 100.0),
        # Team 2 defensive actions: 2 in the pressing zone (x >= 48), 1 too deep (ignored).
        (1, 2, "Interception", None, 60.0),
        (1, 2, "Duel", "Tackle", 90.0),
        (1, 2, "Foul Committed", None, 20.0),
        # Team 2 passes: 2 counted for team 1's PPDA.
        (1, 2, "Pass", None, 30.0),
        (1, 2, "Pass", None, 40.0),
        # Team 1 defensive action: 1.
        (1, 1, "Dribbled Past", None, 55.0),
    ]
    events = pd.DataFrame(rows, columns=["match_id", "team_id", "type", "duel_type", "x"])
    out = ppda(events).set_index("team_id")
    assert math.isclose(out.loc[2, "ppda"], 3 / 2)  # allowed 3 passes / 2 actions
    assert math.isclose(out.loc[1, "ppda"], 2 / 1)  # allowed 2 passes / 1 action


def test_ppda_no_defensive_actions_is_nan() -> None:
    events = pd.DataFrame(
        [(1, 1, "Pass", None, 10.0), (1, 2, "Pass", None, 10.0)],
        columns=["match_id", "team_id", "type", "duel_type", "x"],
    )
    assert ppda(events)["ppda"].isna().all()
