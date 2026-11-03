from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statsbomb_analytics.metrics.comparison import (
    filter_minutes,
    mean_percentile_by_group,
    outfield_players,
    rank_shifts,
    spearman_by_group,
    spearman_matrix,
    top_players,
)

METHODS = {"A": "a_p90", "B": "b_p90"}


def _table() -> pd.DataFrame:
    n = 30
    a = np.arange(n, dtype=float)
    return pd.DataFrame(
        {
            "player_name": [f"p{i}" for i in range(n)],
            "primary_position": ["Center Back"] * n,
            "position_group": ["DEF"] * 15 + ["FWD"] * 15,
            "minutes": [100.0] * 5 + [900.0] * (n - 5),
            "a_p90": a,
            "b_p90": a**3,  # monotonic transform: same ranks
        }
    )


def test_spearman_is_one_for_monotonic_transform() -> None:
    corr = spearman_matrix(_table(), METHODS)
    assert corr.loc["A", "B"] == pytest.approx(1.0)


def test_spearman_is_minus_one_for_reversed_ranks() -> None:
    table = _table().assign(b_p90=lambda d: -d["a_p90"])
    assert spearman_matrix(table, METHODS).loc["A", "B"] == pytest.approx(-1.0)


def test_spearman_by_group_skips_small_groups() -> None:
    table = _table()
    out = spearman_by_group(table, methods=METHODS, min_size=20)
    assert out.empty
    out = spearman_by_group(table, methods=METHODS, min_size=10)
    assert set(out["position_group"]) == {"DEF", "FWD"}
    assert out["spearman"].eq(1.0).all()


def test_filter_minutes_applies_threshold() -> None:
    assert len(filter_minutes(_table(), 300.0)) == 25


def test_top_players_orders_by_value() -> None:
    top = top_players(_table(), METHODS, n=3)
    assert list(top["A"]["player_name"]) == ["p29", "p28", "p27"]


def test_rank_shifts_finds_largest_gap() -> None:
    table = _table().assign(b_p90=lambda d: -d["a_p90"])
    out = rank_shifts(table, "A", "B", METHODS, n=1)
    assert out.loc[0, "player_name"] == "p29"
    assert out.loc[0, "gap"] > 0.9


def test_mean_percentile_by_group_favours_high_group() -> None:
    out = mean_percentile_by_group(_table(), methods=METHODS)
    assert out.loc["FWD", "A"] > out.loc["DEF", "A"]


def test_outfield_players_drops_goalkeepers_and_unknown() -> None:
    table = _table().iloc[:4].assign(position_group=["GK", "DEF", None, "FWD"])
    out = outfield_players(table)
    assert list(out["position_group"]) == ["DEF", "FWD"]
