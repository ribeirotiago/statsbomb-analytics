from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statsbomb_analytics.metrics.player_metrics import add_completion_rates, team_competitions
from statsbomb_analytics.metrics.scouting import (
    CATEGORY_COLORS,
    METRICS,
    PCTILE_SUFFIX,
    add_percentiles,
    breakdown_table,
    format_p90,
    format_total,
    percentile_ranks,
    pizza_data,
)


def _pool() -> pd.DataFrame:
    n = 4
    data = {m.column: np.arange(1.0, n + 1) for m in METRICS}
    data |= {m.total: np.arange(10.0, 10.0 * n + 1, 10.0) for m in METRICS if m.total}
    data |= {
        "passes_completed": [8.0, 9, 10, 11],
        "passes": [10.0, 10, 10, 10],
        "dribbles_completed": [1.0, 2, 3, 4],
        "dribbles": [2.0, 4, 6, 8],
        "player_name": list("abcd"),
        "team_name": ["T"] * n,
        "competition_name": ["L"] * n,
        "primary_position": ["Center Back"] * n,
        "position_group": ["DEF", "DEF", "FWD", "FWD"],
        "minutes": [900.0] * n,
    }
    return pd.DataFrame(data)


def test_percentile_ranks_span_zero_to_hundred() -> None:
    pool = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
    ranks = percentile_ranks(pool, ["a"])["a"]
    assert ranks.iloc[-1] == pytest.approx(100.0)
    assert ranks.iloc[0] == pytest.approx(25.0)


def test_percentile_ranks_keep_nan() -> None:
    pool = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
    assert np.isnan(percentile_ranks(pool, ["a"])["a"].iloc[1])


def test_every_metric_has_a_known_category() -> None:
    assert {m.category for m in METRICS} <= set(CATEGORY_COLORS)
    assert all(m.total or m.ratio for m in METRICS)


def test_add_percentiles_global_vs_by_group() -> None:
    pool = _pool()
    col = f"goals_p90{PCTILE_SUFFIX}"
    overall = add_percentiles(pool)[col]
    by_group = add_percentiles(pool, by_group=True)[col]
    assert overall.iloc[-1] == pytest.approx(100.0)
    assert overall.iloc[0] == pytest.approx(25.0)
    # inside each 2-player group the best has 100 and the other 50
    assert by_group.tolist() == [50.0, 100.0, 50.0, 100.0]


def test_format_total_and_p90() -> None:
    row = add_percentiles(_pool()).iloc[0]
    by_col = {m.column: m for m in METRICS}
    assert format_total(row, by_col["goals_p90"]) == "10"
    assert format_total(row, by_col["pass_pct"]) == "8/10"
    assert format_p90(row, by_col["pass_pct"]) == "1.0%"
    assert format_p90(row, by_col["goals_p90"]) == "1.00"


def test_format_total_missing_values() -> None:
    metric = next(m for m in METRICS if m.column == "pass_pct")
    row = pd.Series({"passes_completed": np.nan, "passes": 3})
    assert format_total(row, metric) == "—"


def test_pizza_data_skips_metrics_without_percentile() -> None:
    pool = _pool()
    pool.loc[0, "dribble_pct"] = np.nan
    row = add_percentiles(pool).iloc[0]
    data = pizza_data(row)
    assert "Dribble %" not in data["params"]
    assert len(data["params"]) == len(METRICS) - 1
    assert len(data["values"]) == len(data["colors"]) == len(data["raw_totals"])
    assert all(isinstance(v, int) for v in data["values"])


def test_breakdown_table_matches_pizza() -> None:
    row = add_percentiles(_pool()).iloc[2]
    table = breakdown_table(row)
    assert list(table.columns) == ["Dimension", "Metric", "Total", "Per 90", "Percentile"]
    assert table["Metric"].tolist() == pizza_data(row)["params"]


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
