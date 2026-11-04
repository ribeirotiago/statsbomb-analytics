from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statsbomb_analytics.metrics.scouting import METRICS, add_percentiles
from statsbomb_analytics.metrics.similarity import (
    ARCHETYPES,
    SIMILARITY_FEATURES,
    build_feature_matrix,
    ensure_percentiles,
    evaluate_kmeans_range,
    find_similar_players,
    fit_kmeans_clusters,
    fit_pca_profile,
    metric_deltas,
    player_archetype_profile,
    score_archetypes,
)


def _synthetic_pool() -> pd.DataFrame:
    """Create a synthetic 12-player pool with distinct FWD, MID and DEF profiles."""
    rng = np.random.default_rng(42)
    rows: list[dict[str, object]] = []
    groups = ["FWD"] * 4 + ["MID"] * 4 + ["DEF"] * 4
    positions = ["Center Forward"] * 4 + ["Center Midfield"] * 4 + ["Center Back"] * 4
    for idx in range(12):
        grp = groups[idx]
        base: dict[str, object] = {
            "player_id": 100 + idx,
            "team_id": 10 + (idx % 3),
            "player_name": f"Player_{idx:02d}",
            "team_name": f"Team_{idx % 3}",
            "competition_name": "La Liga",
            "primary_position": positions[idx],
            "position_group": grp,
            "matches": 20,
            "minutes": 1800.0 - idx * 20.0,
        }
        for m in METRICS:
            val = float(rng.uniform(0.1, 0.5))
            if grp == "FWD" and m.category == "Attack":
                val += 2.0 + 0.1 * (idx % 4)
            elif grp == "MID" and m.category in ("Creation", "Possession"):
                val += 2.5 + 0.1 * (idx % 4)
            elif grp == "DEF" and m.category == "Defense":
                val += 2.5 + 0.1 * (idx % 4)
            if m.unit == "%":
                val = min(95.0, 55.0 + val * 10.0)
            base[m.column] = val
            if m.total:
                base[m.total] = round(val * 20.0, 1)
        base["passes"] = 800
        base["passes_completed"] = 680
        base["dribbles"] = 40
        base["dribbles_completed"] = 24
        rows.append(base)
    return pd.DataFrame(rows)


def test_archetype_specs_are_valid_and_normalised() -> None:
    feature_set = set(SIMILARITY_FEATURES)
    for spec in ARCHETYPES:
        assert set(spec.weights) <= feature_set
        assert sum(spec.weights.values()) == pytest.approx(1.0)
        assert len(spec.target_groups) >= 1


def test_build_feature_matrix_percentiles_and_zscore() -> None:
    pool = _synthetic_pool()
    ranked_pct, mat_pct = build_feature_matrix(pool, representation="percentiles")
    assert mat_pct.shape == (len(pool), len(SIMILARITY_FEATURES))
    assert np.all((mat_pct >= 0.0) & (mat_pct <= 100.0))
    assert "goals_p90_pctile" in ranked_pct.columns

    _, mat_z = build_feature_matrix(pool, representation="zscore")
    assert mat_z.shape == (len(pool), len(SIMILARITY_FEATURES))
    assert np.allclose(mat_z.mean(axis=0), 0.0, atol=1e-7)

    # Single-row edge case returns zeros for zscore
    _, single_z = build_feature_matrix(pool.iloc[:1], representation="zscore")
    assert np.all(single_z == 0.0)

    with pytest.raises(ValueError, match="Unsupported feature representation"):
        build_feature_matrix(pool, representation="invalid")


def test_find_similar_players_identical_twin_has_100_similarity() -> None:
    pool = _synthetic_pool()
    twin = pool.iloc[[0]].copy()
    twin["player_id"] = 999
    twin["team_id"] = 99
    twin["player_name"] = "Twin_Player"
    augmented = pd.concat([pool, twin], ignore_index=True)

    for metric in ("euclidean", "cosine"):
        for rep in ("percentiles", "zscore"):
            sim = find_similar_players(
                augmented,
                target_player_id=100,
                target_team_id=10,
                top_k=5,
                metric=metric,
                representation=rep,
                by_group=True,
                same_position_group=True,
            )
            assert len(sim) == 4  # 4 other FWDs (3 original + 1 twin)
            assert int(sim.iloc[0]["player_id"]) == 999
            assert float(sim.iloc[0]["distance"]) == pytest.approx(0.0, abs=1e-6)
            assert float(sim.iloc[0]["similarity_pct"]) == pytest.approx(100.0, abs=1e-5)


def test_find_similar_players_filters_and_errors() -> None:
    pool = _synthetic_pool()
    # Exclude same team
    sim_no_team = find_similar_players(
        pool,
        target_player_id=100,
        target_team_id=10,
        top_k=10,
        same_position_group=False,
        include_same_team=False,
    )
    assert (sim_no_team["team_id"] != 10).all()

    with pytest.raises(ValueError, match="not found in the pool"):
        find_similar_players(pool, target_player_id=99999, target_team_id=10)

    with pytest.raises(ValueError, match="Unsupported distance metric"):
        find_similar_players(pool, target_player_id=100, target_team_id=10, metric="manhattan")


def test_metric_deltas_sorted_by_abs_percentile_diff() -> None:
    ranked = add_percentiles(_synthetic_pool(), by_group=False)
    deltas = metric_deltas(ranked.iloc[0], ranked.iloc[1])
    assert len(deltas) == len(SIMILARITY_FEATURES)
    diffs = deltas["abs_pctile_diff"].to_numpy()
    assert np.all(diffs[:-1] <= diffs[1:])
    assert np.allclose(
        deltas["pctile_diff"],
        deltas["candidate_pctile"] - deltas["target_pctile"],
        atol=0.11,
    )


def test_score_archetypes_assigns_expected_domain_roles() -> None:
    pool = _synthetic_pool()
    scored = score_archetypes(pool, by_group=False)
    assert "archetype" in scored.columns
    assert "secondary_archetype" in scored.columns

    # FWDs have boosted Attack metrics -> Poacher & Box Finisher
    fwd_roles = set(scored[scored["position_group"] == "FWD"]["archetype"])
    assert "Poacher & Box Finisher" in fwd_roles

    # DEFs have boosted Defense metrics -> Ball-Winning Destroyer or Defender & Stopper
    def_roles = set(scored[scored["position_group"] == "DEF"]["archetype"])
    assert def_roles <= {"Ball-Winning Destroyer", "Ball-Playing Defender & Stopper"}

    profile = player_archetype_profile(scored.iloc[0])
    assert len(profile) == len(ARCHETYPES)
    assert profile.iloc[0]["Archetype"] == scored.iloc[0]["archetype"]
    assert list(profile.columns) == ["Archetype", "Target Groups", "Score", "Description"]


def test_ensure_percentiles_idempotent() -> None:
    pool = _synthetic_pool()
    once = ensure_percentiles(pool)
    twice = ensure_percentiles(once)
    assert once.equals(twice)


def test_fit_pca_profile_shapes_and_variance() -> None:
    pool = _synthetic_pool()
    res = fit_pca_profile(pool, representation="zscore", n_components=3)
    assert len(res.projection) == len(pool)
    assert {"pc1", "pc2", "pc3"} <= set(res.projection.columns)
    assert len(res.explained_variance_ratio) == 3
    assert 0.0 < sum(res.explained_variance_ratio) <= 1.0 + 1e-6
    assert res.explained_variance_ratio[0] >= res.explained_variance_ratio[1]
    assert len(res.loadings) == len(SIMILARITY_FEATURES)


def test_evaluate_and_fit_kmeans_clusters_recover_positional_structure() -> None:
    pool = _synthetic_pool()
    diag = evaluate_kmeans_range(pool, k_values=(2, 3, 4), representation="zscore")
    assert list(diag["k"]) == [2, 3, 4]
    assert (diag["silhouette"] > 0.0).all()
    # With 3 clear synthetic groups (FWD, MID, DEF), k=3 should achieve high ARI
    row_k3 = diag[diag["k"] == 3].iloc[0]
    assert float(row_k3["ari_position_group"]) > 0.7

    km_res = fit_kmeans_clusters(pool, n_clusters=3, representation="zscore")
    assert len(km_res.clustered) == len(pool)
    assert len(km_res.summary) == 3
    assert km_res.centroids_pctile.shape == (3, len(SIMILARITY_FEATURES))
    # Clusters are ordered by descending mean VAEP/90
    vaep_means = km_res.summary["mean_vaep_p90"].to_numpy()
    assert np.all(vaep_means[:-1] >= vaep_means[1:])
    assert km_res.ari_position_group > 0.7
