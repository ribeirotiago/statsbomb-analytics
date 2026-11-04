"""Player similarity (k-NN), domain tactical archetypes, PCA projection and K-Means clustering.

Assumptions and limitations:
- Methodology follows a strict baseline-first progression (GEMINI.md §4):
  1. Explainable k-NN similarity (Euclidean and Cosine) over rank percentiles or Z-scores.
  2. Transparent rule-based tactical archetype scoring (weighted percentile combinations).
  3. Unsupervised ML (PCA dimensionality reduction and K-Means clustering) evaluated against
     positional groups and domain archetypes via Silhouette, ARI and NMI.
- Euclidean distance captures both profile shape and overall volume magnitude; Cosine distance
  normalises each player vector to unit length, capturing tactical style/proportion regardless
  of team possession volume.
- Single season (2015/16) and event-data grain: off-ball positioning, pressing intensity without
  on-ball actions, and physical tracking speeds are unobserved.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

from statsbomb_analytics.metrics.scouting import (
    METRICS,
    PCTILE_SUFFIX,
    RadarMetric,
    add_percentiles,
)

SIMILARITY_FEATURES: tuple[str, ...] = tuple(m.column for m in METRICS)
FEATURE_BY_COLUMN: dict[str, RadarMetric] = {m.column: m for m in METRICS}


@dataclass(frozen=True)
class ArchetypeSpec:
    """Domain-defined tactical role scored as a weighted average of metric percentiles."""

    name: str
    description: str
    target_groups: tuple[str, ...]
    weights: dict[str, float]


ARCHETYPES: tuple[ArchetypeSpec, ...] = (
    ArchetypeSpec(
        name="Poacher & Box Finisher",
        description="High-volume penalty-box shooter and clinical goal scorer (npxG, Goals, VAEP).",
        target_groups=("FWD",),
        weights={
            "goals_p90": 0.30,
            "npxg_p90": 0.30,
            "shots_p90": 0.20,
            "vaep_p90": 0.20,
        },
    ),
    ArchetypeSpec(
        name="Direct Winger & Ball Carrier",
        description="1v1 dribbler and progressive carrier attacking the box from wide zones.",
        target_groups=("FWD", "MID"),
        weights={
            "dribbles_completed_p90": 0.35,
            "dribble_pct": 0.15,
            "xt_p90": 0.20,
            "key_passes_p90": 0.15,
            "shots_p90": 0.15,
        },
    ),
    ArchetypeSpec(
        name="Chance Creator & Playmaker",
        description="Final-third creator supplying shot assists, goals Assists, xT and xGChain.",
        target_groups=("MID", "FWD"),
        weights={
            "key_passes_p90": 0.30,
            "assists_p90": 0.20,
            "xt_p90": 0.20,
            "xgchain_p90": 0.15,
            "axt_p90": 0.15,
        },
    ),
    ArchetypeSpec(
        name="Deep-Lying Progressor",
        description="High-volume accurate passer driving possession chains before the final shot.",
        target_groups=("MID", "DEF"),
        weights={
            "passes_p90": 0.25,
            "pass_pct": 0.20,
            "xgbuildup_p90": 0.25,
            "axt_p90": 0.20,
            "xt_p90": 0.10,
        },
    ),
    ArchetypeSpec(
        name="Box-to-Box Engine",
        description="Two-way midfielder combining counter-pressing, recoveries and progression.",
        target_groups=("MID",),
        weights={
            "pressures_p90": 0.25,
            "ball_recoveries_p90": 0.25,
            "xgchain_p90": 0.20,
            "tackles_p90": 0.15,
            "dribbles_completed_p90": 0.15,
        },
    ),
    ArchetypeSpec(
        name="Ball-Winning Destroyer",
        description="Aggressive ball-winner specialising in tackles, interceptions and pressures.",
        target_groups=("MID", "DEF"),
        weights={
            "tackles_p90": 0.30,
            "interceptions_p90": 0.25,
            "pressures_p90": 0.25,
            "ball_recoveries_p90": 0.20,
        },
    ),
    ArchetypeSpec(
        name="Ball-Playing Defender & Stopper",
        description="Defensive anchor combining clearances and interceptions with safe build-up.",
        target_groups=("DEF",),
        weights={
            "clearances_p90": 0.30,
            "interceptions_p90": 0.25,
            "pass_pct": 0.20,
            "passes_p90": 0.15,
            "xgbuildup_p90": 0.10,
        },
    ),
)


@dataclass(frozen=True)
class PCAResult:
    """Output of a PCA projection over the 19 scouting metrics."""

    projection: pd.DataFrame
    explained_variance_ratio: tuple[float, ...]
    loadings: pd.DataFrame


@dataclass(frozen=True)
class KMeansResult:
    """Output of K-Means clustering with diagnostic agreement metrics."""

    clustered: pd.DataFrame
    centroids_pctile: pd.DataFrame
    summary: pd.DataFrame
    silhouette: float
    ari_position_group: float
    nmi_position_group: float
    ari_archetype: float
    nmi_archetype: float


def ensure_percentiles(pool: pd.DataFrame, by_group: bool = False) -> pd.DataFrame:
    """Return `pool` with `<col>_pctile` columns (0-100), computing them if absent."""
    expected = [f"{col}{PCTILE_SUFFIX}" for col in SIMILARITY_FEATURES]
    if all(col in pool.columns for col in expected):
        return pool.copy()
    return add_percentiles(pool, by_group=by_group)


def build_feature_matrix(
    pool: pd.DataFrame,
    representation: str = "percentiles",
    by_group: bool = False,
    features: tuple[str, ...] = SIMILARITY_FEATURES,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Extract the numerical `(N, D)` feature matrix for similarity or ML.

    Parameters
    ----------
    pool:
        Player season table (`player_metrics.parquet` slice).
    representation:
        `"percentiles"` uses `[0, 100]` empirical ranks (missing rates filled with 0.0);
        `"zscore"` standardises raw per-90 / rate columns to zero mean and unit variance.
    by_group:
        When `representation="percentiles"`, rank within `position_group` if True.
    features:
        Metric column names to include.
    """
    ranked = add_percentiles(pool, by_group=by_group)
    if representation == "percentiles":
        pct_cols = [f"{col}{PCTILE_SUFFIX}" for col in features]
        mat = ranked[pct_cols].fillna(0.0).to_numpy(dtype=float)
        return ranked, mat
    if representation == "zscore":
        raw = ranked[list(features)].fillna(0.0).to_numpy(dtype=float)
        if len(ranked) <= 1:
            return ranked, np.zeros_like(raw, dtype=float)
        scaler = StandardScaler()
        mat = scaler.fit_transform(raw)
        return ranked, mat
    raise ValueError(f"Unsupported feature representation: {representation!r}")


def find_similar_players(
    pool: pd.DataFrame,
    target_player_id: int,
    target_team_id: int,
    *,
    top_k: int = 10,
    metric: str = "euclidean",
    representation: str = "percentiles",
    by_group: bool = True,
    same_position_group: bool = True,
    include_same_team: bool = True,
) -> pd.DataFrame:
    """Rank candidate players by Euclidean or Cosine similarity to the target player.

    Returns a DataFrame of up to `top_k` neighbours with `distance`, `similarity_pct` (0-100),
    player metadata, key per-90 columns and all `<col>_pctile` columns.
    """
    ranked, matrix = build_feature_matrix(
        pool.reset_index(drop=True),
        representation=representation,
        by_group=by_group,
    )
    target_mask = (ranked["player_id"] == int(target_player_id)) & (
        ranked["team_id"] == int(target_team_id)
    )
    if not target_mask.any():
        raise ValueError(
            f"Target player ({target_player_id}, {target_team_id}) not found in the pool."
        )

    target_idx = int(np.flatnonzero(target_mask.to_numpy())[0])
    target_vec = matrix[target_idx]
    target_group = ranked.iloc[target_idx]["position_group"]
    n_features = max(matrix.shape[1], 1)

    if metric == "euclidean":
        diffs = matrix - target_vec
        distances = np.linalg.norm(diffs, axis=1)
        rmsd = distances / np.sqrt(n_features)
        if representation == "percentiles":
            similarity_pct = np.clip(100.0 - rmsd, 0.0, 100.0)
        else:
            similarity_pct = np.clip(100.0 * np.exp(-0.5 * rmsd), 0.0, 100.0)
    elif metric == "cosine":
        norms = np.linalg.norm(matrix, axis=1)
        target_norm = float(np.linalg.norm(target_vec))
        denom = np.where((norms > 0) & (target_norm > 0), norms * target_norm, 1.0)
        cos_sim = np.where((norms > 0) & (target_norm > 0), (matrix @ target_vec) / denom, 0.0)
        cos_sim = np.clip(cos_sim, -1.0, 1.0)
        distances = 1.0 - cos_sim
        if representation == "percentiles":
            similarity_pct = np.clip(cos_sim * 100.0, 0.0, 100.0)
        else:
            similarity_pct = np.clip((cos_sim + 1.0) * 50.0, 0.0, 100.0)
    else:
        raise ValueError(f"Unsupported distance metric: {metric!r}")

    out = ranked.copy()
    out["distance"] = distances
    out["similarity_pct"] = similarity_pct

    candidates = out[~target_mask].copy()
    if same_position_group and pd.notna(target_group):
        candidates = candidates[candidates["position_group"] == target_group].copy()
    if not include_same_team:
        candidates = candidates[candidates["team_id"] != int(target_team_id)].copy()

    candidates = candidates.sort_values(
        ["distance", "minutes", "player_name"], ascending=[True, False, True]
    )
    candidates = candidates.head(int(top_k)).reset_index(drop=True)
    candidates.insert(0, "rank", np.arange(1, len(candidates) + 1))
    return candidates


def metric_deltas(target_row: pd.Series, candidate_row: pd.Series) -> pd.DataFrame:
    """Per-metric comparison between target and candidate (`<col>_pctile` required).

    Sorted by `abs_pctile_diff` ascending so the closest matching attributes appear first.
    """
    rows: list[dict[str, object]] = []
    for m in METRICS:
        pct_col = f"{m.column}{PCTILE_SUFFIX}"
        t_pct = float(target_row.get(pct_col, np.nan))
        c_pct = float(candidate_row.get(pct_col, np.nan))
        if np.isnan(t_pct) or np.isnan(c_pct):
            continue
        diff = c_pct - t_pct
        rows.append(
            {
                "category": m.category,
                "column": m.column,
                "label": m.label,
                "target_p90": float(target_row.get(m.column, np.nan)),
                "candidate_p90": float(candidate_row.get(m.column, np.nan)),
                "target_pctile": round(t_pct, 1),
                "candidate_pctile": round(c_pct, 1),
                "pctile_diff": round(diff, 1),
                "abs_pctile_diff": round(abs(diff), 1),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["abs_pctile_diff", "label"], ascending=[True, True]).reset_index(
        drop=True
    )


def score_archetypes(pool: pd.DataFrame, by_group: bool = False) -> pd.DataFrame:
    """Compute `[0, 100]` scores for each domain archetype and assign primary/secondary roles."""
    ranked = ensure_percentiles(pool, by_group=by_group)
    score_cols: list[str] = []
    for spec in ARCHETYPES:
        col_name = f"arch_{spec.name}"
        score = np.zeros(len(ranked), dtype=float)
        for metric_col, weight in spec.weights.items():
            pct_series = ranked[f"{metric_col}{PCTILE_SUFFIX}"].fillna(0.0).to_numpy(dtype=float)
            score += weight * pct_series
        ranked[col_name] = np.round(score, 1)
        score_cols.append(col_name)

    scores_mat = ranked[score_cols].to_numpy(dtype=float)
    names = np.array([spec.name for spec in ARCHETYPES])
    order = np.argsort(-scores_mat, axis=1)
    primary_idx = order[:, 0]
    secondary_idx = order[:, 1] if scores_mat.shape[1] > 1 else order[:, 0]

    ranked["archetype"] = names[primary_idx]
    ranked["archetype_score"] = np.round(scores_mat[np.arange(len(ranked)), primary_idx], 1)
    ranked["secondary_archetype"] = names[secondary_idx]
    ranked["secondary_archetype_score"] = np.round(
        scores_mat[np.arange(len(ranked)), secondary_idx], 1
    )
    return ranked


def player_archetype_profile(row: pd.Series) -> pd.DataFrame:
    """Return the 7 archetype scores (0-100) for a single player row from `score_archetypes`."""
    records = []
    for spec in ARCHETYPES:
        col_name = f"arch_{spec.name}"
        if col_name in row:
            val = float(row[col_name])
        else:
            val = sum(
                weight * float(row.get(f"{m_col}{PCTILE_SUFFIX}", 0.0) or 0.0)
                for m_col, weight in spec.weights.items()
            )
        records.append(
            {
                "Archetype": spec.name,
                "Target Groups": ", ".join(spec.target_groups),
                "Score": round(val, 1),
                "Description": spec.description,
            }
        )
    return pd.DataFrame(records).sort_values("Score", ascending=False).reset_index(drop=True)


def fit_pca_profile(
    pool: pd.DataFrame,
    *,
    representation: str = "zscore",
    by_group: bool = False,
    n_components: int = 3,
) -> PCAResult:
    """Project players onto the top `n_components` principal components of the scouting metrics."""
    ranked, matrix = build_feature_matrix(
        pool.reset_index(drop=True),
        representation=representation,
        by_group=by_group,
    )
    if representation == "percentiles" and len(ranked) > 1:
        matrix = StandardScaler().fit_transform(matrix)

    k = min(int(n_components), matrix.shape[0], matrix.shape[1])
    pca = PCA(n_components=k, random_state=42)
    coords = pca.fit_transform(matrix)

    # Deterministic sign convention: orient each PC so its largest-magnitude loading is positive.
    components = pca.components_.copy()
    for i in range(k):
        max_idx = int(np.argmax(np.abs(components[i])))
        if components[i, max_idx] < 0:
            components[i] *= -1.0
            coords[:, i] *= -1.0

    proj = ranked.copy()
    for i in range(3):
        proj[f"pc{i + 1}"] = coords[:, i] if i < k else 0.0

    ratios = tuple(float(pca.explained_variance_ratio_[i]) if i < k else 0.0 for i in range(3))

    loading_rows = []
    for feat_idx, col in enumerate(SIMILARITY_FEATURES):
        meta = FEATURE_BY_COLUMN[col]
        pc1_l = float(components[0, feat_idx]) if k >= 1 else 0.0
        pc2_l = float(components[1, feat_idx]) if k >= 2 else 0.0
        pc3_l = float(components[2, feat_idx]) if k >= 3 else 0.0
        loading_rows.append(
            {
                "column": col,
                "label": meta.label,
                "category": meta.category,
                "pc1": round(pc1_l, 4),
                "pc2": round(pc2_l, 4),
                "pc3": round(pc3_l, 4),
                "loading_norm_2d": round(float(np.hypot(pc1_l, pc2_l)), 4),
            }
        )
    loadings = (
        pd.DataFrame(loading_rows)
        .sort_values("loading_norm_2d", ascending=False)
        .reset_index(drop=True)
    )
    return PCAResult(
        projection=proj,
        explained_variance_ratio=ratios,
        loadings=loadings,
    )


def evaluate_kmeans_range(
    pool: pd.DataFrame,
    *,
    k_values: tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8),
    representation: str = "zscore",
    by_group: bool = False,
) -> pd.DataFrame:
    """Evaluate K-Means across `k_values` using Inertia, Silhouette, and ARI/NMI vs roles."""
    with_arch = score_archetypes(pool.reset_index(drop=True), by_group=by_group)
    _, matrix = build_feature_matrix(with_arch, representation=representation, by_group=by_group)
    if representation == "percentiles" and len(with_arch) > 1:
        matrix = StandardScaler().fit_transform(matrix)

    pos_labels = with_arch["position_group"].fillna("UNK").astype(str).to_numpy()
    arch_labels = with_arch["archetype"].fillna("UNK").astype(str).to_numpy()

    rows = []
    for k in k_values:
        if k < 2 or k >= len(with_arch):
            continue
        km = KMeans(n_clusters=int(k), random_state=42, n_init=10)
        labels = km.fit_predict(matrix)
        sil = float(silhouette_score(matrix, labels)) if len(np.unique(labels)) > 1 else 0.0
        rows.append(
            {
                "k": int(k),
                "inertia": float(km.inertia_),
                "silhouette": round(sil, 4),
                "ari_position_group": round(float(adjusted_rand_score(pos_labels, labels)), 4),
                "nmi_position_group": round(
                    float(normalized_mutual_info_score(pos_labels, labels)), 4
                ),
                "ari_archetype": round(float(adjusted_rand_score(arch_labels, labels)), 4),
                "nmi_archetype": round(float(normalized_mutual_info_score(arch_labels, labels)), 4),
            }
        )
    return pd.DataFrame(rows)


def fit_kmeans_clusters(
    pool: pd.DataFrame,
    *,
    n_clusters: int = 5,
    representation: str = "zscore",
    by_group: bool = False,
) -> KMeansResult:
    """Fit K-Means, attach PCA coordinates and domain archetypes, and summarise cluster profiles."""
    pca_res = fit_pca_profile(pool, representation=representation, by_group=by_group)
    df = score_archetypes(pca_res.projection, by_group=by_group)
    _, matrix = build_feature_matrix(df, representation=representation, by_group=by_group)
    if representation == "percentiles" and len(df) > 1:
        matrix = StandardScaler().fit_transform(matrix)

    k = max(2, min(int(n_clusters), len(df) - 1))
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    raw_labels = km.fit_predict(matrix)

    # Order clusters deterministically by descending mean VAEP/90 so C1..CK are stable.
    df["_raw_cluster"] = raw_labels
    cluster_order = (
        df.groupby("_raw_cluster")["vaep_p90"].mean().sort_values(ascending=False).index.to_list()
    )
    remap = {old_c: new_idx for new_idx, old_c in enumerate(cluster_order)}
    df["cluster_id"] = df["_raw_cluster"].map(remap).astype(int)
    df = df.drop(columns=["_raw_cluster"])

    pct_cols = [f"{col}{PCTILE_SUFFIX}" for col in SIMILARITY_FEATURES]
    label_by_pct_col = {f"{m.column}{PCTILE_SUFFIX}": m.label for m in METRICS}

    cluster_labels: dict[int, str] = {}
    summary_rows: list[dict[str, object]] = []
    centroid_rows: list[dict[str, object]] = []

    for cid in range(k):
        sub = df[df["cluster_id"] == cid]
        means = sub[pct_cols].mean()
        top_3_cols = means.sort_values(ascending=False).head(3).index.to_list()
        top_2_names = [label_by_pct_col[c] for c in top_3_cols[:2]]
        signature = ", ".join(f"{label_by_pct_col[c]} ({means[c]:.0f}p)" for c in top_3_cols)
        c_label = f"C{cid + 1}: {' & '.join(top_2_names)}"
        cluster_labels[cid] = c_label

        top_pos = (
            sub["position_group"].mode().iloc[0]
            if not sub["position_group"].dropna().empty
            else "—"
        )
        top_arch = sub["archetype"].mode().iloc[0] if not sub["archetype"].dropna().empty else "—"
        summary_rows.append(
            {
                "cluster_id": cid,
                "cluster_label": c_label,
                "players": int(len(sub)),
                "top_position_group": str(top_pos),
                "top_archetype": str(top_arch),
                "signature_metrics": signature,
                "mean_minutes": round(float(sub["minutes"].mean()), 0),
                "mean_xt_p90": round(float(sub["xt_p90"].mean()), 3),
                "mean_vaep_p90": round(float(sub["vaep_p90"].mean()), 3),
            }
        )
        c_row: dict[str, object] = {"cluster_label": c_label}
        for c in pct_cols:
            c_row[label_by_pct_col[c]] = round(float(means[c]), 1)
        centroid_rows.append(c_row)

    df["cluster_label"] = df["cluster_id"].map(cluster_labels)

    ordered_labels = df["cluster_id"].to_numpy()
    sil = (
        float(silhouette_score(matrix, ordered_labels))
        if len(np.unique(ordered_labels)) > 1
        else 0.0
    )
    pos_arr = df["position_group"].fillna("UNK").astype(str).to_numpy()
    arch_arr = df["archetype"].fillna("UNK").astype(str).to_numpy()

    centroids_df = pd.DataFrame(centroid_rows).set_index("cluster_label")
    summary_df = pd.DataFrame(summary_rows)

    return KMeansResult(
        clustered=df,
        centroids_pctile=centroids_df,
        summary=summary_df,
        silhouette=round(sil, 4),
        ari_position_group=round(float(adjusted_rand_score(pos_arr, ordered_labels)), 4),
        nmi_position_group=round(float(normalized_mutual_info_score(pos_arr, ordered_labels)), 4),
        ari_archetype=round(float(adjusted_rand_score(arch_arr, ordered_labels)), 4),
        nmi_archetype=round(float(normalized_mutual_info_score(arch_arr, ordered_labels)), 4),
    )
