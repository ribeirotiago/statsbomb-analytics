# Phase 4 — Player Similarity, Tactical Archetypes & Exploratory ML

Player similarity and role discovery built on top of `player_metrics.parquet` (19 per-90 / rate
metrics across **Attack**, **Creation**, **Defense** and **Possession**, including all five
action-valuation models). Following `GEMINI.md §4`, the methodology progresses strictly from
**explainable baselines** to **exploratory unsupervised ML**.

## Architecture & Files

| Layer | File | Role |
|---|---|---|
| **Metrics** | `src/statsbomb_analytics/metrics/similarity.py` | Feature matrix builder (`percentiles` / `zscore`), Euclidean & Cosine k-NN (`find_similar_players`), per-metric delta breakdown (`metric_deltas`), 7 domain tactical archetypes (`score_archetypes`), PCA biplot projection (`fit_pca_profile`) and K-Means clustering + diagnostics (`evaluate_kmeans_range`, `fit_kmeans_clusters`) |
| **App Page** | `app/pages/9_Similar_Players.py` | `Similar Players & Profiles` under **Player Analysis**: 3-tab workflow (`k-NN Player Similarity`, `Tactical Archetypes`, `PCA & K-Means Clustering`) with 1x1 `Barpolar` pizza overlay |
| **Tests** | `tests/test_metrics_similarity.py`, `tests/test_app_smoke.py` | Unit tests for distance metrics, archetype weights, PCA variance and K-Means positional recovery, plus `AppTest` smoke tests |

## Decisions

| # | Decision | Rationale |
|---|---|---|
| D26 | **Baseline first (`find_similar_players`):** Euclidean and Cosine k-NN over the 19 scouting metrics (`SIMILARITY_FEATURES`), supporting both within-group / pool-wide **Percentiles (0–100)** and **Standardised Per-90 (Z-Score)**. | Zero black-box parameters; every similarity score decomposes into exact per-attribute deltas (`metric_deltas`). |
| D27 | **Euclidean vs Cosine distance:** Euclidean similarity (`100 - RMSD` in percentile space) measures both profile shape and volume magnitude; Cosine similarity normalises player vectors to unit length, capturing **proportional tactical style** independent of team possession volume. | Solves a classic football scouting problem: finding a lower-table player whose proportional action profile matches a starter on a high-possession team. |
| D28 | **Domain Tactical Archetypes (`ARCHETYPES`) before K-Means:** 7 domain-defined roles (`Poacher & Box Finisher`, `Direct Winger & Ball Carrier`, `Chance Creator & Playmaker`, `Deep-Lying Progressor`, `Box-to-Box Engine`, `Ball-Winning Destroyer`, `Ball-Playing Defender & Stopper`) scored as weighted averages of percentiles. | Provides transparent, domain-grounded role scores (primary + secondary role per player) and serves as an external benchmark to evaluate unsupervised K-Means clusters. |
| D29 | **PCA Biplot (`fit_pca_profile`):** Principal components fitted on unit-variance features with a deterministic sign convention (largest-magnitude loading oriented positive), plotting `PC1` vs `PC2` alongside top feature loading vectors and highlighting the target player + Top-K k-NN neighbours. | Connects local k-NN neighbourhoods in 19D space to the global 2D low-rank manifold. |
| D30 | **K-Means Diagnostics (`evaluate_kmeans_range`, `fit_kmeans_clusters`):** Clusters are ordered deterministically by descending mean `vaep_p90`, labelled automatically by their top signature metrics, and evaluated across $k = 2 \dots 8$ via **Inertia (Elbow)**, **Silhouette**, **Adjusted Rand Index (ARI)** and **Normalized Mutual Information (NMI)** against both `position_group` and domain `archetype`. | Quantifies cluster separation and measures how closely unsupervised geometry aligns with football positions vs tactical roles. |

## Assumptions, Limitations and Risks

- **Feature correlation & collinearity**: The 19 scouting metrics include correlated creation/progression indicators (`xt_p90`, `axt_p90`, `xgchain_p90`, `xgbuildup_p90`), which naturally gives ball progression strong weight in unweighted Euclidean distance and `PC1`.
- **Percentile compression vs Z-score outliers**: Rank percentiles (`0–100`) bound outliers (robust for general scouting) but compress extreme tail gaps (e.g. Messi vs the 95th percentile); switching to `Z-Score` preserves tail magnitude at the cost of sensitivity to small-minute outliers.
- **Single season (2015/16) & event grain**: Off-ball positioning, pressing runs without on-ball pressure events, and physical tracking speeds are unobserved.
