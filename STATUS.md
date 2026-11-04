# Status

- [x] Phase 0 — Coverage survey ([docs/phase0_coverage.md](docs/phase0_coverage.md)).
- [x] Scaffold: pyproject, Ruff, pytest, GEMINI.md, `config.py`, `ingest/coverage.py`.
- [x] Phase 1 — ETL: matches, teams, players, lineups, events → Parquet; DuckDB views
  ([docs/phase1_etl.md](docs/phase1_etl.md)). Merged into `main`.
- [x] Phase 2 — Metrics ([docs/phase2_metrics.md](docs/phase2_metrics.md)): minutes, per-90, xG,
  xT, PPDA, xGChain/xGBuildup, action-based xT, in-house out-of-fold VAEP and the method comparison
  (D1–D16). Merged into `main`.
- [x] Phase 3 — Streamlit app ([docs/phase3_app.md](docs/phase3_app.md)): hierarchical
  `st.navigation` across **Match Analysis**, **Player Analysis** and **Season Analysis**;
  prebuilt `player_metrics.parquet`, `player_match_metrics.parquet`, `shots.parquet` and
  `xt_grid.npy`; smoke-tested with `AppTest` (D17–D25). Merged into `main`.
- [x] Phase 4 — Similarity, Tactical Archetypes, Exploratory ML & Lightweight Deploy Subset
  ([docs/phase4_similarity.md](docs/phase4_similarity.md)): Euclidean & Cosine k-NN baseline
  (`find_similar_players`, `metric_deltas`), 7 domain tactical archetypes (`score_archetypes`),
  PCA biplot (`fit_pca_profile`), K-Means clustering with Silhouette/Elbow and ARI/NMI agreement
  diagnostics (`evaluate_kmeans_range`, `fit_kmeans_clusters`), `Similar Players & Profiles` page
  (`app/pages/9_Similar_Players.py`), lightweight ~12 MB deploy subset builder
  (`src/statsbomb_analytics/storage/deploy.py`, `scripts/phase4_build_deploy_subset.py`), and
  complete portfolio `README.md` (D26–D31). Ready to merge into `main`.

## Last session summary (2026-11-04)

1. Merged Phase 3 (`feat/phase3-app`) into `main` and completed Phase 4 on `feat/phase4-similarity`.
2. Implemented `src/statsbomb_analytics/metrics/similarity.py` (`tests/test_metrics_similarity.py`)
   following a baseline-first progression (D26–D30):
   - **k-NN baseline (`find_similar_players`, `metric_deltas`)**: Euclidean and Cosine similarity
     over the 19 scouting metrics in either `percentiles` (0–100) or standardised `zscore` space.
   - **Domain Tactical Archetypes (`ARCHETYPES`, `score_archetypes`, `player_archetype_profile`)**:
     7 transparent rule-based tactical roles scored as weighted averages of percentiles.
   - **Exploratory ML (`fit_pca_profile`, `evaluate_kmeans_range`, `fit_kmeans_clusters`)**: 3D PCA
     projection with biplot loading vectors, plus K-Means clustering evaluated across $k = 2 \dots 8$
     via Inertia, Silhouette, ARI and NMI against `position_group` and `archetype`.
3. Added `Similar Players & Profiles` (`app/pages/9_Similar_Players.py`) to the **Player Analysis**
   section in `app/Home.py`.
4. Added lightweight deployment bundle support (`src/statsbomb_analytics/storage/deploy.py`,
   `scripts/phase4_build_deploy_subset.py`, `tests/test_deploy_subset.py`, D31), prebuilt
   `shots.parquet` support in `app/data_service.py`, automatic partial-event-tree detection in
   `app/common.py` and `app/pages/6_Player_Match_Report.py`, and expanded `README.md` into a full
   portfolio showcase.
5. Full test suite at **120 passed** and Ruff check/format clean.
