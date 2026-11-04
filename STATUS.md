# Status

- [x] Phase 0 — Coverage survey ([docs/phase0_coverage.md](docs/phase0_coverage.md)).
- [x] Scaffold: pyproject, Ruff, pytest, GEMINI.md, `config.py`, `ingest/coverage.py`.
- [x] Phase 1 — ETL: matches, teams, players, lineups, events → Parquet; DuckDB views
  ([docs/phase1_etl.md](docs/phase1_etl.md)). Merged into `main`.
- [x] Phase 2 — Metrics ([docs/phase2_metrics.md](docs/phase2_metrics.md)): minutes, per-90, xG,
  xT, PPDA, xGChain/xGBuildup, action-based xT, in-house out-of-fold VAEP and the method comparison
  (D1–D16). Merged into `main`.
- [x] Phase 3 — Streamlit app ([docs/phase3_app.md](docs/phase3_app.md)): hierarchical
  `st.navigation` across **Match Analysis** (enhanced Match Report with PPDA, Shot Map/Table tabs &
  5-column Top 3 contributor cards per valuation method; Pass Network), **Player Analysis** (Player
  Match Report with 5 tactical pitch maps, Player Match Log & Valuation trajectory, Scouting Radar,
  Shot Map & Finishing) and **Season Analysis** (Action Valuation Comparison, League Explorer);
  prebuilt `player_metrics.parquet`, `player_match_metrics.parquet` and `xt_grid.npy`; smoke-tested
  with `AppTest` (D17–D25). Ready to merge into `main`.
- [x] Phase 4 — Similarity, Tactical Archetypes & Exploratory ML
  ([docs/phase4_similarity.md](docs/phase4_similarity.md)): Euclidean & Cosine k-NN baseline
  (`find_similar_players`, `metric_deltas`), 7 domain tactical archetypes (`score_archetypes`),
  PCA biplot (`fit_pca_profile`) and K-Means clustering with Silhouette/Elbow and ARI/NMI agreement
  diagnostics (`evaluate_kmeans_range`, `fit_kmeans_clusters`), exposed in
  `app/pages/9_Similar_Players.py` (`Similar Players & Profiles`, D26–D30). Branch
  `feat/phase4-similarity`.

## Last session summary (2026-11-04)

1. Closed Phase 3 (`feat/phase3-app`) and started Phase 4 on `feat/phase4-similarity`.
2. Implemented `src/statsbomb_analytics/metrics/similarity.py` (`tests/test_metrics_similarity.py`)
   following a baseline-first progression (D26–D30):
   - **k-NN baseline (`find_similar_players`, `metric_deltas`)**: Euclidean and Cosine distance over
     the 19 scouting metrics in either `percentiles` (0–100) or standardised `zscore` space, with
     position-group and same-team filters plus per-attribute delta decomposition.
   - **Domain Tactical Archetypes (`ARCHETYPES`, `score_archetypes`, `player_archetype_profile`)**:
     7 transparent rule-based tactical roles scored as weighted averages of percentiles.
   - **Exploratory ML (`fit_pca_profile`, `evaluate_kmeans_range`, `fit_kmeans_clusters`)**: 3D PCA
     projection with deterministic sign convention and loading vectors, plus K-Means clustering
     ordered by mean `vaep_p90` and evaluated across $k = 2 \dots 8$ via Inertia, Silhouette, ARI
     and NMI against `position_group` and `archetype`.
3. Added `Similar Players & Profiles` (`app/pages/9_Similar_Players.py`) to the **Player Analysis**
   section in `app/Home.py`, including 1x1 `Barpolar` pizza overlay, closest/diverging attribute
   tables, archetype spectrum and interactive PCA biplot + K-Means diagnostics.
4. Added unit and `AppTest` smoke tests (`116 passed`).

## Next session

Branch: `feat/phase4-similarity`.

1. Review `Similar Players & Profiles` in the UI (`uv run streamlit run app/Home.py`).
2. Prepare a lightweight aggregated deployment subset (for Streamlit Community Cloud without raw
   event files) if needed, and merge `feat/phase4-similarity` into `main`.
