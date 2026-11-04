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
  with `AppTest` (D17–D25). Branch `feat/phase3-app`.
- [ ] Phase 4 — Similarity (k-NN → PCA/K-Means), documentation, deploy with an aggregated subset.

## Last session summary (2026-11-04)

1. Reorganised the Streamlit app into three analytical sections (`Match Analysis`,
   `Player Analysis`, `Season Analysis`) using `st.navigation` in `app/Home.py` (D22).
2. Added `metrics/player_match.py` (`tests/test_metrics_player_match.py`) and extended
   `scripts/phase3_build_player_metrics.py` (`build_all_app_artifacts`) to build
   `player_metrics.parquet` (2,770 rows), `player_match_metrics.parquet` (50,606 player-match
   appearances) and `xt_grid.npy` (12×16) in a single pass.
3. Enhanced `Match Report` (`app/pages/1_Match_Report.py`, D23) with match PPDA, `Pitch Map` /
   `Shot Table` tabs and a 5-column card layout listing the **Top 3 Contributors per Team** across
   `Singh xT`, `xGChain`, `xGBuildup`, `Action xT` and `VAEP`.
4. Built `Player Match Report` (`app/pages/6_Player_Match_Report.py`, `app/match_plots.py`, D24)
   with 5-model match KPIs, 4-column stat breakdown and 5 tactical pitch maps (`xT Gain`, `xT Loss`
   with turnover cost, `Shots`, `Defensive Actions` with `xTP`, and `KDE Touch Heatmap`).
5. Built `Player Match Log & Valuation` (`app/pages/7_Player_Match_Log.py`, D25) and
   `Action Valuation Comparison` (`app/pages/8_Valuation_Comparison.py`, D25), and added mean
   quadrant lines to `League Explorer` (`app/pages/5_League_Explorer.py`). Test suite at 106 passed.

## Next session

Branch: `feat/phase3-app`.

1. Finish any remaining visual review of the Phase 3 pages (`uv run streamlit run app/Home.py`),
   then merge `feat/phase3-app` into `main`.
2. Start Phase 4 on a new branch (`feat/phase4-similarity`):
   - Add a `Similar Players & Profiles` page inside the **Player Analysis** section:
     - Explainable baseline first: Euclidean / cosine k-NN on percentiles / standardised per-90
       features, with position-group and minimum-minutes filters plus 1x1 pizza overlay.
     - Role/archetype scoring and exploratory ML next: PCA projection (biplot + loadings) and
       K-Means clustering (Silhouette, elbow, ARI/NMI agreement vs position/archetypes).
   - Prepare a lightweight aggregated subset for deployment and finalise documentation.
