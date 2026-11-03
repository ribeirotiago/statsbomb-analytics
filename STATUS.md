# Status

- [x] Phase 0 — Coverage survey ([docs/phase0_coverage.md](docs/phase0_coverage.md)).
- [x] Scaffold: pyproject, Ruff, pytest, GEMINI.md, `config.py`, `ingest/coverage.py`.
- [x] Phase 1 — ETL: matches, teams, players, lineups, events → Parquet; DuckDB views
  ([docs/phase1_etl.md](docs/phase1_etl.md)). Merged into `main`.
- [x] Phase 2 — Metrics ([docs/phase2_metrics.md](docs/phase2_metrics.md)): minutes, per-90, xG,
  xT, PPDA, xGChain/xGBuildup, action-based xT, in-house out-of-fold VAEP and the method comparison
  (D1–D16). Merged into `main`.
- [x] Phase 3 — Streamlit app ([docs/phase3_app.md](docs/phase3_app.md)): Home, Match Report,
  Shot Map, Pass Network, Scouting Radar (PyPizza-style `Barpolar`, 1x1 comparison, breakdown table)
  and League Explorer; cascading team filters; prebuilt `player_metrics.parquet`; smoke-tested with
  `AppTest` (D17–D21). Branch `feat/phase3-app`.
- [ ] Phase 4 — Similarity (k-NN → PCA/K-Means), documentation, deploy with an aggregated subset.

## Last session summary (2026-11-03)

1. Completed the Phase 2 action-valuation comparison (`metrics/comparison.py`,
   `scripts/phase2_comparison.py`, D16), including the outfield-only Spearman matrix, and merged
   `feat/phase2-metrics` into `main`.
2. Built the Phase 3 multipage Streamlit app (`app/`, `scripts/phase3_build_player_metrics.py`,
   `metrics/player_metrics.py`, `metrics/scouting.py`, `metrics/pass_network.py`, `selection.py`)
   and its documentation ([docs/phase3_app.md](docs/phase3_app.md), D17–D21).
3. Added cascading Competition → Team → Match/Player selectors in Match Report, Pass Network and
   Scouting Radar, and redesigned Scouting Radar to match the `atletico-analytics` reference
   (19-slice 4-dimension `Barpolar` pizza, 1x1 overlay, KPI summary cards and `ProgressColumn`
   breakdown table).
4. Updated `README.md` to use `uv sync` / `uv run` and committed `uv.lock`. Test suite at 94 passed.

## Next session

Branch: `feat/phase3-app`.

1. Finish any remaining visual review/polish of the Phase 3 pages in the browser
   (`uv run streamlit run app/Home.py`), then merge `feat/phase3-app` into `main`.
2. Start Phase 4 on a new branch (`feat/phase4-similarity`):
   - Explainable baseline first: Euclidean / cosine k-NN on standardised per-90 features, with
     position-group and minimum-minutes filters.
   - Exploratory ML next: PCA projection and K-Means clustering, with justification and methodology
     notes in the UI.
   - Prepare a lightweight aggregated subset for deployment (never versioning raw or heavy data).
