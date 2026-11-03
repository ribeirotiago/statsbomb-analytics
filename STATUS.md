# Status

- [x] Phase 0 — Coverage survey ([docs/phase0_coverage.md](docs/phase0_coverage.md)).
- [x] Scaffold: pyproject, Ruff, pytest, GEMINI.md, `config.py`, `ingest/coverage.py`.
- [x] Phase 1 — ETL: matches, teams, players, lineups, events → Parquet; DuckDB views
  ([docs/phase1_etl.md](docs/phase1_etl.md)). Merged into `main`.
- [x] Phase 2 — Metrics ([docs/phase2_metrics.md](docs/phase2_metrics.md)): minutes, per-90, xG, xT, PPDA, xGChain/xGBuildup, action-based xT, in-house VAEP and the method comparison (D16). Pending merge into `main`.
- [ ] Phase 3 — Streamlit app ([docs/phase3_app.md](docs/phase3_app.md)): all five pages implemented and smoke-tested; pending visual review in the browser and polish.
- [ ] Phase 4 — Similarity (k-NN → PCA/K-Means), documentation, deploy with an aggregated subset.

## Next session

Branch: `feat/phase3-app` (Phase 2 merged or pending merge into `main`).

1. Review the app visually (`./.venv/bin/streamlit run app/Home.py`) and polish layout.
2. Add the Phase 3 section to the README (run instructions, credit already present).
3. Phase 4: similarity (k-NN, then PCA/K-Means), deploy with an aggregated subset.

## Open points

- Minutes played must be derived from `lineup_positions` (from/to time and period, including
  stoppage time); define the rule and test it before computing per-90.
- Decide the minimum minutes threshold for per-90 rankings (small-sample risk).
- Pending README change from earlier sessions was committed separately by the user.
