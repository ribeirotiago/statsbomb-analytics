# Status

- [x] Phase 0 — Coverage survey ([docs/phase0_coverage.md](docs/phase0_coverage.md)).
- [x] Scaffold: pyproject, Ruff, pytest, GEMINI.md, `config.py`, `ingest/coverage.py`.
- [x] Phase 1 — ETL: matches, teams, players, lineups, events → Parquet; DuckDB views
  ([docs/phase1_etl.md](docs/phase1_etl.md)). Merged into `main`.
- [ ] Phase 2 — Metrics: per-90 by position, xG, xT, PPDA.
- [ ] Phase 3 — Streamlit app: League Explorer, Match Report, Shot Map, Pass Network, Scouting Radar.
- [ ] Phase 4 — Similarity (k-NN → PCA/K-Means), documentation, deploy with an aggregated subset.

## Next session

1. Create branch `feat/phase2-metrics` from `main` (one branch per phase; commits by the user).
2. Start with the simple metrics: per-90 by position (minutes from `lineup_positions`) and xG
   aggregates (`shot_xg`); then xT and PPDA.
3. Reuse from `atletico-analytics`: xT grid/value iteration, percentiles and radar.
4. Each metric function needs a test in `tests/`; document assumptions and limitations.
5. Access data only through `storage/duckdb_store.py`; regenerate Parquet with
   `scripts/phase1_etl.py` if `data/processed/` is missing.

## Open points

- Minutes played must be derived from `lineup_positions` (from/to time and period, including
  stoppage time); define the rule and test it before computing per-90.
- Decide the minimum minutes threshold for per-90 rankings (small-sample risk).
- Pending README change from earlier sessions was committed separately by the user.
