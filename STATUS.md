# Status

- [x] Phase 0 — Coverage survey ([docs/phase0_coverage.md](docs/phase0_coverage.md)).
- [x] Scaffold: pyproject, Ruff, pytest, GEMINI.md, `config.py`, `ingest/coverage.py`.
- [x] Phase 1 — ETL: matches, teams, players, lineups, events → Parquet (`etl/tables.py`, `etl/events.py`, `scripts/phase1_etl.py`); DuckDB views in `storage/duckdb_store.py`.
- [ ] Phase 2 — Metrics: per-90 by position, xG, xT, PPDA.
- [ ] Phase 3 — Streamlit app: League Explorer, Match Report, Shot Map, Pass Network, Scouting Radar.
- [ ] Phase 4 — Similarity (k-NN → PCA/K-Means), documentation, deploy with an aggregated subset.
