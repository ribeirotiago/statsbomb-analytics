# Status

- [x] Phase 0 — Coverage survey ([docs/phase0_coverage.md](docs/phase0_coverage.md)).
- [x] Scaffold: pyproject, Ruff, pytest, GEMINI.md, `config.py`, `ingest/coverage.py`.
- [x] Phase 1 — ETL: matches, teams, players, lineups, events → Parquet; DuckDB views
  ([docs/phase1_etl.md](docs/phase1_etl.md)). Merged into `main`.
- [ ] Phase 2 — Metrics ([docs/phase2_metrics.md](docs/phase2_metrics.md)): minutes, per-90, xG, xT, PPDA done and validated; next: xGChain/xGBuildup, action-based xT, VAEP and the model comparison.
- [ ] Phase 3 — Streamlit app: League Explorer, Match Report, Shot Map, Pass Network, Scouting Radar.
- [ ] Phase 4 — Similarity (k-NN → PCA/K-Means), documentation, deploy with an aggregated subset.

## Next session

Branch: `feat/phase2-metrics` (decisions D1-D12 in [docs/phase2_metrics.md](docs/phase2_metrics.md)).

1. Commit the pending files (`action_xt.py`, `vaep.py`, their tests, docs, STATUS).
2. ~~Action-based xT~~ done in `metrics/action_xt.py` (D14).
3. ~~VAEP in-house~~ done in `metrics/vaep.py` (D15), out-of-fold by match.
4. ~~Comparison table~~ done in `metrics/comparison.py` (D16, results in docs/phase2_metrics.md); outfield-only matrix included. Original item: comparison table of the four methods (Singh xT, xGChain/xGBuildup, action-based xT, VAEP):
   Spearman correlation, players each method favours, analysis by position. Record D13+ in the
   decision log (include the xGBuildup rules).
5. Then `docs/phase2_metrics.md` wrap-up and Phase 3 (the UI minutes slider defaults to 300).

## Open points

- Minutes played must be derived from `lineup_positions` (from/to time and period, including
  stoppage time); define the rule and test it before computing per-90.
- Decide the minimum minutes threshold for per-90 rankings (small-sample risk).
- Pending README change from earlier sessions was committed separately by the user.
