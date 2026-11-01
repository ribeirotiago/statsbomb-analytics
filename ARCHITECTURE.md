# Architecture

One-way layers (each only knows the ones below it):

```
data (StatsBomb clone, read-only)
  → ingest/  (JSON reading, coverage)
  → etl/     (pure parsers: events, lineups → tables)
  → storage/ (Parquet/DuckDB in data/processed/)
  → metrics/ (xT, xG, per-90, percentiles, similarity)
  → plots/ + app/ (Streamlit)
```

Code lives in `src/statsbomb_analytics/`. Reuse from `atletico-analytics`: xT grid/value iteration,
percentiles and radar, k-NN/PCA/K-Means (adapting inputs to StatsBomb events).
