# StatsBomb Analytics

Exploratory analyses and visualizations (xT, xG, scouting, player similarity) built on
[StatsBomb Open Data](https://github.com/statsbomb/open-data), covering the **2015/16** season of
Europe's top 5 leagues. **Academic/portfolio project, non-commercial**.

> Data: StatsBomb. Exploratory results.

## Coverage (Phase 0)
See [docs/phase0_coverage.md](docs/phase0_coverage.md). 2015/16 summary: Premier League, La Liga and
Serie A with 380 matches; Bundesliga with 306 (complete); Ligue 1 with 377 of 380; **Champions League
with only 1 match (the final)**, so it is excluded from aggregate analyses. No 360 data.

## Setup
```bash
uv sync --extra dev   # creates .venv, installs the project and dev tools, writes uv.lock
cp .env.example .env   # adjust STATSBOMB_DATA_DIR if needed
uv run python scripts/phase0_coverage.py
uv run python scripts/phase1_etl.py   # Parquet in data/processed/ (git-ignored)
uv run pytest
```

## ETL (Phase 1)
StatsBomb JSON → Parquet, queried with DuckDB (no ORM, no persisted database file). Details,
tables and limitations: [docs/phase1_etl.md](docs/phase1_etl.md). Progress: [STATUS.md](STATUS.md).

## App & Similarity (Phases 3–4)
```bash
uv run python scripts/phase3_build_player_metrics.py   # builds season & match player tables once
uv run streamlit run app/Home.py
```
Sections and pages:
- **Match Analysis**: Match Report (with PPDA, Shot Map/Table tabs & Top 3 contributors across 5 valuation models), Pass Network.
- **Player Analysis**: Player Match Report (5 tactical pitch maps), Player Match Log & Valuation, Scouting Radar (19-slice `Barpolar` pizza), Similar Players & Profiles (Euclidean/Cosine k-NN, 7 tactical archetypes, PCA biplot & K-Means), Shot Map & Finishing.
- **Season Analysis**: Action Valuation Comparison (Singh xT, xGChain, xGBuildup, Action xT, out-of-fold VAEP), League Explorer.

Details: [docs/phase2_metrics.md](docs/phase2_metrics.md), [docs/phase3_app.md](docs/phase3_app.md) and [docs/phase4_similarity.md](docs/phase4_similarity.md).

## Limitations
- Single season (no temporal evolution); small per-player samples; team style bias.
- Cross-league comparisons are not calibrated (different teams and styles).
