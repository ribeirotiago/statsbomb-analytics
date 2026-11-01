# StatsBomb Analytics

Exploratory analyses and visualizations (xT, xG, scouting, player similarity) built on
[StatsBomb Open Data](https://github.com/statsbomb/open-data), covering the **2015/16** season of
Europe's top 5 leagues. **Academic/portfolio project, non-commercial**, inspired by
`atletico-analytics`.

> Data: StatsBomb. Exploratory results.

## Coverage (Phase 0)
See [docs/phase0_coverage.md](docs/phase0_coverage.md). 2015/16 summary: Premier League, La Liga and
Serie A with 380 matches; Bundesliga with 306 (complete); Ligue 1 with 377 of 380; **Champions League
with only 1 match (the final)**, so it is excluded from aggregate analyses. No 360 data.

## Setup
```bash
uv venv .venv && uv pip install -p .venv/bin/python -e ".[dev]"
cp .env.example .env   # adjust STATSBOMB_DATA_DIR if needed
./.venv/bin/python scripts/phase0_coverage.py
./.venv/bin/pytest
```

## Limitations
- Single season (no temporal evolution); small per-player samples; team style bias.
- Cross-league comparisons are not calibrated (different teams and styles).
