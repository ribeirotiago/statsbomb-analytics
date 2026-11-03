# Phase 3 — Streamlit app

Multipage app in `app/` (dark theme in `.streamlit/config.toml`, Plotly `plotly_dark`). Every page
shows "Data: StatsBomb" in the sidebar and a collapsed *Methodology and limitations* note.

## Run

```bash
./.venv/bin/python scripts/phase3_build_player_metrics.py   # once; ~3 min (trains VAEP); git-ignored
./.venv/bin/streamlit run app/Home.py
```

## Structure

| File | Role |
|---|---|
| `app/Home.py` | Overview, counters, global limitations |
| `app/pages/1_Match_Report.py` | Score, team numbers, xG race, shot table |
| `app/pages/2_Shot_Map.py` | Season shots by league/team/player; size = xG |
| `app/pages/3_Pass_Network.py` | Average pass positions and links until the first substitution |
| `app/pages/4_Scouting_Radar.py` | Percentile radar by position group, 1x1 comparison |
| `app/pages/5_League_Explorer.py` | Scatter of any two metrics + full table (`height=600`) |
| `app/data_service.py` | Cached loaders (DuckDB over Parquet) |
| `app/common.py`, `app/pitch.py` | Page setup/credit/methodology, Plotly pitch |
| `src/statsbomb_analytics/metrics/player_metrics.py` | Player table for the app |
| `src/statsbomb_analytics/metrics/scouting.py` | Radar profiles and percentiles |
| `src/statsbomb_analytics/metrics/pass_network.py` | Nodes/edges of a pass network |

## Decisions

| # | Decision | Reason |
|---|---|---|
| D17 | The app reads a prebuilt `data/processed/player_metrics.parquet` (2,770 players) instead of computing metrics on start. | VAEP training takes minutes; the file is small but derived, so it stays git-ignored. |
| D18 | Radar percentiles are rank-based inside a pool (same position group, minutes >= slider, one league or all). Profiles per group (FWD/MID/DEF); no goalkeeper profile. | Explainable baseline; event data barely describes goalkeeping. |
| D19 | Pass network: completed passes with a recipient until the team's first substitution; undirected links; node = mean location of the player's passes. | Standard convention; avoids mixing line-ups. |
| D20 | Player league = league of the team (Champions League excluded). | The competition has one match in the open data. |

## Limitations

See `docs/phase2_metrics.md`. In the app: radar and per-90 values are noisy under ~900 minutes,
volume metrics reflect team style, cross-league pools are not calibrated, one season only, and
the pass network of one match is a small sample.

## Tests

`tests/test_app_smoke.py` runs every page with Streamlit's `AppTest` (skipped when
`player_metrics.parquet` is missing); the logic is covered by `test_metrics_scouting.py` and
`test_metrics_pass_network.py`.
