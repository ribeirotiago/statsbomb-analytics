# Phase 3 — Streamlit app

Multipage app in `app/` (dark theme in `.streamlit/config.toml`, Plotly `plotly_dark`). Every page
shows "Data: StatsBomb" in the sidebar and a collapsed *Methodology and limitations* note.

## Run

```bash
uv run python scripts/phase3_build_player_metrics.py   # once; ~3 min (trains VAEP); git-ignored
uv run streamlit run app/Home.py
```

## Structure

| File | Role |
|---|---|
| `app/Home.py` | Overview, counters, global limitations |
| `app/pages/1_Match_Report.py` | Competition -> Team -> Match selector, score, team numbers, xG race, shot table |
| `app/pages/2_Shot_Map.py` | Season shots by league/team/player; size = xG |
| `app/pages/3_Pass_Network.py` | Competition -> Team -> Match selector, average pass positions and links until the first substitution |
| `app/pages/4_Scouting_Radar.py` | Pizza radar (`Barpolar`) across 4 dimensions, KPI cards, metric breakdown table with progress bars, 1x1 comparison |
| `app/pages/5_League_Explorer.py` | Scatter of any two metrics + full table (`height=600`) |
| `app/data_service.py` | Cached loaders (DuckDB over Parquet) |
| `app/common.py`, `app/pitch.py`, `app/radar_plots.py` | Page setup/credit/methodology, cascading match picker, Plotly pitch and pizza charts |
| `src/statsbomb_analytics/selection.py` | Pure helpers for team/match cascading filters |
| `src/statsbomb_analytics/metrics/player_metrics.py` | Player table for the app |
| `src/statsbomb_analytics/metrics/scouting.py` | 19-metric 4-dimension pizza profiles, percentiles and breakdown table |
| `src/statsbomb_analytics/metrics/pass_network.py` | Nodes/edges of a pass network |

## Decisions

| # | Decision | Reason |
|---|---|---|
| D17 | The app reads a prebuilt `data/processed/player_metrics.parquet` (2,770 players) instead of computing metrics on start. | VAEP training takes minutes; the file is small but derived, so it stays git-ignored. |
| D18 | Scouting Radar uses a 19-slice Plotly `Barpolar` pizza grouped into 4 colour-coded dimensions (**Attack**, **Creation**, **Defense**, **Possession**), side-by-side with a breakdown table (`ProgressColumn` 0–100) and KPI summary cards. Toggles control 1x1 head-to-head comparison and whether percentiles are computed within each `position_group` or across all positions. | Matches the `atletico-analytics` reference while exposing all five action-valuation models. |
| D19 | Pass network: completed passes with a recipient until the team's first substitution; undirected links; node = mean location of the player's passes. | Standard convention; avoids mixing line-ups. |
| D20 | Player league = league of the team (Champions League excluded). | The competition has one match in the open data. |
| D21 | Cascading team filter in Match Report, Pass Network and Scouting Radar. In Scouting Radar, filtering by team only narrows the player dropdown without shrinking the percentile comparison pool. | Makes finding matches/players fast without distorting league percentiles. |

## Limitations

See `docs/phase2_metrics.md`. In the app: radar and per-90 values are noisy under ~900 minutes,
volume metrics reflect team style, cross-league pools are not calibrated, one season only, and
the pass network of one match is a small sample.

## Tests

`tests/test_app_smoke.py` runs every page with Streamlit's `AppTest` (skipped when
`player_metrics.parquet` is missing); the logic is covered by `test_metrics_scouting.py`,
`test_metrics_pass_network.py` and `test_selection.py`.
