# Phase 3 — Streamlit app

Multipage app in `app/` (dark theme in `.streamlit/config.toml`, Plotly `plotly_dark`), organised
into **three analytical sections** via `st.navigation` (`Match Analysis`, `Player Analysis`,
`Season Analysis`). Every page shows "Data: StatsBomb" in the sidebar and a collapsed
*Methodology and limitations* note.

## Run

```bash
uv run python scripts/phase3_build_player_metrics.py   # once; ~2 min (trains VAEP); git-ignored
uv run streamlit run app/Home.py
```

## Structure

| Section / File | Role |
|---|---|
| **Overview** · `app/Home.py` | Entry router (`st.navigation`), counters, global limitations |
| **Match Analysis** · `app/pages/1_Match_Report.py` | Competition -> Team -> Match selector, score, team numbers (with match PPDA), xG race, Shot Map / Shot Table tabs, Top 3 contributors per team across all 5 valuation models |
| **Match Analysis** · `app/pages/3_Pass_Network.py` | Competition -> Team -> Match selector, average pass positions and links until the first substitution |
| **Player Analysis** · `app/pages/6_Player_Match_Report.py` | Competition -> Team -> Player -> Match selector, 6 KPIs, 5 valuation totals, 4-column stat breakdown and 5 tactical pitch maps (xT Gain, xT Loss, Shots, Defensive Actions with xTP, KDE Touch Heatmap) |
| **Player Analysis** · `app/pages/7_Player_Match_Log.py` | Match-by-match trajectory across the 5 valuation models (within-player percentile / Z-score / raw), intra-player Spearman matrix, pass/carry and offensive/defensive decompositions |
| **Player Analysis** · `app/pages/4_Scouting_Radar.py` | Pizza radar (`Barpolar`) across 4 dimensions, KPI cards, metric breakdown table with progress bars, 1x1 comparison |
| **Player Analysis** · `app/pages/2_Shot_Map.py` | Season shots by league/team/player; size = xG |
| **Season Analysis** · `app/pages/8_Valuation_Comparison.py` | Conceptual taxonomy, positional bias (`mean_percentile_by_group`), Spearman correlation heatmaps (with/without GK), Rank Shifts explorer and per-90 leaderboards |
| **Season Analysis** · `app/pages/5_League_Explorer.py` | Quadrant scatter of any two metrics + full player table (`height=600`) |
| **Shared** · `app/data_service.py`, `app/common.py`, `app/pitch.py`, `app/match_plots.py`, `app/radar_plots.py` | Cached loaders, cascading match picker, Plotly/mplsoccer pitch charts and pizza charts |
| **Backend** · `src/statsbomb_analytics/metrics/player_metrics.py`, `player_match.py`, `scouting.py`, `pass_network.py`, `selection.py` | Pure functions for season/match player tables, tactical breakdowns, percentiles, pass networks and cascading filters |

## Decisions

| # | Decision | Reason |
|---|---|---|
| D17 | `scripts/phase3_build_player_metrics.py` builds `player_metrics.parquet` (2,770 players), `player_match_metrics.parquet` (50,606 player-match appearances) and `xt_grid.npy` (12×16) in a single pass. | Fitting VAEP and Action xT once produces both season and match tables with no extra training overhead; derived files stay git-ignored. |
| D18 | Scouting Radar uses a 19-slice Plotly `Barpolar` pizza grouped into 4 colour-coded dimensions (**Attack**, **Creation**, **Defense**, **Possession**), side-by-side with a breakdown table (`ProgressColumn` 0–100) and KPI summary cards. Toggles control 1x1 head-to-head comparison and whether percentiles are computed within each `position_group` or across all positions. | Matches the `atletico-analytics` reference while exposing all five action-valuation models. |
| D19 | Pass network: completed passes with a recipient until the team's first substitution; undirected links; node = mean location of the player's passes. | Standard convention; avoids mixing line-ups. |
| D20 | Player league = league of the team (Champions League excluded). | The competition has one match in the open data. |
| D21 | Cascading team filter in Match Report, Pass Network, Player Match Report, Player Match Log and Scouting Radar. In Scouting Radar, filtering by team only narrows the player dropdown without shrinking the percentile comparison pool. | Makes finding matches/players fast without distorting league percentiles. |
| D22 | Hierarchical navigation (`st.navigation`) splits the app into **Match Analysis**, **Player Analysis** and **Season Analysis**. | Separates single-match event grain, individual player trajectories/profiles, and macro league/methodology studies. |
| D23 | Match Report includes match-level PPDA, a mirrored two-team Shot Map tab alongside the Shot Table, and a **Top 3 Contributors per Team** comparison across Singh xT, xGChain, xGBuildup, Action xT and VAEP. | Shows how the five valuation paradigms highlight different roles within the exact same 90 minutes. |
| D24 | Player Match Report pairs the 5-model match summary and 4-column stat breakdown with 5 tactical pitch maps (Passes/Carries xT Gain, xT Loss with turnover cost, Shots, Defensive Actions with xTP, and KDE Touch Heatmap). | Replicates the `atletico-analytics` single-match scouting workflow on StatsBomb 120×80 event coordinates. |
| D25 | Action-valuation comparisons are exposed both longitudinally per player (`Player Match Log`, using within-player percentiles/Z-scores) and cross-sectionally across the season (`Action Valuation Comparison`, showing positional bias, Spearman matrices and rank shifts). | Core academic contribution of the project: contrasting Markov grids, possession chains, polynomial geometry and gradient-boosted state models. |

## Limitations

See `docs/phase2_metrics.md`. In the app: radar and per-90 values are noisy under ~900 minutes,
volume metrics reflect team style, cross-league pools are not calibrated, one season only,
single-match action values are small samples, and StatsBomb 2015/16 open event data contains no
physical GPS tracking (distance covered or sprint speed).

## Tests

`tests/test_app_smoke.py` runs every page with Streamlit's `AppTest` (skipped when
processed artifacts are missing); the logic is covered by `test_metrics_player_match.py`,
`test_metrics_scouting.py`, `test_metrics_pass_network.py` and `test_selection.py`.
