# StatsBomb Analytics ⚽

Exploratory and methodological football analytics platform built on
[StatsBomb Open Data](https://github.com/statsbomb/open-data), covering the **2015/16** season
across Europe's Big 5 leagues (**1,823 league matches + Champions League final**, **2,770 players**,
**50,606 player-match appearances**, **~6.4M on-ball events**).

> **Data: StatsBomb** (Open Data, non-commercial license). Academic and portfolio project.

---

## Highlights & Academic Contributions

1. **Five Action-Valuation Paradigms Compared on Identical Players (`Phase 2`)**
   - **Singh xT** ($12 \times 16$ Markov value-iteration grid over 120×80 pitch coordinates; transitions fitted on all move attempts so turnover risk is priced into zone values).
   - **xGChain & xGBuildup** (possession-chain credit assignment from non-penalty shot xG, with and without the shooter/key passer).
   - **Action-based xT** (degree-3 polynomial logistic $P(\text{shot})$ $\times$ linear $\mathbb{E}[\text{xG}]$ evaluated on held-out matches).
   - **Out-of-Fold VAEP** (in-house 5-fold match cross-validated `HistGradientBoostingClassifier` estimating $\Delta P(\text{score}) - \Delta P(\text{concede})$ over a 10-action horizon across 13 action types).
   - **Empirical finding**: Volume/progression models correlate strongly with one another ($\rho = 0.55\text{--}0.66$ on outfield players), whereas out-of-fold **VAEP** diverges ($\rho = 0.34\text{--}0.53$) because it penalises turnovers and heavily rewards clinical finishing and goalkeeper shot-stopping.

2. **Baseline-First Player Similarity, Tactical Archetypes & Exploratory ML (`Phase 4`)**
   - **Explainable k-NN (`Euclidean` & `Cosine`)** over 19 scouting metrics (`percentiles` or standardised `zscore`) with per-attribute delta decomposition (`metric_deltas`).
   - **7 Domain Tactical Archetypes** (*Poacher & Box Finisher*, *Direct Winger & Ball Carrier*, *Chance Creator & Playmaker*, *Deep-Lying Progressor*, *Box-to-Box Engine*, *Ball-Winning Destroyer*, *Ball-Playing Defender & Stopper*) scored as transparent weighted averages of percentiles.
   - **Unsupervised ML (`PCA` & `K-Means`)** with 2D biplot loading vectors, Silhouette/Elbow curves ($k = 2 \dots 8$), and external cluster agreement (**ARI** & **NMI**) against positional groups and domain archetypes.

3. **Three-Section Interactive Streamlit Application (`Phase 3 & 4`)**
   - Dark-theme multipage app (`plotly_dark` + `mplsoccer`) organised into **Match Analysis**, **Player Analysis** and **Season Analysis**, backed by DuckDB views over partitioned Parquet files and a lightweight (~12 MB) cloud deployment bundle builder.

---

## Architecture

One-way layered architecture (`src/statsbomb_analytics/`):

```text
StatsBomb Open Data clone (read-only JSON)
  └── ingest/   (JSON readers, competition/season coverage survey)
  └── etl/      (pure parsers: matches, teams, players, lineups, Hive-partitioned events)
  └── storage/  (DuckDB in-memory views over Parquet + lightweight deploy bundle builder)
  └── metrics/  (minutes, per-90, Singh xT, PPDA, xGChain/xGBuildup, Action xT, VAEP,
                 method comparison, pass network, scouting percentiles, k-NN/PCA/K-Means)
  └── app/      (Streamlit multipage frontend, Plotly & mplsoccer visualisations)
```

---

## Streamlit App Structure

| Section | Page | Description |
|---|---|---|
| **Overview** | `app/Home.py` | Navigation router (`st.navigation`), dataset counters and global methodological notes |
| **1. Match Analysis** | `1_Match_Report.py` | Scoreboard, team stats (with match **PPDA**), cumulative **xG race**, mirrored **Shot Map / Shot Table** tabs, and **Top 3 Contributors per Team** across all 5 valuation models |
| **1. Match Analysis** | `3_Pass_Network.py` | Average pass locations and undirected connection volume up to the team's first substitution |
| **2. Player Analysis** | `6_Player_Match_Report.py` | Single-match deep dive: 5 valuation KPIs, 4-column stat breakdown and **5 tactical pitch maps** (*Passes & Carries xT Gain*, *xT Loss* with turnover cost, *Shots*, *Defensive Actions* with `xTP`, and *KDE Touch Heatmap*) |
| **2. Player Analysis** | `7_Player_Match_Log.py` | Match-by-match season trajectory across the 5 valuation models (within-player percentiles, Z-score or raw), intra-player Spearman matrix, and pass/carry & offensive/defensive decompositions |
| **2. Player Analysis** | `4_Scouting_Radar.py` | 19-slice 4-dimension `Barpolar` pizza profile (**Attack**, **Creation**, **Defense**, **Possession**), KPI cards, `ProgressColumn` breakdown table and 1x1 comparison |
| **2. Player Analysis** | `9_Similar_Players.py` | **Euclidean/Cosine k-NN** similarity with 1x1 pizza overlay and attribute delta breakdown, **7 Tactical Archetypes**, and **PCA Biplot & K-Means** clustering diagnostics |
| **2. Player Analysis** | `2_Shot_Map.py` | Season shot map on a vertical half-pitch filtered by competition, team or player (`Goals - xG`) |
| **3. Season Analysis** | `8_Valuation_Comparison.py` | Conceptual taxonomy of the 5 valuation models, positional bias (`mean_percentile_by_group`), Spearman correlation heatmaps (with/without GK), **Rank Shifts** explorer and per-90 leaderboards |
| **3. Season Analysis** | `5_League_Explorer.py` | Interactive quadrant scatter plot (with sample mean lines) of any two per-90 metrics and full sortable player table (`height=600`) |

---

## Quickstart & Reproduction

### 1. Environment & Data Configuration
```bash
uv sync --extra dev          # creates .venv and installs dependencies from uv.lock
cp .env.example .env         # set STATSBOMB_DATA_DIR to your local statsbomb/open-data clone
```

### 2. Run the Pipeline (Phases 0–4)
```bash
# Phase 0: verify 2015/16 competition coverage
uv run python scripts/phase0_coverage.py

# Phase 1: parse JSON -> Parquet tables and partitioned events in data/processed/
uv run python scripts/phase1_etl.py

# Phase 2: run the 5-model action-valuation comparison report
uv run python scripts/phase2_comparison.py

# Phase 3: build season & per-match player tables, shots table and 12x16 xT grid (~2 min)
uv run python scripts/phase3_build_player_metrics.py

# Phase 4 (optional): export a ~12 MB lightweight bundle for cloud deployment
uv run python scripts/phase4_build_deploy_subset.py --matches-per-competition 3
```

### 3. Launch the Streamlit App
```bash
uv run streamlit run app/Home.py
```

### 4. Quality & Test Suite
```bash
./.venv/bin/ruff check app/ src/ tests/ scripts/
./.venv/bin/ruff format --check app/ src/ tests/ scripts/
./.venv/bin/pytest
```

---

## Documentation & Decision Log (`D1–D31`)

- [Phase 0 — Coverage Survey](docs/phase0_coverage.md): 2015/16 match counts across Premier League, La Liga, Serie A, Bundesliga, Ligue 1 and Champions League.
- [Phase 1 — ETL & DuckDB Storage](docs/phase1_etl.md): Parquet schema, Hive partitioning and read-only DuckDB views.
- [Phase 2 — Metrics & Action Valuation (`D1–D16`)](docs/phase2_metrics.md): Mathematical formulation, validation metrics (Brier, AUC, calibration) and Spearman rank correlations across Singh xT, xGChain, xGBuildup, Action xT and VAEP.
- [Phase 3 — Streamlit App (`D17–D25`)](docs/phase3_app.md): UI architecture, match/player/season granularity and tactical pitch maps.
- [Phase 4 — Similarity, Archetypes & Deploy Subset (`D26–D31`)](docs/phase4_similarity.md): Euclidean/Cosine k-NN, domain archetypes, PCA/K-Means diagnostics and lightweight deployment subset.

---

## Methodological Limitations & Risks

- **Single season (`2015/16`) & uneven Champions League coverage**: Only the final is present for the Champions League in StatsBomb Open Data, so it is excluded from aggregate season tables.
- **Small-sample variance**: Per-90 rates are noisy under ~300–900 minutes; all season tables and radars expose an interactive minimum-minutes filter (default `300 min`).
- **Team-style bias**: Possession-heavy teams inflate volume-based progression metrics (`passes_p90`, `xt_p90`, `axt_p90`, `xgchain_p90`, `xgbuildup_p90`). Using **Cosine similarity** helps isolate proportional style from raw volume.
- **Event-data grain**: Off-ball positioning, defensive screen prevention without on-ball contact, and physical tracking speeds are unobserved in event data.
- **Uncalibrated cross-league pools**: Percentiles across all five leagues combined do not adjust for differences in league strength or tactical pace.
