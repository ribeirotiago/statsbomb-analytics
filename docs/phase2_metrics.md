# Phase 2 — Metrics

Metrics are computed from the processed Parquet files (see [phase1_etl.md](phase1_etl.md)),
always through `storage/duckdb_store.py`. Data: StatsBomb Open Data.

## Decision log

| # | Decision | Rationale |
|---|---|---|
| D1 | Minutes played come from `lineup_positions` stints on the cumulative match clock. A null `to_time` means "until the final whistle", approximated by the last event of periods 1–4; the shootout is excluded. | Only source of substitution/red-card timing. Limitation: the match end can be a few seconds short. |
| D2 | **No minimum-minutes threshold in the metrics layer.** The UI exposes a slider (default 300 min, adjustable in real time) that filters on the `minutes` column, as in `atletico-analytics`. | The threshold is an exploration choice, not a property of the metric. |
| D3 | Per-90 = `count / minutes * 90`; zero minutes gives NaN, never inf. | Avoids silent infinities. |
| D4 | Primary position = position with the most minutes per (team, player), grouped into GK/DEF/MID/FWD. | Simple and explainable; blurs hybrid roles. |
| D5 | xG is the StatsBomb value; `npxg` excludes penalties. | We do not train our own xG model in this phase. |
| D6 | xT follows Singh's method re-implemented on StatsBomb coordinates (120 x 80, 12 x 16 grid), solved to convergence. | Baseline already used in `atletico-analytics`. |
| D7 | **Validation fix (xT):** failed moves must enter the fit (denominator of P(move)) and only successful moves create transitions. A first version fitted on successes only and gave deep zones a value of ~0.10 (vs 0.28 in the box) because value was inherited with no turnover cost. After the fix: ~0.005 at the own goal. A regression test covers it. | Sanity check against real data caught a methodological error. |
| D8 | PPDA = opponent passes in their own 60% / team defensive actions (tackles, interceptions, fouls committed, dribbled past) in the opposite 60%, summed over the season. | Standard definition; ignores `Pressure` events (a possible variant). |
| D9 | **Action-valuation comparison:** compare four models on the same players — Singh xT, xGChain/xGBuildup, action-based xT (possession chains, from the Soccermatics course), and VAEP — and document which players each method favours. | Academic goal: understand how the choice of model changes player valuation. |
| D10 | Possession chains use StatsBomb's `possession` id (plus `index`/`timestamp` for order). | Available in the open data; no heuristics needed. |
| D11 | VAEP is implemented in-house with scikit-learn (no new dependency such as `socceraction`), SPADL-like features of the last 3 actions, two classifiers (score/concede within 10 actions). Train/test split **by match**, never by action. | Explainable methodology; avoids leakage. |
| D12 | Baselines before ML: xT and xGChain/xGBuildup first, VAEP last. | Project rule (GEMINI.md §4). |

References used for ideas only (code is rewritten for StatsBomb events):
`football/soccermatics/4_valuingActions` (xT, possession chains, action-based xT) and
`football/fot-valuing-actions-vaep` (VAEP tutorials, Wyscout + `socceraction` 0.2.0).

## Assumptions, limitations and risks

- One season (15/16) and uneven Champions League coverage.
- Per-90 rates are inflated by small samples; filter by minutes (D2).
- Team style biases volume metrics (possession teams accumulate more passes and xT).
- xT: season-specific grid, noisy in rarely visited zones, no per-action turnover penalty.
- PPDA is sensitive to game state and opponent style.
- VAEP (planned): rare labels (goals), defenders undervalued, style bias; to be validated with
  calibration and Brier score.

## Validation notes

- xT grid sanity (full season): max 0.278 (last column), ~0.005 near the own goal, monotonic
  growth towards the opposing goal along the central row.
- PPDA by team ranges from ~8.9 to ~16.7 (plausible for top European leagues).
- xGChain/xGBuildup (min. 300 min): xGChain per 90 is led by attacking creators (Neymar, Messi,
  Di María); xGBuildup per 90 is led by centre-backs and midfielders (Boateng, Kimpembe,
  Kroos), as expected from the definition (D13: the shooter and key passer are excluded from
  xGBuildup; penalties are excluded from possession value).
