# Phase 1 — ETL (StatsBomb JSON → Parquet + DuckDB)

Data: StatsBomb. The source clone (`STATSBOMB_DATA_DIR`) is read-only; outputs go to
`data/processed/` (git-ignored, ~380 MB for 2015/16).

## Decision: DuckDB + Parquet (no ORM)

`atletico-analytics` uses SQLite + SQLAlchemy, which fits a small relational dataset with
incremental scraper writes. Here the data is static, columnar and large (~6.4M events), and the
workload is read-only aggregation, so we use Parquet files queried through DuckDB:

- Parquet is the single source of truth; **no `.duckdb` file is persisted**. Views are created in
  memory on each connection ([`storage/duckdb_store.py`](../src/statsbomb_analytics/storage/duckdb_store.py)).
- Rebuild is deterministic: re-running the ETL replaces the output (no duplicates).
- Layering ideas from `atletico-analytics` are kept (one-way layers, single access point, pure
  functions with tests).

## Outputs

| Table | Grain | Rows (2015/16) |
|---|---|---:|
| `matches` | match | 1,824 |
| `teams` | team | 98 |
| `players` | player | 3,069 |
| `lineups` | match × team × player | 68,688 |
| `lineup_positions` | position stint (player, from/to time) | 69,080 |
| `events/competition_id=*/season_id=*/events.parquet` | event | ~6.4M |

Events columns are listed in `EVENT_DTYPES` ([`etl/events.py`](../src/statsbomb_analytics/etl/events.py)):
core fields plus selected pass, carry, shot, dribble, duel, interception, ball recovery, foul
card and goalkeeper fields. `competition_id` and `season_id` are exposed by Hive partitioning.

## How to run

```bash
./.venv/bin/python scripts/phase1_etl.py                 # 2015/2016 (default), ~4 min, ~2.6 GB RAM
./.venv/bin/python scripts/phase1_etl.py --skip-events   # small tables only
./.venv/bin/python scripts/phase1_etl.py --season-name ''  # all seasons (much heavier)
```

```python
from statsbomb_analytics.config import get_settings
from statsbomb_analytics.storage.duckdb_store import query

query(get_settings().processed_dir, "SELECT type, count(*) FROM events GROUP BY ALL")
```

## Validation

Match counts equal the Phase 0 coverage (1,824). Shot xG sums are close to goals scored from shots
(e.g. Premier League: xG 971 vs 988 goals). Goals counted from shots exclude own goals, so they
are slightly below official league totals.

## Assumptions and limitations

- Pass `outcome` is NULL for completed passes (StatsBomb convention); boolean flags are NULL when
  absent in the source (not `False`).
- Coordinates use the StatsBomb 120×80 pitch; no 360 data in this season.
- Not all event fields are flattened (e.g. shot freeze frames, tactics/lineup events); add them
  when a metric needs them.
- Events are processed one competition/season at a time to bound memory (~2.6 GB peak).
- The Champions League has only 1 match (final); exclude it from aggregate analyses.
