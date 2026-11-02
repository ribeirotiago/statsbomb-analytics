"""Single entry point to the processed data: DuckDB views over the Parquet files.

No database file is persisted: views are created in memory on each connection, so the Parquet
files remain the single source of truth (deterministic, nothing heavy to version).
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

TABLES = ("matches", "teams", "players", "lineups", "lineup_positions")


def connect(processed_dir: Path) -> duckdb.DuckDBPyConnection:
    """In-memory connection with a view per table and `events` (Hive partitions exposed)."""
    con = duckdb.connect(":memory:")
    for name in TABLES:
        path = processed_dir / f"{name}.parquet"
        if path.exists():
            con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{path.as_posix()}')")
    events = processed_dir / "events"
    if events.exists():
        glob = (events / "*" / "*" / "*.parquet").as_posix()
        con.execute(
            f"CREATE VIEW events AS SELECT * FROM read_parquet('{glob}', hive_partitioning = true)"
        )
    return con


def query(processed_dir: Path, sql: str, params: list | None = None) -> pd.DataFrame:
    """Run a read-only SQL query and return a DataFrame."""
    con = connect(processed_dir)
    try:
        return con.execute(sql, params or []).df()
    finally:
        con.close()
