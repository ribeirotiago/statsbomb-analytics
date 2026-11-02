"""Phase 1: build matches/teams/players/lineups Parquet tables from StatsBomb Open Data."""

from __future__ import annotations

import argparse
from pathlib import Path

from statsbomb_analytics.config import get_settings
from statsbomb_analytics.etl.tables import build_tables, write_parquet


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=settings.statsbomb_data_dir)
    parser.add_argument("--out-dir", type=Path, default=settings.processed_dir)
    parser.add_argument("--season-name", default="2015/2016", help="Use '' for all seasons.")
    args = parser.parse_args()

    tables = build_tables(args.data_dir, args.season_name or None)
    for name, path in write_parquet(tables, args.out_dir).items():
        print(f"{name}: {len(tables[name])} rows -> {path}")


if __name__ == "__main__":
    main()
