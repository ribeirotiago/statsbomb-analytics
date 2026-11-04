"""Phase 3: build the player tables and xT grid read by the Streamlit app (git-ignored)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from statsbomb_analytics.config import get_settings
from statsbomb_analytics.metrics.player_metrics import (
    PLAYER_MATCH_METRICS_FILE,
    PLAYER_METRICS_FILE,
    SHOTS_FILE,
    XT_GRID_FILE,
    build_all_app_artifacts,
    build_shots_table,
)


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=settings.processed_dir / PLAYER_METRICS_FILE)
    args = parser.parse_args()

    out_dir = args.out.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    match_out = out_dir / PLAYER_MATCH_METRICS_FILE
    shots_out = out_dir / SHOTS_FILE
    grid_out = out_dir / XT_GRID_FILE

    season_table, match_table, grid = build_all_app_artifacts(settings.processed_dir)
    shots_table = build_shots_table(settings.processed_dir)
    season_table.to_parquet(args.out, index=False)
    match_table.to_parquet(match_out, index=False)
    shots_table.to_parquet(shots_out, index=False)
    np.save(grid_out, grid)

    print(f"{len(season_table)} season player rows -> {args.out}")
    print(f"{len(match_table)} player-match rows -> {match_out}")
    print(f"{len(shots_table)} shots -> {shots_out}")
    print(f"xT grid {grid.shape} -> {grid_out}")


if __name__ == "__main__":
    main()
