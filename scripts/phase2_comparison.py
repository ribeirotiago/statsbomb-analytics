"""Phase 2: compare Singh xT, xGChain/xGBuildup, action-based xT and VAEP (decision D9)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from statsbomb_analytics.config import get_settings
from statsbomb_analytics.metrics.comparison import (
    MIN_MINUTES,
    build_comparison_table,
    filter_minutes,
    mean_percentile_by_group,
    outfield_players,
    rank_shifts,
    spearman_by_group,
    spearman_matrix,
    top_players,
)

SHIFT_PAIRS = (("VAEP", "Action xT"), ("Action xT", "VAEP"), ("Singh xT", "xGChain"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-minutes", type=float, default=MIN_MINUTES)
    parser.add_argument("--out", type=Path, default=None, help="Optional Parquet of the table.")
    args = parser.parse_args()

    table = build_comparison_table(get_settings().processed_dir)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(args.out)
    sample = filter_minutes(table, args.min_minutes)
    pd.set_option("display.width", 200)
    print(f"Players (>= {args.min_minutes:.0f} min): {len(sample)}\n")
    print("Spearman (per 90):\n", spearman_matrix(sample).round(2), "\n")
    outfield = outfield_players(sample)
    print(
        f"Spearman, outfield only ({len(outfield)} players):\n",
        spearman_matrix(outfield).round(2),
        "\n",
    )
    print("Mean percentile by position group:\n", mean_percentile_by_group(sample).round(2), "\n")
    print("Spearman by position group:\n", spearman_by_group(sample).round(2).to_string(), "\n")
    for name, top in top_players(sample, n=8).items():
        print(f"Top 8 {name} per 90:\n{top.round(3).to_string()}\n")
    for a, b in SHIFT_PAIRS:
        print(f"Ranked higher by {a} than by {b}:\n{rank_shifts(sample, a, b).round(2)}\n")


if __name__ == "__main__":
    main()
