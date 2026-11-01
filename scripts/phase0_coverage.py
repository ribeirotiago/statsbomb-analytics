"""Phase 0: generate the StatsBomb Open Data coverage report (docs/phase0_coverage.md)."""

from __future__ import annotations

import argparse
from pathlib import Path

from statsbomb_analytics.config import PROJECT_ROOT, get_settings
from statsbomb_analytics.ingest.coverage import compute_coverage, to_markdown


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=get_settings().statsbomb_data_dir)
    parser.add_argument("--season-name", default="2015/2016", help="Use '' for all seasons.")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "docs" / "phase0_coverage.md")
    args = parser.parse_args()

    rows = compute_coverage(args.data_dir, args.season_name or None)
    table = to_markdown(rows)
    print(table)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "# Phase 0 — StatsBomb Open Data Coverage\n\n"
        f"Source: `{args.data_dir}` (season: `{args.season_name or 'all'}`).\n\n{table}",
        encoding="utf-8",
    )
    print(f"Report saved to {args.output}")


if __name__ == "__main__":
    main()
