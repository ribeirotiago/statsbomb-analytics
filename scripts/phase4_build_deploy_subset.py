"""Phase 4: build a lightweight (~12 MB) processed subset for cloud deployment (git-ignored)."""

from __future__ import annotations

import argparse
from pathlib import Path

from statsbomb_analytics.config import get_settings
from statsbomb_analytics.storage.deploy import build_deploy_subset


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=settings.processed_dir,
        help="Full processed data directory (default: data/processed)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=settings.processed_dir.parent / "deploy_subset",
        help="Target directory for the lightweight deploy bundle (default: data/deploy_subset)",
    )
    parser.add_argument(
        "--matches-per-competition",
        type=int,
        default=3,
        help="Number of featured matches per competition to keep full event partitions for",
    )
    args = parser.parse_args()

    summary = build_deploy_subset(
        args.source,
        args.out,
        matches_per_competition=args.matches_per_competition,
    )
    print(
        f"Deploy subset built in {args.out}: "
        f"{summary['event_matches']}/{summary['total_matches']} event matches "
        f"({summary['files_copied']} files, {summary['total_size_mb']:.2f} MB)"
    )


if __name__ == "__main__":
    main()
