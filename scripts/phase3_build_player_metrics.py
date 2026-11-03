"""Phase 3: build the player table read by the Streamlit app (small, git-ignored Parquet)."""

from __future__ import annotations

import argparse
from pathlib import Path

from statsbomb_analytics.config import get_settings
from statsbomb_analytics.metrics.player_metrics import PLAYER_METRICS_FILE, build_player_metrics


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=settings.processed_dir / PLAYER_METRICS_FILE)
    args = parser.parse_args()

    table = build_player_metrics(settings.processed_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.out, index=False)
    print(f"{len(table)} players -> {args.out}")


if __name__ == "__main__":
    main()
