"""Lightweight deployment subset builder for hosting without the full 377 MB `events/` tree.

Assumptions and limitations:
- Season-wide and match-log pages (`Home`, `Player Match Log & Valuation`, `Scouting Radar`,
  `Similar Players & Profiles`, `Shot Map & Finishing`, `Action Valuation Comparison`,
  `League Explorer`) only need the pre-aggregated tables (`player_metrics.parquet`,
  `player_match_metrics.parquet`, `shots.parquet`, `matches.parquet`, `teams.parquet`,
  `players.parquet`), which total ~9 MB for the entire 2015/16 season (1,824 matches).
- Single-match pitch event pages (`Match Report`, `Pass Network`, `Player Match Report`) query
  `events/competition_id=.../season_id=.../match_id=<id>.parquet`. Copying a curated subset of
  featured matches (e.g. 3 per competition, ~3 MB) keeps the entire deploy bundle under ~13 MB
  while preserving full functionality across all 9 pages.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pandas as pd

from statsbomb_analytics.metrics.player_metrics import (
    PLAYER_MATCH_METRICS_FILE,
    PLAYER_METRICS_FILE,
    SHOTS_FILE,
    XT_GRID_FILE,
    build_shots_table,
)
from statsbomb_analytics.storage.duckdb_store import TABLES

DEPLOY_MANIFEST_FILE = "deploy_manifest.json"
_MATCH_ID_RE = re.compile(r"match_id=(\d+)\.parquet$")

AGGREGATE_FILES: tuple[str, ...] = (
    *(f"{t}.parquet" for t in TABLES),
    PLAYER_METRICS_FILE,
    PLAYER_MATCH_METRICS_FILE,
    SHOTS_FILE,
    XT_GRID_FILE,
)


def available_event_match_ids(processed_dir: Path) -> set[int]:
    """Return the set of `match_id` integers that have a partition file under `events/`."""
    events_dir = processed_dir / "events"
    if not events_dir.exists():
        return set()
    ids: set[int] = set()
    for path in events_dir.rglob("match_id=*.parquet"):
        match = _MATCH_ID_RE.search(path.name)
        if match:
            ids.add(int(match.group(1)))
    return ids


def select_featured_matches(
    matches: pd.DataFrame, matches_per_competition: int = 3
) -> pd.DataFrame:
    """Deterministically pick up to `matches_per_competition` matches per league.

    Prioritises high-scoring matches (`home_score + away_score`), breaking ties by most recent
    `match_date` and lowest `match_id`.
    """
    if matches.empty or matches_per_competition <= 0:
        return matches.iloc[0:0].copy()
    work = matches.copy()
    work["_total_goals"] = work["home_score"].fillna(0) + work["away_score"].fillna(0)
    work = work.sort_values(
        ["competition_name", "_total_goals", "match_date", "match_id"],
        ascending=[True, False, False, True],
    )
    selected = work.groupby("competition_name", as_index=False).head(int(matches_per_competition))
    return (
        selected.drop(columns=["_total_goals"])
        .sort_values(["match_date", "match_id"])
        .reset_index(drop=True)
    )


def build_deploy_subset(
    source_dir: Path,
    target_dir: Path,
    *,
    matches_per_competition: int = 3,
) -> dict[str, int | float]:
    """Build a self-contained ~12 MB processed directory suitable for cloud deployment."""
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    # Ensure `shots.parquet` exists in target_dir (either copied from source or built from events)
    source_shots = source_dir / SHOTS_FILE
    target_shots = target_dir / SHOTS_FILE
    if not source_shots.exists() and (source_dir / "events").exists():
        shots_df = build_shots_table(source_dir)
        shots_df.to_parquet(target_shots, index=False)

    files_copied = 0
    for filename in AGGREGATE_FILES:
        src = source_dir / filename
        dst = target_dir / filename
        if src.exists():
            shutil.copy2(src, dst)
            files_copied += 1
        elif dst.exists():
            files_copied += 1

    matches_path = source_dir / "matches.parquet"
    total_matches = 0
    event_matches = 0
    if matches_path.exists():
        matches_df = pd.read_parquet(matches_path)
        total_matches = len(matches_df)
        featured = select_featured_matches(
            matches_df, matches_per_competition=matches_per_competition
        )
        keep_ids = set(featured["match_id"].astype(int))

        target_events = target_dir / "events"
        if target_events.exists():
            shutil.rmtree(target_events)

        source_events = source_dir / "events"
        if source_events.exists():
            for path in source_events.rglob("match_id=*.parquet"):
                m = _MATCH_ID_RE.search(path.name)
                if m and int(m.group(1)) in keep_ids:
                    rel = path.relative_to(source_dir)
                    dst = target_dir / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, dst)
                    event_matches += 1
                    files_copied += 1

    total_bytes = sum(p.stat().st_size for p in target_dir.rglob("*") if p.is_file())
    summary: dict[str, int | float] = {
        "total_matches": int(total_matches),
        "event_matches": int(event_matches),
        "matches_per_competition": int(matches_per_competition),
        "files_copied": int(files_copied),
        "total_bytes": int(total_bytes),
        "total_size_mb": round(total_bytes / (1024 * 1024), 2),
    }
    (target_dir / DEPLOY_MANIFEST_FILE).write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary
