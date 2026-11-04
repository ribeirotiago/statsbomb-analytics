from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from statsbomb_analytics.storage.deploy import (
    DEPLOY_MANIFEST_FILE,
    available_event_match_ids,
    build_deploy_subset,
    select_featured_matches,
)


def _sample_matches() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "match_id": [101, 102, 103, 201, 202],
            "competition_name": ["La Liga", "La Liga", "La Liga", "Serie A", "Serie A"],
            "match_date": [
                "2016-01-10",
                "2016-02-15",
                "2016-03-20",
                "2016-01-12",
                "2016-04-01",
            ],
            "home_team_id": [1, 1, 2, 3, 4],
            "away_team_id": [2, 2, 1, 4, 3],
            "home_score": [1, 4, 2, 0, 3],
            "away_score": [0, 2, 2, 0, 2],
        }
    )


def test_available_event_match_ids_missing_and_present(tmp_path: Path) -> None:
    assert available_event_match_ids(tmp_path) == set()

    part_dir = tmp_path / "events" / "competition_id=11" / "season_id=27"
    part_dir.mkdir(parents=True)
    pd.DataFrame({"match_id": [101], "type": ["Pass"]}).to_parquet(
        part_dir / "match_id=101.parquet", index=False
    )
    pd.DataFrame({"match_id": [202], "type": ["Shot"]}).to_parquet(
        part_dir / "match_id=202.parquet", index=False
    )
    assert available_event_match_ids(tmp_path) == {101, 202}


def test_select_featured_matches_prioritises_high_scoring_per_competition() -> None:
    matches = _sample_matches()
    featured = select_featured_matches(matches, matches_per_competition=1)
    assert set(featured["match_id"]) == {102, 202}

    empty = select_featured_matches(matches.iloc[0:0], matches_per_competition=2)
    assert empty.empty


def test_build_deploy_subset_copies_aggregates_and_selected_events(tmp_path: Path) -> None:
    src = tmp_path / "processed"
    dst = tmp_path / "deploy"
    src.mkdir()

    matches = _sample_matches()
    matches.to_parquet(src / "matches.parquet", index=False)
    pd.DataFrame({"team_id": [1, 2, 3, 4], "team_name": ["A", "B", "C", "D"]}).to_parquet(
        src / "teams.parquet", index=False
    )
    pd.DataFrame({"player_id": [10, 11], "player_name": ["P1", "P2"]}).to_parquet(
        src / "players.parquet", index=False
    )
    pd.DataFrame({"player_id": [10], "minutes": [900.0]}).to_parquet(
        src / "player_metrics.parquet", index=False
    )
    pd.DataFrame({"match_id": [102], "player_id": [10], "minutes": [90.0]}).to_parquet(
        src / "player_match_metrics.parquet", index=False
    )
    np.save(src / "xt_grid.npy", np.zeros((12, 16), dtype=float))

    part_dir = src / "events" / "competition_id=11" / "season_id=27"
    part_dir.mkdir(parents=True)
    for mid in matches["match_id"]:
        pd.DataFrame(
            {
                "match_id": [int(mid)],
                "team_id": [1],
                "player_id": [10],
                "period": [1],
                "minute": [12],
                "type": ["Shot"],
                "x": [105.0],
                "y": [40.0],
                "shot_xg": [0.15],
                "shot_outcome": ["Goal"],
                "shot_type": ["Open Play"],
                "shot_body_part": ["Right Foot"],
                "shot_technique": ["Normal"],
            }
        ).to_parquet(part_dir / f"match_id={mid}.parquet", index=False)

    summary = build_deploy_subset(src, dst, matches_per_competition=1)
    assert summary["total_matches"] == 5
    assert summary["event_matches"] == 2
    assert available_event_match_ids(dst) == {102, 202}
    assert (dst / "shots.parquet").exists()
    assert len(pd.read_parquet(dst / "shots.parquet")) == 5
    assert (dst / DEPLOY_MANIFEST_FILE).exists()
    manifest = json.loads((dst / DEPLOY_MANIFEST_FILE).read_text(encoding="utf-8"))
    assert manifest["event_matches"] == 2
