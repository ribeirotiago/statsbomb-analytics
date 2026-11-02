from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from statsbomb_analytics.etl.tables import (
    build_tables,
    flatten_lineup,
    flatten_match,
    flatten_teams,
    write_parquet,
)


def _match(mid: int = 10) -> dict:
    return {
        "match_id": mid,
        "match_date": "2015-08-08",
        "kick_off": "17:00:00.000",
        "competition": {"competition_id": 1, "competition_name": "League X"},
        "season": {"season_id": 9, "season_name": "2015/2016"},
        "home_team": {
            "home_team_id": 1,
            "home_team_name": "A",
            "home_team_gender": "male",
            "country": {"id": 1, "name": "Spain"},
        },
        "away_team": {"away_team_id": 2, "away_team_name": "B", "away_team_gender": "male"},
        "home_score": 2,
        "away_score": 1,
        "match_week": 1,
        "competition_stage": {"id": 1, "name": "Regular Season"},
        "stadium": {"id": 5, "name": "Arena"},
    }


def _lineup() -> list[dict]:
    pos = {
        "position_id": 12,
        "position": "Right Midfield",
        "from": "00:00",
        "to": "78:15",
        "from_period": 1,
        "to_period": 2,
        "start_reason": "Starting XI",
        "end_reason": "Substitution - Off (Tactical)",
    }
    player = {
        "player_id": 7,
        "player_name": "Full Name",
        "player_nickname": "Nick",
        "jersey_number": 14,
        "country": {"id": 31, "name": "Brazil"},
        "cards": [{"card_type": "Yellow Card"}],
        "positions": [pos],
    }
    return [{"team_id": 1, "team_name": "A", "lineup": [player]}]


def test_flatten_match_handles_missing_optional_fields() -> None:
    row = flatten_match(_match())
    assert row["home_team_id"] == 1 and row["away_team_id"] == 2
    assert row["stage"] == "Regular Season"
    assert row["stadium"] == "Arena"
    assert row["referee"] is None


def test_flatten_teams_returns_both_sides() -> None:
    teams = flatten_teams(_match())
    assert [t["team_id"] for t in teams] == [1, 2]
    assert teams[0]["country"] == "Spain" and teams[1]["country"] is None


def test_flatten_lineup_rows() -> None:
    players, lineup, positions = flatten_lineup(10, _lineup()[0])
    assert players[0]["player_nickname"] == "Nick"
    assert lineup[0]["n_cards"] == 1 and lineup[0]["jersey_number"] == 14
    assert positions[0]["from_time"] == "00:00" and positions[0]["position"] == "Right Midfield"


def _build_dir(tmp_path: Path) -> Path:
    (tmp_path / "matches" / "1").mkdir(parents=True)
    (tmp_path / "matches" / "2").mkdir()
    (tmp_path / "lineups").mkdir()
    comps = [
        {"competition_id": 1, "season_id": 9, "season_name": "2015/2016"},
        {"competition_id": 2, "season_id": 3, "season_name": "2018"},
    ]
    (tmp_path / "competitions.json").write_text(json.dumps(comps))
    # the same match twice must not duplicate rows; second competition is filtered out
    (tmp_path / "matches" / "1" / "9.json").write_text(json.dumps([_match(10), _match(10)]))
    (tmp_path / "matches" / "2" / "3.json").write_text(json.dumps([_match(99)]))
    (tmp_path / "lineups" / "10.json").write_text(json.dumps(_lineup()))
    return tmp_path


def test_build_tables_filters_season_and_deduplicates(tmp_path: Path) -> None:
    tables = build_tables(_build_dir(tmp_path), "2015/2016")
    assert tables["matches"]["match_id"].tolist() == [10]
    assert tables["teams"]["team_id"].tolist() == [1, 2]
    assert tables["players"]["player_id"].tolist() == [7]
    assert len(tables["lineups"]) == 1 and len(tables["lineup_positions"]) == 1


def test_build_tables_without_filter_includes_all(tmp_path: Path) -> None:
    tables = build_tables(_build_dir(tmp_path), None)
    assert sorted(tables["matches"]["match_id"]) == [10, 99]


def test_write_parquet_roundtrip(tmp_path: Path) -> None:
    tables = build_tables(_build_dir(tmp_path / "src"), "2015/2016")
    paths = write_parquet(tables, tmp_path / "out")
    assert set(paths) == set(tables)
    back = pd.read_parquet(paths["matches"])
    assert back["match_id"].tolist() == [10]
