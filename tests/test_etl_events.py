from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from statsbomb_analytics.etl.events import (
    EVENT_COLUMNS,
    build_match_events,
    flatten_event,
    write_events,
)
from statsbomb_analytics.etl.tables import write_parquet
from statsbomb_analytics.storage.duckdb_store import connect, query

PASS = {
    "id": "p1",
    "index": 1,
    "period": 1,
    "minute": 0,
    "second": 3,
    "timestamp": "00:00:03.000",
    "type": {"id": 30, "name": "Pass"},
    "possession": 1,
    "possession_team": {"id": 1, "name": "A"},
    "play_pattern": {"id": 1, "name": "Regular Play"},
    "team": {"id": 1, "name": "A"},
    "player": {"id": 7, "name": "P"},
    "position": {"id": 9, "name": "Right Defensive Midfield"},
    "location": [48.3, 43.6],
    "under_pressure": True,
    "pass": {
        "recipient": {"id": 8, "name": "Q"},
        "length": 12.5,
        "angle": 0.3,
        "height": {"id": 1, "name": "Ground Pass"},
        "end_location": [60.0, 40.0],
        "body_part": {"id": 40, "name": "Right Foot"},
        "cross": True,
    },
}
SHOT = {
    "id": "s1",
    "index": 2,
    "period": 1,
    "minute": 10,
    "second": 0,
    "timestamp": "00:10:00.000",
    "type": {"id": 16, "name": "Shot"},
    "possession": 2,
    "possession_team": {"id": 1, "name": "A"},
    "play_pattern": {"id": 1, "name": "Regular Play"},
    "team": {"id": 1, "name": "A"},
    "player": {"id": 7, "name": "P"},
    "position": {"id": 23, "name": "Center Forward"},
    "location": [110.0, 40.0],
    "shot": {
        "statsbomb_xg": 0.31,
        "end_location": [120.0, 38.0, 1.2],
        "outcome": {"id": 97, "name": "Goal"},
        "body_part": {"id": 40, "name": "Right Foot"},
        "technique": {"id": 93, "name": "Normal"},
        "type": {"id": 87, "name": "Open Play"},
        "first_time": True,
    },
}
SYSTEM = {  # e.g. "Half Start": no player, position or location
    "id": "h1",
    "index": 3,
    "period": 1,
    "minute": 0,
    "second": 0,
    "timestamp": "00:00:00.000",
    "type": {"id": 18, "name": "Half Start"},
    "possession": 1,
    "possession_team": {"id": 1, "name": "A"},
    "play_pattern": {"id": 1, "name": "Regular Play"},
    "team": {"id": 1, "name": "A"},
}


def test_flatten_event_pass_fields() -> None:
    row = flatten_event(10, PASS)
    assert set(row) == set(EVENT_COLUMNS)
    assert row["type"] == "Pass" and row["pass_recipient_id"] == 8
    assert (row["x"], row["y"]) == (48.3, 43.6)
    assert (row["pass_end_x"], row["pass_end_y"]) == (60.0, 40.0)
    assert row["pass_outcome"] is None  # completed pass
    assert row["pass_cross"] is True and row["pass_switch"] is None
    assert row["under_pressure"] is True


def test_flatten_event_shot_fields() -> None:
    row = flatten_event(10, SHOT)
    assert row["shot_xg"] == 0.31 and row["shot_outcome"] == "Goal"
    assert row["shot_end_z"] == 1.2 and row["shot_first_time"] is True


def test_flatten_event_without_player_or_location() -> None:
    row = flatten_event(10, SYSTEM)
    assert row["player_id"] is None and row["x"] is None and row["position"] is None


def _build_dir(tmp_path: Path) -> Path:
    (tmp_path / "matches" / "1").mkdir(parents=True)
    (tmp_path / "events").mkdir()
    comps = [
        {"competition_id": 1, "season_id": 9, "season_name": "2015/2016"},
        {"competition_id": 2, "season_id": 3, "season_name": "2018"},
    ]
    (tmp_path / "competitions.json").write_text(json.dumps(comps))
    (tmp_path / "matches" / "1" / "9.json").write_text(
        json.dumps([{"match_id": 10}, {"match_id": 11}])  # 11 has no events file
    )
    (tmp_path / "events" / "10.json").write_text(json.dumps([PASS, SHOT, SYSTEM]))
    return tmp_path


def test_build_match_events_dtypes_and_missing_file(tmp_path: Path) -> None:
    data = _build_dir(tmp_path)
    df = build_match_events(data, 10)
    assert list(df.columns) == EVENT_COLUMNS and len(df) == 3
    assert str(df["shot_xg"].dtype) == "Float64"
    assert build_match_events(data, 11).empty


def test_write_events_partitioned_and_idempotent(tmp_path: Path) -> None:
    data, out = _build_dir(tmp_path / "src"), tmp_path / "out"
    counts = write_events(data, out, "2015/2016")
    assert counts == {(1, 9): 3}
    assert write_events(data, out, "2015/2016") == counts  # rebuild replaces the partition
    files = list((out / "events").rglob("*.parquet"))
    assert len(files) == 1 and "competition_id=1" in files[0].as_posix()


def test_duckdb_views_expose_events_and_partitions(tmp_path: Path) -> None:
    data, out = _build_dir(tmp_path / "src"), tmp_path / "out"
    write_events(data, out, "2015/2016")
    write_parquet({"matches": pd.DataFrame({"match_id": [10]})}, out)
    df = query(
        out,
        "SELECT competition_id, season_id, count(*) AS n, sum(shot_xg) AS xg "
        "FROM events WHERE match_id = ? GROUP BY ALL",
        [10],
    )
    assert df.iloc[0].tolist() == [1, 9, 3, 0.31]
    con = connect(out)
    assert con.execute("SELECT count(*) FROM matches").fetchone() == (1,)
    con.close()


def test_connect_without_data_has_no_views(tmp_path: Path) -> None:
    con = connect(tmp_path)
    assert con.execute("SELECT count(*) FROM information_schema.tables").fetchone() == (0,)
    con.close()
