"""Phase 1 — flatten StatsBomb events into a tidy table, partitioned Parquet output.

One row per event with the nested fields needed by Phase 2/3 (xG, xT, PPDA, per-90, shot map,
pass network). Output layout (Hive style, read by DuckDB with `hive_partitioning`):

    <out_dir>/events/competition_id=<c>/season_id=<s>/events.parquet

Pass `outcome` is NULL for completed passes (StatsBomb convention). Coordinates use the
StatsBomb 120x80 pitch.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

# column -> pandas dtype (explicit so every partition has the same schema)
EVENT_DTYPES: dict[str, str] = {
    "event_id": "string",
    "match_id": "Int64",
    "index": "Int64",
    "period": "Int64",
    "minute": "Int64",
    "second": "Int64",
    "timestamp": "string",
    "type": "string",
    "team_id": "Int64",
    "player_id": "Int64",
    "position": "string",
    "possession": "Int64",
    "possession_team_id": "Int64",
    "play_pattern": "string",
    "x": "Float64",
    "y": "Float64",
    "duration": "Float64",
    "under_pressure": "boolean",
    "counterpress": "boolean",
    "pass_recipient_id": "Int64",
    "pass_length": "Float64",
    "pass_angle": "Float64",
    "pass_height": "string",
    "pass_end_x": "Float64",
    "pass_end_y": "Float64",
    "pass_body_part": "string",
    "pass_type": "string",
    "pass_outcome": "string",
    "pass_cross": "boolean",
    "pass_switch": "boolean",
    "pass_through_ball": "boolean",
    "pass_shot_assist": "boolean",
    "pass_goal_assist": "boolean",
    "carry_end_x": "Float64",
    "carry_end_y": "Float64",
    "shot_xg": "Float64",
    "shot_end_x": "Float64",
    "shot_end_y": "Float64",
    "shot_end_z": "Float64",
    "shot_outcome": "string",
    "shot_body_part": "string",
    "shot_technique": "string",
    "shot_type": "string",
    "shot_first_time": "boolean",
    "dribble_outcome": "string",
    "duel_type": "string",
    "duel_outcome": "string",
    "interception_outcome": "string",
    "ball_recovery_failure": "boolean",
    "foul_committed_card": "string",
    "goalkeeper_type": "string",
    "goalkeeper_outcome": "string",
}
EVENT_COLUMNS = list(EVENT_DTYPES)


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _name(obj: dict[str, Any] | None) -> str | None:
    return obj.get("name") if obj else None


def _id(obj: dict[str, Any] | None) -> int | None:
    return obj.get("id") if obj else None


def _coord(loc: list[float] | None, i: int) -> float | None:
    return loc[i] if loc is not None and len(loc) > i else None


def _flag(d: dict[str, Any], key: str) -> bool | None:
    """Boolean flags are present only when true in the source; absent -> NULL."""
    return True if d.get(key) else None


def flatten_event(match_id: int, e: dict[str, Any]) -> dict[str, Any]:
    """Flatten one event into a dict keyed by `EVENT_COLUMNS`."""
    loc = e.get("location")
    p = e.get("pass") or {}
    c = e.get("carry") or {}
    s = e.get("shot") or {}
    pe, ce, se = p.get("end_location"), c.get("end_location"), s.get("end_location")
    fc = e.get("foul_committed") or {}
    gk = e.get("goalkeeper") or {}
    return {
        "event_id": e["id"],
        "match_id": match_id,
        "index": e.get("index"),
        "period": e.get("period"),
        "minute": e.get("minute"),
        "second": e.get("second"),
        "timestamp": e.get("timestamp"),
        "type": _name(e.get("type")),
        "team_id": _id(e.get("team")),
        "player_id": _id(e.get("player")),
        "position": _name(e.get("position")),
        "possession": e.get("possession"),
        "possession_team_id": _id(e.get("possession_team")),
        "play_pattern": _name(e.get("play_pattern")),
        "x": _coord(loc, 0),
        "y": _coord(loc, 1),
        "duration": e.get("duration"),
        "under_pressure": _flag(e, "under_pressure"),
        "counterpress": _flag(e, "counterpress"),
        "pass_recipient_id": _id(p.get("recipient")),
        "pass_length": p.get("length"),
        "pass_angle": p.get("angle"),
        "pass_height": _name(p.get("height")),
        "pass_end_x": _coord(pe, 0),
        "pass_end_y": _coord(pe, 1),
        "pass_body_part": _name(p.get("body_part")),
        "pass_type": _name(p.get("type")),
        "pass_outcome": _name(p.get("outcome")),
        "pass_cross": _flag(p, "cross"),
        "pass_switch": _flag(p, "switch"),
        "pass_through_ball": _flag(p, "through_ball"),
        "pass_shot_assist": _flag(p, "shot_assist"),
        "pass_goal_assist": _flag(p, "goal_assist"),
        "carry_end_x": _coord(ce, 0),
        "carry_end_y": _coord(ce, 1),
        "shot_xg": s.get("statsbomb_xg"),
        "shot_end_x": _coord(se, 0),
        "shot_end_y": _coord(se, 1),
        "shot_end_z": _coord(se, 2),
        "shot_outcome": _name(s.get("outcome")),
        "shot_body_part": _name(s.get("body_part")),
        "shot_technique": _name(s.get("technique")),
        "shot_type": _name(s.get("type")),
        "shot_first_time": _flag(s, "first_time"),
        "dribble_outcome": _name((e.get("dribble") or {}).get("outcome")),
        "duel_type": _name((e.get("duel") or {}).get("type")),
        "duel_outcome": _name((e.get("duel") or {}).get("outcome")),
        "interception_outcome": _name((e.get("interception") or {}).get("outcome")),
        "ball_recovery_failure": _flag(e.get("ball_recovery") or {}, "recovery_failure"),
        "foul_committed_card": _name(fc.get("card")),
        "goalkeeper_type": _name(gk.get("type")),
        "goalkeeper_outcome": _name(gk.get("outcome")),
    }


def events_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Build a DataFrame with the fixed column order and dtypes."""
    df = pd.DataFrame(rows, columns=EVENT_COLUMNS)
    return df.astype(EVENT_DTYPES)


def build_match_events(data_dir: Path, match_id: int) -> pd.DataFrame:
    """Events of one match (empty frame if the file is missing)."""
    path = data_dir / "events" / f"{match_id}.json"
    if not path.exists():
        return events_frame([])
    return events_frame([flatten_event(match_id, e) for e in _read_json(path)])


def write_events(
    data_dir: Path, out_dir: Path, season_name: str | None = None
) -> dict[tuple[int, int], int]:
    """Write one Parquet per competition/season; returns {(competition_id, season_id): rows}.

    Existing partitions of the processed competition/seasons are replaced (deterministic rebuild).
    Memory stays bounded to one competition/season at a time.
    """
    counts: dict[tuple[int, int], int] = {}
    for comp in _read_json(data_dir / "competitions.json"):
        if season_name and comp["season_name"] != season_name:
            continue
        cid, sid = comp["competition_id"], comp["season_id"]
        mpath = data_dir / "matches" / str(cid) / f"{sid}.json"
        if not mpath.exists():
            continue
        frames = [build_match_events(data_dir, m["match_id"]) for m in _read_json(mpath)]
        frames = [f for f in frames if not f.empty]
        if not frames:
            continue
        df = pd.concat(frames, ignore_index=True).drop_duplicates("event_id")
        part = out_dir / "events" / f"competition_id={cid}" / f"season_id={sid}"
        if part.exists():
            shutil.rmtree(part)
        part.mkdir(parents=True)
        df.to_parquet(part / "events.parquet", index=False)
        counts[(cid, sid)] = len(df)
    return counts
