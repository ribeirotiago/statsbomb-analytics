"""Phase 1 — flatten StatsBomb matches/lineups into tidy tables and write Parquet.

Pure functions (flatten_*) are separated from I/O (load_*, write_parquet) for testability.
The source directory is read-only; outputs go to `processed_dir`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

MATCH_COLUMNS = [
    "match_id",
    "competition_id",
    "season_id",
    "competition_name",
    "season_name",
    "match_date",
    "kick_off",
    "match_week",
    "stage",
    "home_team_id",
    "away_team_id",
    "home_score",
    "away_score",
    "stadium",
    "referee",
]
TEAM_COLUMNS = ["team_id", "team_name", "gender", "country"]
PLAYER_COLUMNS = ["player_id", "player_name", "player_nickname", "country"]
LINEUP_COLUMNS = ["match_id", "team_id", "player_id", "jersey_number", "n_cards"]
POSITION_COLUMNS = [
    "match_id",
    "team_id",
    "player_id",
    "position_id",
    "position",
    "from_time",
    "to_time",
    "from_period",
    "to_period",
    "start_reason",
    "end_reason",
]


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _name(obj: dict[str, Any] | None, key: str = "name") -> str | None:
    return obj.get(key) if obj else None


def flatten_match(m: dict[str, Any]) -> dict[str, Any]:
    """One row per match."""
    return {
        "match_id": m["match_id"],
        "competition_id": m["competition"]["competition_id"],
        "season_id": m["season"]["season_id"],
        "competition_name": m["competition"]["competition_name"],
        "season_name": m["season"]["season_name"],
        "match_date": m["match_date"],
        "kick_off": m.get("kick_off"),
        "match_week": m.get("match_week"),
        "stage": _name(m.get("competition_stage")),
        "home_team_id": m["home_team"]["home_team_id"],
        "away_team_id": m["away_team"]["away_team_id"],
        "home_score": m.get("home_score"),
        "away_score": m.get("away_score"),
        "stadium": _name(m.get("stadium")),
        "referee": _name(m.get("referee")),
    }


def flatten_teams(m: dict[str, Any]) -> list[dict[str, Any]]:
    """Both teams of a match (deduplicated later)."""
    rows = []
    for side in ("home", "away"):
        t = m[f"{side}_team"]
        rows.append(
            {
                "team_id": t[f"{side}_team_id"],
                "team_name": t[f"{side}_team_name"],
                "gender": t.get(f"{side}_team_gender"),
                "country": _name(t.get("country")),
            }
        )
    return rows


def flatten_lineup(
    match_id: int, entry: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Flatten one team's lineup into (players, lineup rows, position stints)."""
    team_id = entry["team_id"]
    players, lineup, positions = [], [], []
    for p in entry.get("lineup", []):
        players.append(
            {
                "player_id": p["player_id"],
                "player_name": p["player_name"],
                "player_nickname": p.get("player_nickname"),
                "country": _name(p.get("country")),
            }
        )
        lineup.append(
            {
                "match_id": match_id,
                "team_id": team_id,
                "player_id": p["player_id"],
                "jersey_number": p.get("jersey_number"),
                "n_cards": len(p.get("cards", [])),
            }
        )
        for pos in p.get("positions", []):
            positions.append(
                {
                    "match_id": match_id,
                    "team_id": team_id,
                    "player_id": p["player_id"],
                    "position_id": pos.get("position_id"),
                    "position": pos.get("position"),
                    "from_time": pos.get("from"),
                    "to_time": pos.get("to"),
                    "from_period": pos.get("from_period"),
                    "to_period": pos.get("to_period"),
                    "start_reason": pos.get("start_reason"),
                    "end_reason": pos.get("end_reason"),
                }
            )
    return players, lineup, positions


def _frame(rows: list[dict[str, Any]], columns: list[str], key: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=columns)
    return df.drop_duplicates(subset=key).sort_values(key).reset_index(drop=True)


def build_tables(data_dir: Path, season_name: str | None = None) -> dict[str, pd.DataFrame]:
    """Build matches, teams, players, lineups and lineup_positions (read-only on `data_dir`)."""
    matches, teams, players, lineups, positions = [], [], [], [], []
    for comp in _read_json(data_dir / "competitions.json"):
        if season_name and comp["season_name"] != season_name:
            continue
        path = data_dir / "matches" / str(comp["competition_id"]) / f"{comp['season_id']}.json"
        if not path.exists():
            continue
        for m in _read_json(path):
            matches.append(flatten_match(m))
            teams.extend(flatten_teams(m))
            lineup_path = data_dir / "lineups" / f"{m['match_id']}.json"
            if not lineup_path.exists():
                continue
            for entry in _read_json(lineup_path):
                p, ln, pos = flatten_lineup(m["match_id"], entry)
                players.extend(p)
                lineups.extend(ln)
                positions.extend(pos)
    return {
        "matches": _frame(matches, MATCH_COLUMNS, ["match_id"]),
        "teams": _frame(teams, TEAM_COLUMNS, ["team_id"]),
        "players": _frame(players, PLAYER_COLUMNS, ["player_id"]),
        "lineups": _frame(lineups, LINEUP_COLUMNS, ["match_id", "team_id", "player_id"]),
        "lineup_positions": _frame(
            positions,
            POSITION_COLUMNS,
            ["match_id", "player_id", "from_period", "from_time", "position_id"],
        ),
    }


def write_parquet(tables: dict[str, pd.DataFrame], out_dir: Path) -> dict[str, Path]:
    """Write each table to `out_dir/<name>.parquet` (overwrites: deterministic rebuild)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, df in tables.items():
        paths[name] = out_dir / f"{name}.parquet"
        df.to_parquet(paths[name], index=False)
    return paths
