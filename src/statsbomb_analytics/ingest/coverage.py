"""Phase 0 — StatsBomb Open Data coverage survey (local reads only, no network)."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CompetitionCoverage:
    competition_id: int
    season_id: int
    competition_name: str
    season_name: str
    matches: int
    teams: int
    min_matches_per_team: int
    max_matches_per_team: int
    with_events: int
    with_lineups: int
    with_360: int
    date_min: str
    date_max: str

    @property
    def pct_events(self) -> float:
        return 100 * self.with_events / self.matches if self.matches else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "pct_events": round(self.pct_events, 1)}


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def matches_per_team(matches: list[dict[str, Any]]) -> Counter[str]:
    """Count matches per team (home + away)."""
    counter: Counter[str] = Counter()
    for m in matches:
        counter[m["home_team"]["home_team_name"]] += 1
        counter[m["away_team"]["away_team_name"]] += 1
    return counter


def summarize_matches(
    comp: dict[str, Any], matches: list[dict[str, Any]], data_dir: Path
) -> CompetitionCoverage:
    """Summarize coverage of one competition/season from its match list."""
    per_team = matches_per_team(matches)
    ids = [m["match_id"] for m in matches]
    dates = sorted(m["match_date"] for m in matches)
    counts = per_team.values()
    return CompetitionCoverage(
        competition_id=comp["competition_id"],
        season_id=comp["season_id"],
        competition_name=comp["competition_name"],
        season_name=comp["season_name"],
        matches=len(matches),
        teams=len(per_team),
        min_matches_per_team=min(counts, default=0),
        max_matches_per_team=max(counts, default=0),
        with_events=sum((data_dir / "events" / f"{i}.json").exists() for i in ids),
        with_lineups=sum((data_dir / "lineups" / f"{i}.json").exists() for i in ids),
        with_360=sum((data_dir / "three-sixty" / f"{i}.json").exists() for i in ids),
        date_min=dates[0] if dates else "",
        date_max=dates[-1] if dates else "",
    )


def compute_coverage(data_dir: Path, season_name: str | None = None) -> list[CompetitionCoverage]:
    """Coverage for all competitions (optionally filtered by `season_name`)."""
    result: list[CompetitionCoverage] = []
    for comp in _read_json(data_dir / "competitions.json"):
        if season_name and comp["season_name"] != season_name:
            continue
        mpath = data_dir / "matches" / str(comp["competition_id"]) / f"{comp['season_id']}.json"
        if not mpath.exists():
            continue
        result.append(summarize_matches(comp, _read_json(mpath), data_dir))
    return sorted(result, key=lambda c: (c.season_name, c.competition_name))


def to_markdown(rows: list[CompetitionCoverage]) -> str:
    """Markdown coverage table."""
    header = (
        "| Competition | Season | Matches | Teams | Matches/team (min–max) | Events | "
        "Lineups | 360 | Period |\n|---|---|---:|---:|---:|---:|---:|---:|---|\n"
    )
    lines = [
        f"| {r.competition_name} | {r.season_name} | {r.matches} | {r.teams} | "
        f"{r.min_matches_per_team}–{r.max_matches_per_team} | {r.with_events} "
        f"({r.pct_events:.0f}%) | {r.with_lineups} | {r.with_360} | "
        f"{r.date_min} → {r.date_max} |"
        for r in rows
    ]
    return header + "\n".join(lines) + "\n"
