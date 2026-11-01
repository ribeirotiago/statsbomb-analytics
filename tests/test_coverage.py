from __future__ import annotations

import json
from pathlib import Path

from statsbomb_analytics.ingest.coverage import compute_coverage, to_markdown


def _match(mid: int, home: str, away: str, date: str) -> dict:
    return {
        "match_id": mid,
        "match_date": date,
        "home_team": {"home_team_name": home},
        "away_team": {"away_team_name": away},
    }


def _build(tmp_path: Path) -> Path:
    (tmp_path / "matches" / "1").mkdir(parents=True)
    (tmp_path / "events").mkdir()
    (tmp_path / "lineups").mkdir()
    (tmp_path / "three-sixty").mkdir()
    comps = [
        {
            "competition_id": 1,
            "season_id": 9,
            "competition_name": "League X",
            "season_name": "2015/2016",
        },
        {
            "competition_id": 2,
            "season_id": 9,
            "competition_name": "No Matches",
            "season_name": "2015/2016",
        },
        {
            "competition_id": 1,
            "season_id": 8,
            "competition_name": "League X",
            "season_name": "2014/2015",
        },
    ]
    (tmp_path / "competitions.json").write_text(json.dumps(comps))
    matches = [_match(10, "A", "B", "2015-08-01"), _match(11, "B", "A", "2015-09-01")]
    (tmp_path / "matches" / "1" / "9.json").write_text(json.dumps(matches))
    (tmp_path / "events" / "10.json").write_text("[]")
    (tmp_path / "lineups" / "10.json").write_text("[]")
    (tmp_path / "lineups" / "11.json").write_text("[]")
    return tmp_path


def test_compute_coverage_counts(tmp_path: Path) -> None:
    rows = compute_coverage(_build(tmp_path), "2015/2016")
    assert len(rows) == 1  # competition without a matches file is ignored
    r = rows[0]
    assert (r.matches, r.teams) == (2, 2)
    assert (r.min_matches_per_team, r.max_matches_per_team) == (2, 2)
    assert (r.with_events, r.with_lineups, r.with_360) == (1, 2, 0)
    assert r.pct_events == 50.0
    assert (r.date_min, r.date_max) == ("2015-08-01", "2015-09-01")


def test_to_markdown_has_row(tmp_path: Path) -> None:
    md = to_markdown(compute_coverage(_build(tmp_path), "2015/2016"))
    assert "| League X | 2015/2016 | 2 | 2 |" in md
