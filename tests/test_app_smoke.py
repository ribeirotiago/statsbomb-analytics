"""Smoke tests: every page runs without exceptions (skipped without processed data)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from statsbomb_analytics.config import get_settings
from statsbomb_analytics.metrics.player_metrics import PLAYER_METRICS_FILE

APP_DIR = Path(__file__).parents[1] / "app"
# `streamlit run app/Home.py` puts `app/` on the path; AppTest does not.
sys.path.insert(0, str(APP_DIR))
PAGES = ["Home.py"] + [f"pages/{p.name}" for p in sorted((APP_DIR / "pages").glob("*.py"))]

pytestmark = pytest.mark.skipif(
    not (get_settings().processed_dir / PLAYER_METRICS_FILE).exists(),
    reason="processed data not built",
)


@pytest.mark.parametrize("page", PAGES)
def test_page_runs_without_exception(page: str) -> None:
    at = AppTest.from_file(str(APP_DIR / page), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]


@pytest.mark.parametrize("group", ["FWD", "MID", "DEF"])
def test_radar_runs_for_every_group_in_compare_mode(group: str) -> None:
    at = AppTest.from_file(str(APP_DIR / "pages/4_Scouting_Radar.py"), default_timeout=120).run()
    at.selectbox[2].select(group).run()
    at.toggle[0].set_value(True).run()
    assert not at.exception, [e.value for e in at.exception]


def test_radar_team_filter() -> None:
    at = AppTest.from_file(str(APP_DIR / "pages/4_Scouting_Radar.py"), default_timeout=120).run()
    team = at.selectbox[1].options[1]
    at.selectbox[1].select(team).run()
    assert not at.exception, [e.value for e in at.exception]


def test_match_report_team_filter() -> None:
    at = AppTest.from_file(str(APP_DIR / "pages/1_Match_Report.py"), default_timeout=120).run()
    team = at.selectbox[1].options[1]
    at.selectbox[1].select(team).run()
    assert not at.exception, [e.value for e in at.exception]


def test_pass_network_for_second_team() -> None:
    from statsbomb_analytics.storage.duckdb_store import query

    matches = query(get_settings().processed_dir, "SELECT * FROM matches ORDER BY match_date")
    first = matches[matches["competition_name"] == "1. Bundesliga"].iloc[0]
    at = AppTest.from_file(str(APP_DIR / "pages/3_Pass_Network.py"), default_timeout=120).run()
    at.radio[0].set_value(int(first["away_team_id"])).run()
    assert not at.exception, [e.value for e in at.exception]
