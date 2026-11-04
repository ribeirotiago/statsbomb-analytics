"""Cached data access for the Streamlit pages (read-only, Parquet/DuckDB under `data/processed`)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from statsbomb_analytics.config import get_settings
from statsbomb_analytics.metrics.player_metrics import (
    PLAYER_MATCH_METRICS_FILE,
    PLAYER_METRICS_FILE,
    SHOTS_APP_SQL,
    SHOTS_FILE,
    XT_GRID_FILE,
)
from statsbomb_analytics.storage.deploy import available_event_match_ids
from statsbomb_analytics.storage.duckdb_store import query

MISSING_METRICS = (
    "Player tables not found. Build them once with "
    "`uv run python scripts/phase3_build_player_metrics.py` (takes a few minutes)."
)

MATCHES_SQL = """
SELECT m.match_id, m.competition_name, m.match_date, m.match_week, m.stadium, m.referee,
       m.home_team_id, h.team_name AS home_team, m.home_score,
       m.away_team_id, a.team_name AS away_team, m.away_score
FROM matches m
JOIN teams h ON h.team_id = m.home_team_id
JOIN teams a ON a.team_id = m.away_team_id
ORDER BY m.match_date, m.match_id
"""

MATCH_EVENTS_SQL = """
SELECT e.*, p.player_name, t.team_name
FROM events e
LEFT JOIN players p ON p.player_id = e.player_id
LEFT JOIN teams t ON t.team_id = e.team_id
WHERE e.match_id = ?
ORDER BY e.index
"""

SHOTS_SQL = SHOTS_APP_SQL


@st.cache_data(show_spinner=False)
def load_event_match_ids() -> frozenset[int]:
    """Set of `match_id` integers that have an `events` partition file on disk."""
    return frozenset(available_event_match_ids(get_settings().processed_dir))


@st.cache_data(show_spinner=False)
def load_player_metrics() -> pd.DataFrame | None:
    """Player table with per-90 values and the five valuation methods (None if not built)."""
    path = get_settings().processed_dir / PLAYER_METRICS_FILE
    return pd.read_parquet(path) if path.exists() else None


@st.cache_data(show_spinner=False)
def load_player_match_metrics() -> pd.DataFrame | None:
    """Per-match player table with minutes, basic stats and all five valuation methods."""
    path = get_settings().processed_dir / PLAYER_MATCH_METRICS_FILE
    return pd.read_parquet(path) if path.exists() else None


@st.cache_resource(show_spinner=False)
def load_xt_grid() -> np.ndarray | None:
    """Pre-fitted 12x16 Singh xT grid (None if not built)."""
    path = get_settings().processed_dir / XT_GRID_FILE
    return np.load(path) if path.exists() else None


@st.cache_data(show_spinner=False)
def load_matches() -> pd.DataFrame:
    """All matches with team names, ordered by date."""
    return query(get_settings().processed_dir, MATCHES_SQL)


@st.cache_data(show_spinner="Loading match events...")
def load_match_events(match_id: int) -> pd.DataFrame:
    """Events of one match with player and team names."""
    return query(get_settings().processed_dir, MATCH_EVENTS_SQL, [int(match_id)])


@st.cache_data(show_spinner="Loading shots...")
def load_shots() -> pd.DataFrame:
    """All shots of the season (reads `shots.parquet` when prebuilt, else queries `events`)."""
    processed_dir = get_settings().processed_dir
    shots_path = processed_dir / SHOTS_FILE
    if shots_path.exists():
        return pd.read_parquet(shots_path)
    return query(processed_dir, SHOTS_SQL)


def match_label(row: pd.Series) -> str:
    """`2015-08-08 · Home 2-1 Away` label for match pickers."""
    date = str(row["match_date"])[:10]
    return f"{date} · {row['home_team']} {row['home_score']}-{row['away_score']} {row['away_team']}"
