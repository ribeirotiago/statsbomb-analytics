"""StatsBomb Analytics: entry page of the Streamlit app."""

from __future__ import annotations

import streamlit as st

from common import CREDIT, methodology, setup_page
from data_service import load_matches, load_player_metrics

setup_page("StatsBomb Analytics", "⚽", "Exploratory football analytics on StatsBomb Open Data")

st.markdown(
    """
    Exploratory analyses of the **2015/16** season of Europe's top 5 leagues. Use the menu on the
    left:

    - **Match Report**: score, xG race, key team numbers and the shots of a match.
    - **Shot Map**: shots on the pitch by league, team, player or match (size = xG).
    - **Pass Network**: average positions and pass links of a team in a match.
    - **Scouting Radar**: percentile radar of a player against his position group, with a 1x1
      comparison mode.
    - **League Explorer**: full player table and an interactive scatter of any two metrics,
      including the five valuation methods (Singh xT, xGChain, xGBuildup, action xT, VAEP).
    """
)

matches = load_matches()
players = load_player_metrics()
col1, col2, col3 = st.columns(3)
col1.metric("Matches", f"{len(matches):,}")
col2.metric("Players with minutes", f"{0 if players is None else len(players):,}")
col3.metric("Leagues", matches["competition_name"].nunique())

methodology(
    """
    - **One season (15/16)**: no temporal evolution; the Champions League has a single match
      (the final) and is excluded from aggregate analyses.
    - **Small samples**: per-90 rates are inflated for players with few minutes. Tables and
      radars default to a 300-minute minimum.
    - **Team style bias**: volume metrics (passes, xT, xGChain) favour possession-heavy teams.
    - **Cross-league comparisons** are not calibrated.
    - Detailed methods: `docs/phase2_metrics.md`.
    """
)
st.caption(CREDIT)
