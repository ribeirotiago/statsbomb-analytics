"""Shot map: season shots by league, team, player or match."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from common import methodology, require, setup_page
from data_service import load_matches, load_player_metrics, load_shots
from pitch import pitch_figure

setup_page("Shot Map", "🎯", "Shot locations; marker size is proportional to xG")

shots = require(load_shots(), "No shots found. Run the ETL (scripts/phase1_etl.py).")
matches = load_matches()
shots = shots.merge(matches[["match_id", "competition_name"]], on="match_id", how="left").dropna(
    subset=["x", "y"]
)

col1, col2, col3 = st.columns(3)
league = col1.selectbox("Competition", sorted(shots["competition_name"].dropna().unique()))
in_league = shots[shots["competition_name"] == league]
team = col2.selectbox("Team", ["All"] + sorted(in_league["team_name"].dropna().unique()))
in_team = in_league if team == "All" else in_league[in_league["team_name"] == team]
player = col3.selectbox("Player", ["All"] + sorted(in_team["player_name"].dropna().unique()))
selected = in_team if player == "All" else in_team[in_team["player_name"] == player]

col4, col5 = st.columns(2)
hide_pens = col4.checkbox("Exclude penalties", value=True)
only_goals = col5.checkbox("Goals only", value=False)
if hide_pens:
    selected = selected[selected["shot_type"] != "Penalty"]
if only_goals:
    selected = selected[selected["shot_outcome"] == "Goal"]
require(selected, "No shots for the selected filters.")

metrics = st.columns(4)
goals = int((selected["shot_outcome"] == "Goal").sum())
xg = float(selected["shot_xg"].sum())
metrics[0].metric("Shots", len(selected))
metrics[1].metric("Goals", goals)
metrics[2].metric("xG", f"{xg:.1f}")
metrics[3].metric("Goals - xG", f"{goals - xg:+.1f}")

fig = pitch_figure(half=True)
for outcome, color, symbol in (("Goal", "#2ecc71", "star"), ("Other", "#e74c3c", "circle")):
    part = (
        selected[selected["shot_outcome"] == "Goal"]
        if outcome == "Goal"
        else selected[selected["shot_outcome"] != "Goal"]
    )
    fig.add_trace(
        go.Scatter(
            x=part["x"],
            y=part["y"],
            mode="markers",
            name=outcome,
            marker={
                "size": (part["shot_xg"].fillna(0.03) * 60 + 6),
                "color": color,
                "symbol": symbol,
                "opacity": 0.75,
                "line": {"width": 0.5, "color": "white"},
            },
            customdata=part[["player_name", "minute", "shot_xg", "shot_outcome", "team_name"]],
            hovertemplate="%{customdata[0]} (%{customdata[4]})<br>Min %{customdata[1]} · "
            "xG %{customdata[2]:.2f} · %{customdata[3]}<extra></extra>",
        )
    )
st.plotly_chart(fig, width="stretch")

if player != "All":
    players = load_player_metrics()
    if players is not None:
        row = players[players["player_name"] == player]
        if len(row):
            st.caption(f"Minutes in the season: {row['minutes'].sum():.0f}")

methodology(
    """
    - xG is the StatsBomb model value (not our own). Goals - xG over a season is a noisy
      finishing indicator: small samples and shot selection matter.
    - Coordinates are attacking left to right; only the attacking half is drawn.
    - Penalties are excluded by default (they inflate xG per shot); the shootout is never shown.
    """
)
