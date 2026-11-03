"""Pass network: average positions and pass links of one team in one match."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from common import methodology, require, setup_page
from data_service import load_match_events, load_matches, match_label
from pitch import pitch_figure
from statsbomb_analytics.metrics.pass_network import pass_network

setup_page("Pass Network", "🕸️", "Average pass positions and links until the first substitution")

matches = require(load_matches(), "No matches found. Run the ETL (scripts/phase1_etl.py).")
col1, col2 = st.columns(2)
league = col1.selectbox("Competition", sorted(matches["competition_name"].unique()))
pool = matches[matches["competition_name"] == league].reset_index(drop=True)
labels = pool.apply(match_label, axis=1)
choice = col2.selectbox("Match", range(len(pool)), format_func=lambda i: labels.iloc[i])
match = pool.iloc[choice]

team_options = {
    int(match["home_team_id"]): match["home_team"],
    int(match["away_team_id"]): match["away_team"],
}
team_id = st.radio("Team", list(team_options), format_func=team_options.get, horizontal=True)
min_passes = st.slider("Minimum passes per link", 1, 10, 3)

events = load_match_events(int(match["match_id"]))
team_events = events[events["team_id"] == team_id]
nodes, edges = pass_network(team_events, min_passes=min_passes)
require(nodes, "No completed passes with a recipient for this team in this match.")

names = (
    events.dropna(subset=["player_id"])
    .drop_duplicates("player_id")
    .set_index("player_id")["player_name"]
)
short = names.map(lambda n: str(n).split()[-1])
nodes = nodes.assign(label=nodes["player_id"].map(short))
position = nodes.set_index("player_id")[["x", "y"]]

fig = pitch_figure()
max_links = max(int(edges["passes"].max()), 1) if len(edges) else 1
for edge in edges.itertuples():
    a, b = position.loc[edge.player_a], position.loc[edge.player_b]
    fig.add_trace(
        go.Scatter(
            x=[a["x"], b["x"]],
            y=[a["y"], b["y"]],
            mode="lines",
            line={"width": 1 + 7 * edge.passes / max_links, "color": "rgba(76, 201, 240, 0.55)"},
            hovertext=f"{short[edge.player_a]} - {short[edge.player_b]}: {edge.passes} passes",
            hoverinfo="text",
            showlegend=False,
        )
    )
fig.add_trace(
    go.Scatter(
        x=nodes["x"],
        y=nodes["y"],
        mode="markers+text",
        text=nodes["label"],
        textposition="top center",
        marker={
            "size": 12 + 20 * nodes["passes"] / nodes["passes"].max(),
            "color": "#f72585",
            "line": {"width": 1, "color": "white"},
        },
        customdata=nodes["passes"],
        hovertemplate="%{text}: %{customdata} completed passes<extra></extra>",
        name="Players",
    )
)
st.plotly_chart(fig, width="stretch")
st.caption(f"{team_options[team_id]} · {len(nodes)} players · {len(edges)} links shown")

methodology(
    """
    - Only completed passes with a known recipient are used, until the team's first
      substitution (later passes mix line-ups).
    - A node is the average location of the passes made by the player (a rough proxy of his
      role); node size is the number of completed passes. Links are undirected (A->B plus B->A).
    - Network structure depends on the opponent and the game state; one match is a small sample.
    """
)
