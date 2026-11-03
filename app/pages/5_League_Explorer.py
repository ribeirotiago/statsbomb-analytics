"""League explorer: full player table and an interactive scatter of any two metrics."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from common import methodology, require, setup_page
from data_service import MISSING_METRICS, load_player_metrics

setup_page("League Explorer", "📊", "All players normalised per 90 minutes")

METRICS: dict[str, str] = {
    "goals_p90": "Goals / 90",
    "npxg_p90": "npxG / 90",
    "shots_p90": "Shots / 90",
    "key_passes_p90": "Key passes / 90",
    "assists_p90": "Assists / 90",
    "passes_p90": "Passes / 90",
    "pass_pct": "Pass %",
    "dribbles_completed_p90": "Dribbles completed / 90",
    "tackles_p90": "Tackles / 90",
    "interceptions_p90": "Interceptions / 90",
    "clearances_p90": "Clearances / 90",
    "pressures_p90": "Pressures / 90",
    "ball_recoveries_p90": "Ball recoveries / 90",
    "xt_p90": "Singh xT / 90",
    "xgchain_p90": "xGChain / 90",
    "xgbuildup_p90": "xGBuildup / 90",
    "axt_p90": "Action xT / 90",
    "vaep_p90": "VAEP / 90",
    "minutes": "Minutes",
}

table = require(load_player_metrics(), MISSING_METRICS)

col1, col2, col3 = st.columns([2, 2, 3])
league = col1.selectbox(
    "Competition", ["All", *sorted(table["competition_name"].dropna().unique())]
)
groups = col2.multiselect("Position group", ["GK", "DEF", "MID", "FWD"], placeholder="All")
min_minutes = col3.slider("Minimum minutes", 0, 1500, 300, step=50)

df = table[table["minutes"] >= min_minutes]
if league != "All":
    df = df[df["competition_name"] == league]
if groups:
    df = df[df["position_group"].isin(groups)]
teams = col1.multiselect("Teams", sorted(df["team_name"].dropna().unique()), placeholder="All")
if teams:
    df = df[df["team_name"].isin(teams)]
df = require(df, "No players for the selected filters.")
st.caption(f"{len(df)} players")

col_x, col_y, col_c = st.columns(3)
x_col = col_x.selectbox(
    "X axis", list(METRICS), index=list(METRICS).index("xt_p90"), format_func=METRICS.get
)
y_col = col_y.selectbox(
    "Y axis", list(METRICS), index=list(METRICS).index("vaep_p90"), format_func=METRICS.get
)
color_by = col_c.selectbox("Color by", ["position_group", "competition_name", "team_name"])

fig = px.scatter(
    df,
    x=x_col,
    y=y_col,
    color=color_by,
    hover_name="player_name",
    hover_data={"team_name": True, "minutes": ":.0f", x_col: ":.2f", y_col: ":.2f"},
    labels=METRICS | {"position_group": "Group", "competition_name": "League", "team_name": "Team"},
    template="plotly_dark",
    height=560,
)
fig.update_traces(marker={"size": 7, "opacity": 0.7})
st.plotly_chart(fig, width="stretch")

st.subheader("Player table")
shown = ["player_name", "team_name", "competition_name", "primary_position", "matches", "minutes"]
shown += [c for c in METRICS if c != "minutes"]
column_config = {
    c: st.column_config.NumberColumn(label, format="%.2f") for c, label in METRICS.items()
}
column_config["minutes"] = st.column_config.NumberColumn("Minutes", format="%.0f")
column_config["player_name"] = st.column_config.TextColumn("Player")
column_config["team_name"] = st.column_config.TextColumn("Team")
column_config["competition_name"] = st.column_config.TextColumn("League")
column_config["primary_position"] = st.column_config.TextColumn("Position")
st.dataframe(
    df[shown].sort_values("vaep_p90", ascending=False),
    hide_index=True,
    height=600,
    width="stretch",
    column_config=column_config,
)

methodology(
    """
    - Per-90 rates are inflated by small samples: raise the minutes slider before ranking.
    - Singh xT is a **gain** (successful pass/carry moves the ball to a more valuable zone);
      action xT is a **state value** (threat of the possession after the action); VAEP is the
      change in P(score) minus the change in P(concede) over the next 10 actions (out-of-fold).
      Their scales are not comparable: compare ranks (see `docs/phase2_metrics.md`, D16).
    - Volume methods favour possession teams; VAEP favours finishers and rewards goalkeeper
      saves; defensive prevention leaves little trace in event data.
    - One season (15/16); cross-league comparisons are not calibrated.
    """
)
