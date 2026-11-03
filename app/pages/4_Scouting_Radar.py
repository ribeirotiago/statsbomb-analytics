"""Scouting radar: percentiles against the position group, with a 1x1 comparison mode."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from common import methodology, require, setup_page
from data_service import MISSING_METRICS, load_player_metrics
from statsbomb_analytics.metrics.scouting import RADAR_PROFILES, radar_percentiles

setup_page(
    "Scouting Radar", "🧭", "Percentile profile (0-100) against players of the same position"
)

table = require(load_player_metrics(), MISSING_METRICS)

col1, col2, col3 = st.columns(3)
scope = col1.selectbox(
    "Comparison pool", ["All leagues", *sorted(table["competition_name"].dropna().unique())]
)
min_minutes = col2.slider("Minimum minutes", 0, 1500, 300, step=50)
group = col3.selectbox("Position group", list(RADAR_PROFILES))

pool = table[(table["position_group"] == group) & (table["minutes"] >= min_minutes)]
if scope != "All leagues":
    pool = pool[pool["competition_name"] == scope]
pool = require(pool.reset_index(drop=True), "No players for the selected filters.")

percentiles = radar_percentiles(pool, group)
labels = pool.apply(
    lambda r: f"{r['player_name']} ({r['team_name']}, {r['minutes']:.0f} min)", axis=1
)
st.caption(f"Pool: {len(pool)} players")

col_a, col_b = st.columns(2)
first = col_a.selectbox("Player", range(len(pool)), format_func=lambda i: labels.iloc[i])
compare = col_b.checkbox("Compare with a second player")
second = None
if compare:
    second = col_b.selectbox(
        "Second player",
        range(len(pool)),
        format_func=lambda i: labels.iloc[i],
        index=min(1, len(pool) - 1),
    )

fig = go.Figure()
axes = list(percentiles.columns)
for idx, color in ((first, "#4cc9f0"), (second, "#f72585")):
    if idx is None:
        continue
    values = percentiles.iloc[idx].fillna(0).tolist()
    fig.add_trace(
        go.Scatterpolar(
            r=[*values, values[0]],
            theta=[*axes, axes[0]],
            fill="toself",
            name=pool.loc[idx, "player_name"],
            line={"color": color},
            opacity=0.65,
        )
    )
fig.update_layout(
    template="plotly_dark",
    height=520,
    polar={"radialaxis": {"range": [0, 100], "tickvals": [25, 50, 75, 100]}},
    margin={"t": 30},
)
st.plotly_chart(fig, width="stretch")

profile = RADAR_PROFILES[group]
shown = [first] if second is None else [first, second]
raw = pool.loc[shown, ["player_name", "team_name", "minutes", *profile]].rename(columns=profile)
st.dataframe(
    raw.set_index("player_name"),
    width="stretch",
    column_config={c: st.column_config.NumberColumn(format="%.2f") for c in profile.values()},
)

methodology(
    """
    - Percentiles are rank-based inside the chosen pool (same position group, minutes above the
      slider, league or all five): they show where a player sits, not by how much.
    - Metrics are per 90 (except Pass %); the profile differs by position group.
      Goalkeepers have no profile: event data barely describes their work.
    - Volume metrics reflect team style (possession teams accumulate more passes and xT).
    - The position group comes from the position with most minutes, which blurs hybrid roles.
    """
)
