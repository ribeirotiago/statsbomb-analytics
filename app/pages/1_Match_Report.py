"""Match report: score, xG race, team numbers and shots."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from common import match_picker, methodology, require, setup_page
from data_service import load_match_events, load_matches

setup_page("Match Report", "📋", "Score, xG race and team numbers for a single match")

matches = require(load_matches(), "No matches found. Run the ETL (scripts/phase1_etl.py).")
match, _ = match_picker(matches)

events = load_match_events(int(match["match_id"]))
shots = events[(events["type"] == "Shot") & (events["period"] < 5)].copy()
shots["xg"] = shots["shot_xg"].fillna(0.0)

st.subheader(
    f"{match['home_team']} {match['home_score']} - {match['away_score']} {match['away_team']}"
)
st.caption(f"{str(match['match_date'])[:10]} · {match['stadium']} · Referee: {match['referee']}")

teams = [(match["home_team_id"], match["home_team"]), (match["away_team_id"], match["away_team"])]


def team_numbers(team_id: int) -> dict[str, float]:
    mine = events[events["team_id"] == team_id]
    passes = mine[mine["type"] == "Pass"]
    team_shots = shots[shots["team_id"] == team_id]
    completed = passes["pass_outcome"].isna().mean() * 100 if len(passes) else 0.0
    pass_pct = round(float(completed), 1)
    return {
        "xG": round(float(team_shots["xg"].sum()), 2),
        "Shots": len(team_shots),
        "Passes": len(passes),
        "Pass %": pass_pct,
        "Pressures": int((mine["type"] == "Pressure").sum()),
    }


cols = st.columns(2)
for col, (team_id, name) in zip(cols, teams, strict=True):
    numbers = team_numbers(int(team_id))
    with col:
        st.markdown(f"**{name}**")
        metrics = st.columns(len(numbers))
        for metric_col, (label, value) in zip(metrics, numbers.items(), strict=True):
            metric_col.metric(label, value)

st.subheader("xG race")
fig = go.Figure()
colors = ["#4cc9f0", "#f72585"]
for (team_id, name), color in zip(teams, colors, strict=True):
    team_shots = shots[shots["team_id"] == team_id].sort_values(["minute", "index"])
    minutes = [0, *team_shots["minute"].tolist(), max(95, int(events["minute"].max()))]
    cumulative = [0.0, *team_shots["xg"].cumsum().tolist()]
    cumulative.append(cumulative[-1])
    fig.add_trace(
        go.Scatter(
            x=minutes, y=cumulative, mode="lines", line_shape="hv", name=name, line={"color": color}
        )
    )
    goals = team_shots[team_shots["shot_outcome"] == "Goal"]
    if len(goals):
        fig.add_trace(
            go.Scatter(
                x=goals["minute"],
                y=team_shots["xg"].cumsum().loc[goals.index],
                mode="markers",
                marker={"size": 11, "color": color, "symbol": "star"},
                name=f"{name} goals",
                text=goals["player_name"],
                hovertemplate="%{text} (%{x}')<extra></extra>",
            )
        )
fig.update_layout(
    template="plotly_dark",
    height=380,
    xaxis_title="Minute",
    yaxis_title="Cumulative xG",
    margin={"t": 10},
)
st.plotly_chart(fig, width="stretch")

st.subheader("Shots")
table = shots[["minute", "team_name", "player_name", "xg", "shot_outcome", "shot_body_part"]]
st.dataframe(
    table.sort_values("minute").rename(
        columns={
            "minute": "Min",
            "team_name": "Team",
            "player_name": "Player",
            "xg": "xG",
            "shot_outcome": "Outcome",
            "shot_body_part": "Body part",
        }
    ),
    hide_index=True,
    width="stretch",
    column_config={"xG": st.column_config.NumberColumn(format="%.2f")},
)

methodology(
    """
    - xG is the StatsBomb model value, not our own model. The penalty shootout is excluded.
    - Pass % is completed passes over attempted passes by the team.
    - A single match is a very small sample: xG race and team numbers describe the match, not
      the quality of the teams.
    """
)
