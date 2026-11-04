"""Match report: score, team numbers (with PPDA), xG race, shot map/table and top contributors."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from common import match_picker, methodology, require, setup_page
from data_service import (
    load_match_events,
    load_matches,
    load_player_match_metrics,
)
from match_plots import plot_match_shots_pitch
from statsbomb_analytics.metrics.player_match import MATCH_METHODS, match_top_contributors
from statsbomb_analytics.metrics.xt import ppda

setup_page(
    "Match Report",
    "📋",
    "Score, team numbers, xG race, shot map and top player contributors by valuation model",
)

matches = require(load_matches(), "No matches found. Run the ETL (scripts/phase1_etl.py).")
match, _ = match_picker(matches)
match_id = int(match["match_id"])

events = load_match_events(match_id)
shots = events[(events["type"] == "Shot") & (events["period"] < 5)].copy()
shots["xg"] = shots["shot_xg"].fillna(0.0)

st.subheader(
    f"{match['home_team']} {match['home_score']} - {match['away_score']} {match['away_team']}"
)
st.caption(f"{str(match['match_date'])[:10]} · {match['stadium']} · Referee: {match['referee']}")

teams = [
    (int(match["home_team_id"]), str(match["home_team"])),
    (int(match["away_team_id"]), str(match["away_team"])),
]

ppda_events = events[
    events["x"].notna()
    & (
        events["type"].isin(["Pass", "Interception", "Foul Committed", "Dribbled Past"])
        | ((events["type"] == "Duel") & (events["duel_type"] == "Tackle"))
    )
]
ppda_by_team = ppda(ppda_events).set_index("team_id")["ppda"].to_dict() if len(ppda_events) else {}


def team_numbers(team_id: int) -> dict[str, str | int | float]:
    mine = events[events["team_id"] == team_id]
    passes = mine[mine["type"] == "Pass"]
    team_shots = shots[shots["team_id"] == team_id]
    completed = passes["pass_outcome"].isna().mean() * 100 if len(passes) else 0.0
    pass_pct = round(float(completed), 1)
    team_ppda = ppda_by_team.get(team_id)
    ppda_str = f"{team_ppda:.1f}" if team_ppda is not None and pd.notna(team_ppda) else "—"
    return {
        "xG": f"{float(team_shots['xg'].sum()):.2f}",
        "Shots": len(team_shots),
        "Passes": len(passes),
        "Pass %": f"{pass_pct:.1f}%",
        "Pressures": int((mine["type"] == "Pressure").sum()),
        "PPDA": ppda_str,
    }


cols = st.columns(2)
for col, (team_id, name) in zip(cols, teams, strict=True):
    numbers = team_numbers(team_id)
    with col:
        st.markdown(f"**{name}**")
        metrics = st.columns(len(numbers))
        for metric_col, (label, value) in zip(metrics, numbers.items(), strict=True):
            metric_col.metric(label, value)

st.subheader("xG Race")
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
tab_map, tab_table = st.tabs(["🗺️ Pitch Map", "📋 Shot Table"])
with tab_map:
    fig_shots = plot_match_shots_pitch(
        shots,
        home_team_id=teams[0][0],
        home_name=teams[0][1],
        away_team_id=teams[1][0],
        away_name=teams[1][1],
    )
    st.plotly_chart(fig_shots, width="stretch")

with tab_table:
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

st.divider()
st.subheader("⭐ Top Contributors by Valuation Method")
st.caption(
    "Compare which players stood out in this match according to each action-valuation paradigm: "
    "**Singh xT** (zone progression), **xGChain** (shot-ending possessions), "
    "**xGBuildup** (build-up excluding shooter/assister), **Action xT** (polynomial threat) "
    "and **VAEP** (out-of-fold ΔP(score) − ΔP(concede))."
)

all_match_metrics = load_player_match_metrics()
if all_match_metrics is None:
    st.info(
        "Per-match valuation table not built yet. Run "
        "`uv run python scripts/phase3_build_player_metrics.py` to enable match contributor tables."
    )
else:
    m_df = all_match_metrics[all_match_metrics["match_id"] == match_id].copy()
    if m_df.empty:
        st.info("No player-match valuation rows found for this match.")
    else:
        st.markdown("#### Top 3 Players per Team by Method")
        top_rows: list[dict[str, str]] = []
        for method_label, col_name in MATCH_METHODS.items():
            top_by_team = match_top_contributors(m_df, col_name, n=3)
            for rank_idx in range(3):
                row_entry: dict[str, str] = {"Method": method_label, "Rank": f"#{rank_idx + 1}"}
                for team_id, team_name in teams:
                    t_top = top_by_team.get(team_id, pd.DataFrame())
                    if len(t_top) > rank_idx:
                        r = t_top.iloc[rank_idx]
                        row_entry[team_name] = (
                            f"{r['player_name']} ({r[col_name]:+.3f} · {r['minutes']:.0f}m)"
                        )
                    else:
                        row_entry[team_name] = "—"
                top_rows.append(row_entry)

        st.dataframe(
            pd.DataFrame(top_rows),
            hide_index=True,
            width="stretch",
            height="content",
        )

        st.markdown("#### All Match Players — Method Explorer")
        metric_options = {
            "vaep": "VAEP (Total)",
            "vaep_offensive": "VAEP (Offensive)",
            "vaep_defensive": "VAEP (Defensive)",
            "xt": "Singh xT (Total)",
            "xt_pass": "Singh xT (Passes)",
            "xt_carry": "Singh xT (Carries)",
            "xgchain": "xGChain",
            "xgbuildup": "xGBuildup",
            "axt": "Action xT",
            "xg": "Shot xG",
        }
        chosen_col = st.selectbox(
            "Select metric to rank players in this match:",
            options=list(metric_options),
            format_func=metric_options.get,
        )
        tab_chart, tab_full = st.tabs(["📊 Bar Chart", "📋 Full Match Table"])
        with tab_chart:
            chart_df = m_df.sort_values(chosen_col, ascending=True)
            fig_bar = px.bar(
                chart_df,
                x=chosen_col,
                y="player_name",
                color="team_name",
                orientation="h",
                color_discrete_map={teams[0][1]: "#4cc9f0", teams[1][1]: "#f72585"},
                hover_data=["position", "minutes", "goals", "assists", "xt", "xgchain", "vaep"],
                labels={
                    chosen_col: metric_options[chosen_col],
                    "player_name": "Player",
                    "team_name": "Team",
                },
                template="plotly_dark",
                height=max(420, 22 * len(chart_df)),
            )
            st.plotly_chart(fig_bar, width="stretch")

        with tab_full:
            display_cols = [
                "player_name",
                "team_name",
                "position",
                "minutes",
                "goals",
                "assists",
                "xg",
                "xt",
                "xgchain",
                "xgbuildup",
                "axt",
                "vaep",
                "vaep_offensive",
                "vaep_defensive",
            ]
            st.dataframe(
                m_df[display_cols].sort_values(chosen_col, ascending=False),
                hide_index=True,
                width="stretch",
                height=500,
                column_config={
                    "player_name": "Player",
                    "team_name": "Team",
                    "position": "Position",
                    "minutes": st.column_config.NumberColumn("Min", format="%.0f"),
                    "goals": "G",
                    "assists": "A",
                    "xg": st.column_config.NumberColumn("xG", format="%.2f"),
                    "xt": st.column_config.NumberColumn("Singh xT", format="%.3f"),
                    "xgchain": st.column_config.NumberColumn("xGChain", format="%.3f"),
                    "xgbuildup": st.column_config.NumberColumn("xGBuildup", format="%.3f"),
                    "axt": st.column_config.NumberColumn("Action xT", format="%.3f"),
                    "vaep": st.column_config.NumberColumn("VAEP", format="%.3f"),
                    "vaep_offensive": st.column_config.NumberColumn("VAEP Off", format="%.3f"),
                    "vaep_defensive": st.column_config.NumberColumn("VAEP Def", format="%.3f"),
                },
            )

methodology(
    """
    - **xG** is the StatsBomb model value; the penalty shootout (period 5) is excluded.
    - **PPDA** = opponent passes in their build-up 60% of the pitch divided by defensive actions
      (tackles, interceptions, fouls committed, dribbled past) in the pressing zone. Lower PPDA
      means a higher-intensity press.
    - **Single-match contributor comparison**: notice how **VAEP** highlights decisive actions
      (goals, shot assists, crucial defensive stops/saves), **xGChain** credits everyone involved
      in shot-ending possessions, **xGBuildup** isolates deeper progression in those possessions,
      and **Singh xT / Action xT** reward ball progression into dangerous zones regardless of
      whether a shot followed.
    """
)
