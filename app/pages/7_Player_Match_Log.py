"""Player match log & valuation comparison: match-by-match trajectory across the 5 models."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from common import ALL_TEAMS, methodology, require, setup_page
from data_service import MISSING_METRICS, load_player_match_metrics
from statsbomb_analytics.metrics.player_match import MATCH_METHODS, normalize_player_match_log

setup_page(
    "Player Match Log & Valuation",
    "📈",
    "Match-by-match trajectory and method comparison (xT, xGChain, xGBuildup, Action xT, VAEP)",
)

match_table = require(load_player_match_metrics(), MISSING_METRICS)

col1, col2, col3 = st.columns([1.4, 1.5, 2.5])
leagues = sorted(match_table["competition_name"].dropna().unique())
league = col1.selectbox("Competition", leagues)
in_league = match_table[match_table["competition_name"] == league]

teams = [ALL_TEAMS, *sorted(in_league["team_name"].dropna().unique())]
team = col2.selectbox("Team", teams)
in_team = in_league if team == ALL_TEAMS else in_league[in_league["team_name"] == team]

players = (
    in_team.groupby(["player_id", "player_name", "team_name", "position"], as_index=False)
    .agg(matches=("match_id", "nunique"), minutes=("minutes", "sum"))
    .sort_values(["player_name", "team_name"])
    .reset_index(drop=True)
)
require(players, "No players found for the selected filters.")


def _player_label(idx: int) -> str:
    r = players.iloc[idx]
    prefix = f"[{r['team_name']}] " if team == ALL_TEAMS else ""
    return (
        f"{prefix}{r['player_name']} — {r['position']} "
        f"({int(r['matches'])} matches · {r['minutes']:.0f}m)"
    )


p_idx = col3.selectbox("Player", range(len(players)), format_func=_player_label)
chosen = players.iloc[p_idx]
player_id = int(chosen["player_id"])
player_name = str(chosen["player_name"])

df_log = (
    in_team[in_team["player_id"] == player_id]
    .sort_values(["match_date", "match_id"])
    .reset_index(drop=True)
)
require(df_log, f"No matches found for {player_name}.")


def _short_match(row: pd.Series) -> str:
    date = str(row["match_date"])[:10]
    opp = row["away_team"] if row["team_name"] == row["home_team"] else row["home_team"]
    venue = "H" if row["team_name"] == row["home_team"] else "A"
    return f"{date} vs {opp} ({venue}, {int(row['home_score'])}-{int(row['away_score'])})"


df_log["match_label"] = df_log.apply(_short_match, axis=1)
df_log = normalize_player_match_log(df_log, MATCH_METHODS)

st.subheader(
    f"{player_name} ({chosen['team_name']}) · {len(df_log)} matches · "
    f"{df_log['minutes'].sum():.0f} min"
)

st.markdown("#### Peak Match & Season Summary by Valuation Method")
summary_rows = []
for label, col in MATCH_METHODS.items():
    best_idx = int(df_log[col].idxmax())
    best_row = df_log.loc[best_idx]
    summary_rows.append(
        {
            "Valuation Method": label,
            "Season Total": f"{df_log[col].sum():+.3f}",
            "Mean / Match": f"{df_log[col].mean():+.3f}",
            "Std Dev / Match": f"{df_log[col].std(ddof=0):.3f}",
            "Peak Match": best_row["match_label"],
            "Peak Value": f"{best_row[col]:+.3f} ({best_row['minutes']:.0f}m)",
        }
    )
st.dataframe(pd.DataFrame(summary_rows), hide_index=True, width="stretch", height="content")

tab_multi, tab_decomp, tab_table = st.tabs(
    [
        "⚖️ Multi-Method Match Trajectory",
        "📊 Component Breakdowns (Pass/Carry & Off/Def)",
        "📋 Match-by-Match Table",
    ]
)

METHOD_COLORS = {
    "Singh xT": "#4cc9f0",
    "xGChain": "#f72585",
    "xGBuildup": "#b5179e",
    "Action xT": "#4361ee",
    "VAEP": "#ffb703",
}

with tab_multi:
    c_scale, c_sel = st.columns([1.4, 2.2])
    scale_mode = c_scale.radio(
        "Comparison scale:",
        options=["Within-Player Percentile (0–100)", "Within-Player Z-Score (σ)", "Raw Value"],
        help=(
            "Raw scales differ across models. Normalising each method across the player's own "
            "matches reveals where the models agree on a standout game and where they diverge."
        ),
    )
    selected_methods = c_sel.multiselect(
        "Valuation models to plot:",
        options=list(MATCH_METHODS),
        default=list(MATCH_METHODS),
    )
    if not selected_methods:
        st.warning("Select at least one valuation method.")
    else:
        suffix = (
            "_pct"
            if scale_mode.startswith("Within-Player Percentile")
            else ("_z" if scale_mode.startswith("Within-Player Z-Score") else "")
        )
        fig_traj = go.Figure()
        for m_name in selected_methods:
            raw_col = MATCH_METHODS[m_name]
            y_col = f"{raw_col}{suffix}"
            fig_traj.add_trace(
                go.Scatter(
                    x=df_log["match_label"],
                    y=df_log[y_col],
                    mode="lines+markers",
                    name=m_name,
                    line={"color": METHOD_COLORS.get(m_name, "#ffffff"), "width": 2.4},
                    marker={"size": 8},
                    customdata=df_log[[raw_col, "minutes", "goals", "assists"]],
                    hovertemplate=(
                        f"<b>{m_name}</b>: %{{y:.2f}} (raw: %{{customdata[0]:+.3f}})<br>"
                        "Min: %{customdata[1]:.0f}' · G: %{customdata[2]} · A: %{customdata[3]}"
                        "<extra></extra>"
                    ),
                )
            )
        fig_traj.update_layout(
            template="plotly_dark",
            height=460,
            xaxis_title="Match (chronological)",
            yaxis_title=scale_mode,
            hovermode="x unified",
            legend={"orientation": "h", "y": 1.1},
        )
        st.plotly_chart(fig_traj, width="stretch")

    if len(df_log) >= 3:
        st.markdown("##### Intra-Player Method Agreement Across Matches (Spearman ρ)")
        st.caption(
            "Measures whether this player's best matches according to one method are also his "
            "best matches according to the others."
        )
        corr = (
            df_log[list(MATCH_METHODS.values())]
            .rename(columns={v: k for k, v in MATCH_METHODS.items()})
            .corr(method="spearman")
        )
        fig_corr = px.imshow(
            corr.round(2),
            text_auto=True,
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            template="plotly_dark",
            height=380,
        )
        st.plotly_chart(fig_corr, width="stretch")

with tab_decomp:
    decomp_choice = st.selectbox(
        "Select decomposition view:",
        options=[
            "VAEP: Offensive vs Defensive per Match",
            "Singh xT: Passes vs Carries per Match",
            "Possession Chains: Build-up (xGBuildup) vs Direct Shot/Assist Involvement",
            "Action xT: Passes vs Carries per Match",
        ],
    )
    plot_df = df_log.copy()
    if decomp_choice.startswith("VAEP"):
        cols_bar = ["vaep_offensive", "vaep_defensive"]
        labels_bar = {"vaep_offensive": "VAEP Offensive", "vaep_defensive": "VAEP Defensive"}
        barmode = "relative"
    elif decomp_choice.startswith("Singh xT"):
        cols_bar = ["xt_pass", "xt_carry"]
        labels_bar = {"xt_pass": "xT Passes", "xt_carry": "xT Carries"}
        barmode = "relative"
    elif decomp_choice.startswith("Possession"):
        plot_df["xg_direct_finish_assist"] = (plot_df["xgchain"] - plot_df["xgbuildup"]).clip(
            lower=0.0
        )
        cols_bar = ["xgbuildup", "xg_direct_finish_assist"]
        labels_bar = {
            "xgbuildup": "xGBuildup (Build-up only)",
            "xg_direct_finish_assist": "Shot / Key Pass Chain xG",
        }
        barmode = "stack"
    else:
        cols_bar = ["axt_pass", "axt_carry"]
        labels_bar = {"axt_pass": "Action xT Passes", "axt_carry": "Action xT Carries"}
        barmode = "stack"

    melted = plot_df.melt(
        id_vars=["match_label", "minutes", "goals", "assists"],
        value_vars=cols_bar,
        var_name="Component",
        value_name="Value",
    )
    melted["Component"] = melted["Component"].map(labels_bar)
    fig_decomp = px.bar(
        melted,
        x="match_label",
        y="Value",
        color="Component",
        barmode=barmode,
        hover_data=["minutes", "goals", "assists"],
        template="plotly_dark",
        height=460,
        labels={"match_label": "Match"},
    )
    st.plotly_chart(fig_decomp, width="stretch")

with tab_table:
    table_cols = [
        "match_date",
        "match_label",
        "position",
        "minutes",
        "goals",
        "assists",
        "xg",
        "xt",
        "xt_pass",
        "xt_carry",
        "xgchain",
        "xgbuildup",
        "axt",
        "vaep",
        "vaep_offensive",
        "vaep_defensive",
    ]
    st.dataframe(
        df_log[table_cols].sort_values("match_date", ascending=False),
        hide_index=True,
        width="stretch",
        height=600,
        column_config={
            "match_date": "Date",
            "match_label": "Match",
            "position": "Pos",
            "minutes": st.column_config.NumberColumn("Min", format="%.0f"),
            "goals": "G",
            "assists": "A",
            "xg": st.column_config.NumberColumn("xG", format="%.2f"),
            "xt": st.column_config.NumberColumn("Singh xT", format="%.3f"),
            "xt_pass": st.column_config.NumberColumn("xT Pass", format="%.3f"),
            "xt_carry": st.column_config.NumberColumn("xT Carry", format="%.3f"),
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
    - **Why compare match trajectories across methods?** In a single match, a midfielder can
      record high **Singh xT** and **Action xT** by progressing the ball into the final third,
      yet score near zero in **xGChain / xGBuildup** if teammates fail to shoot at the end of
      those possessions. Conversely, a striker who converts one chance with few touches spikes
      in **VAEP** (`ΔP(score)`) while accumulating minimal **Singh xT**.
    - **Within-Player Percentile / Z-Score**: normalises each metric over the selected player's
      matches in the season so curves with different units can be overlaid without distortion.
    """
)
