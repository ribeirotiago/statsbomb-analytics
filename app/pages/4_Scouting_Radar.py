"""Scouting radar: pizza chart of percentiles, KPI cards, breakdown table and 1x1 comparison."""

from __future__ import annotations

import streamlit as st

from common import ALL_TEAMS, methodology, require, setup_page
from data_service import MISSING_METRICS, load_player_metrics
from radar_plots import plot_comparison_pizza, plot_scouting_pizza
from statsbomb_analytics.metrics.scouting import add_percentiles, breakdown_table, pizza_data

ALL_POSITIONS = "All positions"
POSITION_GROUPS = ["FWD", "MID", "DEF", "GK"]

setup_page(
    "Scouting Radar",
    "🧭",
    "Multi-dimensional percentile profile compared against the competition distribution",
)

table = require(load_player_metrics(), MISSING_METRICS)

col1, col2, col3, col4 = st.columns([1.8, 1.8, 1.5, 1.5])
leagues = ["All leagues", *sorted(table["competition_name"].dropna().unique())]
scope = col1.selectbox("Competition", leagues)

in_league = table if scope == "All leagues" else table[table["competition_name"] == scope]
teams = [ALL_TEAMS, *sorted(in_league["team_name"].dropna().unique())]
selected_team = col2.selectbox(
    "Team (player filter)",
    teams,
    help="Filters the player selector below; percentiles are still computed over the full pool.",
)
min_minutes = col3.slider("Minimum minutes", 0, 1500, 300, step=50)
selected_group = col4.selectbox(
    "Position group",
    [ALL_POSITIONS, *POSITION_GROUPS],
    help="Restricts the comparison pool to the chosen position group before computing percentiles.",
)

position_scoped = selected_group != ALL_POSITIONS

tog1, tog2 = st.columns(2)
compare_mode = tog1.toggle("Head-to-Head Comparison (1x1)", value=False)
benchmark_by_group = tog2.toggle(
    "Percentiles within Position Group",
    value=True,
    disabled=position_scoped,
    help=(
        "Ranks each player only against peers of his own position group (FWD vs FWD, MID vs MID, "
        "DEF vs DEF, GK vs GK). Always active when a specific position group is selected."
    ),
)

pool = in_league[in_league["minutes"] >= min_minutes].copy()
if position_scoped:
    pool = pool[pool["position_group"] == selected_group].copy()
pool = require(pool.reset_index(drop=True), "No players in the comparison pool for these filters.")

effective_by_group = position_scoped or benchmark_by_group
ranked_pool = add_percentiles(pool, by_group=effective_by_group)

candidates = (
    ranked_pool
    if selected_team == ALL_TEAMS
    else ranked_pool[ranked_pool["team_name"] == selected_team]
)
candidates = require(
    candidates.sort_values("player_name").reset_index(drop=True),
    "No players for the selected team with enough minutes.",
)

st.caption(
    f"Percentile pool: **{len(ranked_pool)} players** "
    f"({scope}, >= {min_minutes} min"
    f"{f', {selected_group} only' if position_scoped else ''}) · "
    f"Showing **{len(candidates)}** in the selector."
)


def _label(row) -> str:
    return (
        f"[{row['team_name']}] {row['player_name']} — "
        f"{row['primary_position']} ({row['minutes']:.0f} min)"
    )


cand_labels = candidates.apply(_label, axis=1)
all_sorted = ranked_pool.sort_values("player_name").reset_index(drop=True)
all_labels = all_sorted.apply(_label, axis=1)

st.subheader("Player Selection")
if compare_mode:
    col_p1, col_p2 = st.columns(2)
    idx_1 = col_p1.selectbox(
        "Player 1 (Primary)", range(len(candidates)), format_func=lambda i: cand_labels.iloc[i]
    )
    row_1 = candidates.iloc[idx_1]
    other_pool = all_sorted[
        ~(
            (all_sorted["player_id"] == row_1["player_id"])
            & (all_sorted["team_id"] == row_1["team_id"])
        )
    ].reset_index(drop=True)
    require(other_pool, "No second player available in the pool for comparison.")
    other_labels = other_pool.apply(_label, axis=1)
    idx_2 = col_p2.selectbox(
        "Player 2 (Comparison)", range(len(other_pool)), format_func=lambda i: other_labels.iloc[i]
    )
    row_2 = other_pool.iloc[idx_2]
else:
    idx_1 = st.selectbox(
        "Selected Player", range(len(candidates)), format_func=lambda i: cand_labels.iloc[i]
    )
    row_1 = candidates.iloc[idx_1]
    row_2 = None

st.markdown("### Performance Summary")
kpis = st.columns(6)
kpis[0].metric("Position", f"{row_1['primary_position']} ({row_1['position_group']})")
kpis[1].metric("Minutes", f"{row_1['minutes']:.0f}m")
kpis[2].metric("Matches", f"{int(row_1['matches'])}")
kpis[3].metric("Goals", f"{int(row_1['goals'])}")
kpis[4].metric("Assists", f"{int(row_1['assists'])}")
kpis[5].metric("npxG / VAEP", f"{row_1['npxg']:.2f} / {row_1['vaep']:.2f}")

st.divider()

col_viz, col_tab = st.columns([5, 5])
with col_viz:
    data_1 = pizza_data(row_1)
    if compare_mode and row_2 is not None:
        fig = plot_comparison_pizza(data_1, pizza_data(row_2))
    else:
        fig = plot_scouting_pizza(data_1)
    st.plotly_chart(fig, width="stretch")

with col_tab:
    st.markdown("### Metric Breakdown")
    group_1 = row_1["position_group"]
    if effective_by_group:
        n_peers = int((ranked_pool["position_group"] == group_1).sum())
        st.caption(
            f"Percentiles relative to **{group_1}** peers (**{n_peers} players** "
            f"with >= {min_minutes}m)."
        )
        pct_col_label = f"Percentile ({group_1})"
    else:
        st.caption(
            f"Percentiles relative to all **{len(ranked_pool)} players** "
            f"in the pool (>= {min_minutes}m)."
        )
        pct_col_label = "League Percentile"

    df_table = breakdown_table(row_1)
    st.dataframe(
        df_table,
        width="stretch",
        hide_index=True,
        column_config={
            "Percentile": st.column_config.ProgressColumn(
                pct_col_label,
                help="Empirical percentile from 0 to 100",
                format="%d",
                min_value=0,
                max_value=100,
            ),
        },
        height="content",
    )

methodology(
    """
    - Percentiles are rank-based (0-100) inside the comparison pool (competition or all five,
      minutes above the slider). Team selection only filters the player dropdown; it never
      shrinks the percentile pool.
    - Metrics are grouped into **Attack**, **Creation**, **Defense** and **Possession**, and
      include all five action-valuation models (Singh xT, xGChain, xGBuildup, Action xT, VAEP).
    - Volume metrics reflect team style (possession teams accumulate more passes and xT).
    - The position group comes from the position with most minutes, which blurs hybrid roles.
    """
)
