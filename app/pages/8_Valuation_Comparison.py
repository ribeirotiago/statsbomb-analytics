"""Action valuation comparison: taxonomy, Spearman correlation, positional bias & rank shifts."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from common import methodology, require, setup_page
from data_service import MISSING_METRICS, load_player_metrics
from statsbomb_analytics.metrics.comparison import (
    METHODS,
    mean_percentile_by_group,
    outfield_players,
    rank_shifts,
    spearman_by_group,
    spearman_matrix,
    top_players,
)

setup_page(
    "Action Valuation Comparison",
    "⚖️",
    "Academic comparison of Singh xT, xGChain, xGBuildup, Action xT and Out-of-Fold VAEP (per 90)",
)

raw_table = require(load_player_metrics(), MISSING_METRICS)

col1, col2, col3 = st.columns([1.8, 1.8, 2.2])
leagues = ["All leagues", *sorted(raw_table["competition_name"].dropna().unique())]
scope = col1.selectbox("Competition", leagues)
min_minutes = col2.slider("Minimum minutes", 100, 1500, 300, step=50)
exclude_gk = col3.toggle(
    "Outfield players only (exclude GK)",
    value=True,
    help=(
        "Decision D16: VAEP rewards goalkeeper saves (preventing goals), while possession/zone "
        "models (xT, xGChain) give goalkeepers near-zero values. Toggle this off to observe how "
        "including GKs lowers the Spearman correlation between VAEP and the other four methods."
    ),
)

pool = raw_table[raw_table["minutes"] >= min_minutes].copy()
if scope != "All leagues":
    pool = pool[pool["competition_name"] == scope].copy()

full_pool_with_gk = pool[pool["position_group"].isin(["GK", "DEF", "MID", "FWD"])].reset_index(
    drop=True
)
if exclude_gk:
    pool = outfield_players(pool)
pool = require(pool, "No players match the selected filters.")

st.caption(
    f"Comparison pool: **{len(pool):,} players** "
    f"({scope}, ≥ {min_minutes} min, {'outfield DEF/MID/FWD' if exclude_gk else 'including GK'})."
)

tab_tax, tab_corr, tab_shift, tab_top = st.tabs(
    [
        "📐 Taxonomy & Positional Bias",
        "🔗 Rank Correlation (Spearman)",
        "🔀 Rank Shifts & Disagreements",
        "🏆 Leaderboards by Method",
    ]
)

with tab_tax:
    st.markdown("#### Conceptual Taxonomy of the Five Valuation Models")
    taxonomy = pd.DataFrame(
        [
            {
                "Model": "Singh xT",
                "Paradigm": "Markov chain over 16×12 pitch grid",
                "Action Value": "xT[end] − xT[start] (completed moves)",
                "Actions Scored": "Passes & Carries",
                "Turnover Penalty": "Implicit in grid fit; not subtracted per failed pass",
                "Defensive Actions": "No",
                "Primary Role Bias": "Progressive passers & ball-carriers (MID / full-backs)",
            },
            {
                "Model": "xGChain",
                "Paradigm": "Possession-chain credit assignment",
                "Action Value": "Full non-penalty xG of shot-ending possessions",
                "Actions Scored": "Passes, Carries, Dribbles, Shots in chain",
                "Turnover Penalty": "No",
                "Defensive Actions": "No",
                "Primary Role Bias": "Forwards & attacking creators in high-xG teams",
            },
            {
                "Model": "xGBuildup",
                "Paradigm": "Possession-chain excluding last 2 actions",
                "Action Value": "Possession xG if player neither shot nor assisted",
                "Actions Scored": "Build-up Passes, Carries, Dribbles",
                "Turnover Penalty": "No",
                "Defensive Actions": "No",
                "Primary Role Bias": "Deep-lying playmakers, central midfielders & defenders",
            },
            {
                "Model": "Action xT",
                "Paradigm": "Polynomial logistic P(shot) × linear E[xG]",
                "Action Value": "Predicted threat state after completed move",
                "Actions Scored": "Passes & Carries",
                "Turnover Penalty": "No (sums completed actions)",
                "Defensive Actions": "No",
                "Primary Role Bias": "High-volume final-third passers (strongest team-style bias)",
            },
            {
                "Model": "VAEP (Out-of-Fold)",
                "Paradigm": "Gradient boosting on 3-action game state (10-action horizon)",
                "Action Value": "ΔP(score) − ΔP(concede), 5-fold match CV",
                "Actions Scored": "13 on-ball action types (attack + defense + GK)",
                "Turnover Penalty": "Yes (losing ball drops P(score) and raises P(concede))",
                "Defensive Actions": "Yes (tackles, clearances, recoveries, GK saves)",
                "Primary Role Bias": "Clinical finishers (goals → P=1) & shot-stopping goalkeepers",
            },
        ]
    )
    st.dataframe(taxonomy, hide_index=True, width="stretch", height="content")

    st.markdown("#### Positional Bias: Mean Empirical Percentile by Position Group")
    st.caption(
        "Computed over all four position groups (`GK`, `DEF`, `MID`, `FWD`) to demonstrate which "
        "positions each method systematically favours (0.50 = neutral average rank)."
    )
    pos_bias = mean_percentile_by_group(full_pool_with_gk) * 100.0
    pos_order = [g for g in ["GK", "DEF", "MID", "FWD"] if g in pos_bias.index]
    pos_bias = pos_bias.loc[pos_order]

    melted_bias = pos_bias.reset_index().melt(
        id_vars="position_group", var_name="Method", value_name="Mean Percentile"
    )
    fig_bias = px.bar(
        melted_bias,
        x="position_group",
        y="Mean Percentile",
        color="Method",
        barmode="group",
        template="plotly_dark",
        height=440,
        labels={"position_group": "Position Group", "Mean Percentile": "Mean Percentile (0–100)"},
    )
    fig_bias.add_hline(y=50.0, line_dash="dash", line_color="gray", opacity=0.7)
    st.plotly_chart(fig_bias, width="stretch")

    st.dataframe(
        pos_bias.round(1).reset_index().rename(columns={"position_group": "Position Group"}),
        hide_index=True,
        width="stretch",
        height="content",
    )

with tab_corr:
    c_left, c_right = st.columns([1.1, 1.3])
    with c_left:
        st.markdown("#### Pairwise Spearman Rank Correlation (ρ)")
        corr_mat = spearman_matrix(pool)
        fig_heat = px.imshow(
            corr_mat.round(2),
            text_auto=True,
            color_continuous_scale="Viridis",
            zmin=0.0,
            zmax=1.0,
            template="plotly_dark",
            height=440,
        )
        st.plotly_chart(fig_heat, width="stretch")

    with c_right:
        st.markdown("#### Rank Correlation Inside Each Position Group")
        by_grp = spearman_by_group(pool)
        if by_grp.empty:
            st.info("Not enough players per position group for subgroup correlations.")
        else:
            fig_grp = px.bar(
                by_grp,
                x="pair",
                y="spearman",
                color="position_group",
                barmode="group",
                template="plotly_dark",
                height=440,
                labels={
                    "pair": "Method Pair",
                    "spearman": "Spearman ρ",
                    "position_group": "Group",
                },
            )
            st.plotly_chart(fig_grp, width="stretch")

with tab_shift:
    st.markdown("#### Method Disagreement Explorer (`Rank Shifts`)")
    sc1, sc2, sc3 = st.columns([1.5, 1.5, 1.2])
    method_names = list(METHODS)
    method_a = sc1.selectbox("Method A (Y axis):", method_names, index=method_names.index("VAEP"))
    method_b = sc2.selectbox(
        "Method B (X axis):", method_names, index=method_names.index("Singh xT")
    )
    top_n = int(sc3.number_input("Top N shifts:", min_value=5, max_value=30, value=10, step=5))

    col_a = METHODS[method_a]
    col_b = METHODS[method_b]
    scatter_df = pool[
        [
            "player_name",
            "team_name",
            "competition_name",
            "primary_position",
            "position_group",
            "minutes",
            col_a,
            col_b,
        ]
    ].copy()
    scatter_df["pct_a"] = scatter_df[col_a].rank(pct=True) * 100.0
    scatter_df["pct_b"] = scatter_df[col_b].rank(pct=True) * 100.0
    scatter_df["gap"] = scatter_df["pct_a"] - scatter_df["pct_b"]

    fig_sc = px.scatter(
        scatter_df,
        x="pct_b",
        y="pct_a",
        color="position_group",
        hover_name="player_name",
        hover_data={
            "team_name": True,
            "primary_position": True,
            "minutes": ":.0f",
            col_a: ":.3f",
            col_b: ":.3f",
            "gap": ":+.1f",
        },
        labels={
            "pct_b": f"{method_b} Percentile (0–100)",
            "pct_a": f"{method_a} Percentile (0–100)",
            "position_group": "Position Group",
        },
        template="plotly_dark",
        height=520,
    )
    fig_sc.add_trace(
        go.Scatter(
            x=[0, 100],
            y=[0, 100],
            mode="lines",
            line={"dash": "dash", "color": "rgba(255,255,255,0.45)"},
            name="Equal Rank (y = x)",
            showlegend=True,
        )
    )
    st.plotly_chart(fig_sc, width="stretch")

    t_a, t_b = st.columns(2)
    with t_a:
        st.markdown(f"##### 🔼 Ranked Higher by **{method_a}** than **{method_b}**")
        up_a = rank_shifts(pool, method_a, method_b, n=top_n).copy()
        for c in (f"pct_{method_a}", f"pct_{method_b}", "gap"):
            up_a[c] = (up_a[c] * 100.0).round(1)
        st.dataframe(
            up_a.rename(
                columns={
                    "player_name": "Player",
                    "primary_position": "Position",
                    "minutes": "Min",
                    f"pct_{method_a}": f"{method_a} Pct",
                    f"pct_{method_b}": f"{method_b} Pct",
                    "gap": "Δ Pct",
                }
            ),
            hide_index=True,
            width="stretch",
            column_config={"Min": st.column_config.NumberColumn(format="%.0f")},
        )

    with t_b:
        st.markdown(f"##### 🔼 Ranked Higher by **{method_b}** than **{method_a}**")
        up_b = rank_shifts(pool, method_b, method_a, n=top_n).copy()
        for c in (f"pct_{method_b}", f"pct_{method_a}", "gap"):
            up_b[c] = (up_b[c] * 100.0).round(1)
        st.dataframe(
            up_b.rename(
                columns={
                    "player_name": "Player",
                    "primary_position": "Position",
                    "minutes": "Min",
                    f"pct_{method_b}": f"{method_b} Pct",
                    f"pct_{method_a}": f"{method_a} Pct",
                    "gap": "Δ Pct",
                }
            ),
            hide_index=True,
            width="stretch",
            column_config={"Min": st.column_config.NumberColumn(format="%.0f")},
        )

with tab_top:
    st.markdown("#### Season Per-90 Leaderboards by Valuation Model")
    n_leaders = st.slider("Number of players per leaderboard:", 5, 25, 10, step=5)
    leaders = top_players(pool, n=n_leaders)
    cols_top = st.columns(2)
    for idx, (m_label, df_top) in enumerate(leaders.items()):
        col_val = METHODS[m_label]
        with cols_top[idx % 2]:
            st.markdown(f"##### {m_label} (`{col_val}`)")
            st.dataframe(
                df_top.rename(
                    columns={
                        "player_name": "Player",
                        "primary_position": "Position",
                        "minutes": "Min",
                        col_val: f"{m_label} / 90",
                    }
                ),
                hide_index=True,
                width="stretch",
                column_config={
                    "Min": st.column_config.NumberColumn(format="%.0f"),
                    f"{m_label} / 90": st.column_config.NumberColumn(format="%.3f"),
                },
            )

methodology(
    """
    - **Why Spearman rank correlation (D9, D16)?** Raw values across models have incompatible
      units (Singh xT is a zone-transition gain, xGChain is a possession xG sum, Action xT is a
      state threat value, and VAEP is a 10-action probability difference). Comparing empirical
      ranks (percentiles) is the only mathematically sound approach.
    - **Why does VAEP diverge from possession models?** Possession/progression models (`Singh xT`,
      `Action xT`, `xGBuildup`) reward high-volume ball progression and never subtract turnover
      penalties per failed action. Out-of-fold **VAEP** penalises turnovers and heavily rewards
      high-leverage goal-scoring actions (`P(score) → 1`) and defensive/goalkeeping stops.
    """
)
