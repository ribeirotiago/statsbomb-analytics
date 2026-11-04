"""Similar Players & Profiles: k-NN baseline, domain tactical archetypes, PCA & K-Means ML."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from common import ALL_TEAMS, methodology, require, setup_page
from data_service import MISSING_METRICS, load_player_metrics
from radar_plots import plot_comparison_pizza
from statsbomb_analytics.metrics.scouting import pizza_data
from statsbomb_analytics.metrics.similarity import (
    ARCHETYPES,
    evaluate_kmeans_range,
    find_similar_players,
    fit_kmeans_clusters,
    fit_pca_profile,
    metric_deltas,
    player_archetype_profile,
    score_archetypes,
)

ALL_POSITIONS = "All positions"

setup_page(
    "Similar Players & Profiles",
    "🧬",
    "Baseline-first player similarity (Euclidean/Cosine k-NN), tactical archetypes, PCA & K-Means",
)

raw_table = require(load_player_metrics(), MISSING_METRICS)

f1, f2, f3, f4 = st.columns([1.6, 1.6, 1.4, 1.4])
leagues = ["All leagues", *sorted(raw_table["competition_name"].dropna().unique())]
scope = f1.selectbox("Competition Pool", leagues)

in_league = (
    raw_table if scope == "All leagues" else raw_table[raw_table["competition_name"] == scope]
)
min_minutes = f2.slider("Minimum minutes", 100, 1500, 300, step=50)
exclude_gk = f3.toggle(
    "Outfield players only (exclude GK)",
    value=True,
    help=(
        "Goalkeepers have near-zero volume on outfield progression and attacking metrics, "
        "which creates a trivial GK vs outfield split in PCA/K-Means. Excluding them keeps "
        "the feature space focused on outfield tactical profiles."
    ),
)
by_group_pct = f4.toggle(
    "Percentiles within Position Group",
    value=True,
    help=(
        "When enabled, each player's 0–100 percentiles are ranked against peers of his own "
        "position group (FWD, MID, DEF) rather than all positions combined."
    ),
)

pool = in_league[in_league["minutes"] >= min_minutes].copy()
if exclude_gk:
    pool = pool[pool["position_group"].isin(["DEF", "MID", "FWD"])].copy()
else:
    pool = pool[pool["position_group"].isin(["GK", "DEF", "MID", "FWD"])].copy()
pool = require(pool.reset_index(drop=True), "No players in the comparison pool for these filters.")

scored_pool = score_archetypes(pool, by_group=by_group_pct)

st.markdown("#### Target Player Selection")
s1, s2, s3 = st.columns([1.4, 1.4, 2.8])
avail_groups = [
    g
    for g in (["FWD", "MID", "DEF"] if exclude_gk else ["FWD", "MID", "DEF", "GK"])
    if g in scored_pool["position_group"].unique()
]
sel_group = s1.selectbox("Filter Selector by Position Group", [ALL_POSITIONS, *avail_groups])

selector_df = (
    scored_pool
    if sel_group == ALL_POSITIONS
    else scored_pool[scored_pool["position_group"] == sel_group]
)
teams = [ALL_TEAMS, *sorted(selector_df["team_name"].dropna().unique())]
sel_team = s2.selectbox(
    "Filter Selector by Team",
    teams,
    help="Narrows the target player dropdown below; the comparison pool remains unchanged.",
)
if sel_team != ALL_TEAMS:
    selector_df = selector_df[selector_df["team_name"] == sel_team]
selector_df = require(
    selector_df.sort_values("player_name").reset_index(drop=True),
    "No players match the selector filters.",
)


def _player_label(row: pd.Series) -> str:
    return (
        f"[{row['team_name']}] {row['player_name']} — "
        f"{row['primary_position']} ({row['minutes']:.0f} min)"
    )


labels = selector_df.apply(_player_label, axis=1)
target_idx = s3.selectbox(
    "Target Player", range(len(selector_df)), format_func=lambda i: labels.iloc[i]
)
target_row = selector_df.iloc[target_idx]

st.caption(
    f"Active comparison pool: **{len(scored_pool):,} players** "
    f"({scope}, ≥ {min_minutes} min, "
    f"{'within-group' if by_group_pct else 'pool-wide'} percentiles)."
)

kpis = st.columns(6)
kpis[0].metric("Position", f"{target_row['primary_position']} ({target_row['position_group']})")
kpis[1].metric("Minutes (Matches)", f"{target_row['minutes']:.0f}m ({int(target_row['matches'])})")
kpis[2].metric(
    "Primary Archetype",
    str(target_row["archetype"]),
    f"{target_row['archetype_score']:.1f} / 100",
    delta_color="off",
)
kpis[3].metric(
    "Secondary Archetype",
    str(target_row["secondary_archetype"]),
    f"{target_row['secondary_archetype_score']:.1f} / 100",
    delta_color="off",
)
kpis[4].metric(
    "npxG / Goals (p90)", f"{target_row['npxg_p90']:.2f} / {target_row['goals_p90']:.2f}"
)
kpis[5].metric(
    "Singh xT / VAEP (p90)", f"{target_row['xt_p90']:.2f} / {target_row['vaep_p90']:.2f}"
)

tab_knn, tab_arch, tab_ml = st.tabs(
    [
        "🔎 1. k-NN Player Similarity (Baseline)",
        "🎭 2. Tactical Archetypes (Domain Baseline)",
        "🧭 3. PCA & K-Means Clustering (Exploratory ML)",
    ]
)

with tab_knn:
    c1, c2, c3, c4, c5 = st.columns([1.6, 1.6, 1.2, 1.2, 1.2])
    metric_choice = c1.selectbox(
        "Distance Metric",
        options=["euclidean", "cosine"],
        format_func=lambda m: (
            "Euclidean (Magnitude & Shape)"
            if m == "euclidean"
            else "Cosine (Profile Shape / Style)"
        ),
    )
    rep_choice = c2.selectbox(
        "Feature Space",
        options=["percentiles", "zscore"],
        format_func=lambda r: (
            "Percentiles (0–100, Robust)" if r == "percentiles" else "Standardised Per-90 (Z-Score)"
        ),
    )
    same_group = c3.toggle("Same Position Group", value=True)
    inc_teammates = c4.toggle("Include Teammates", value=True)
    top_k = int(c5.slider("Top K Neighbours", 5, 25, 10, step=5))

    neighbours = find_similar_players(
        scored_pool,
        target_player_id=int(target_row["player_id"]),
        target_team_id=int(target_row["team_id"]),
        top_k=top_k,
        metric=metric_choice,
        representation=rep_choice,
        by_group=by_group_pct,
        same_position_group=same_group,
        include_same_team=inc_teammates,
    )

    if neighbours.empty:
        st.warning("No candidate neighbours found for the selected constraints.")
    else:
        neighbours_scored = score_archetypes(neighbours, by_group=by_group_pct)
        # Preserve original k-NN percentiles from scored_pool
        st.markdown(
            f"#### Top {len(neighbours)} Most Similar Players to **{target_row['player_name']}**"
        )
        display_cols = [
            "rank",
            "player_name",
            "team_name",
            "competition_name",
            "primary_position",
            "archetype",
            "minutes",
            "similarity_pct",
            "distance",
            "npxg_p90",
            "xt_p90",
            "xgbuildup_p90",
            "vaep_p90",
        ]
        # Attach archetype from scored_pool
        arch_Lookup = scored_pool[["player_id", "team_id", "archetype"]].drop_duplicates()
        knn_table = neighbours.drop(columns=["archetype"], errors="ignore").merge(
            arch_Lookup, on=["player_id", "team_id"], how="left"
        )
        st.dataframe(
            knn_table[display_cols],
            hide_index=True,
            width="stretch",
            height="content",
            column_config={
                "rank": st.column_config.NumberColumn("#", format="%d"),
                "player_name": st.column_config.TextColumn("Player"),
                "team_name": st.column_config.TextColumn("Team"),
                "competition_name": st.column_config.TextColumn("League"),
                "primary_position": st.column_config.TextColumn("Position"),
                "archetype": st.column_config.TextColumn("Primary Archetype"),
                "minutes": st.column_config.NumberColumn("Min", format="%.0f"),
                "similarity_pct": st.column_config.ProgressColumn(
                    "Similarity %",
                    help="100% = identical feature vector in the chosen space",
                    format="%.1f%%",
                    min_value=0.0,
                    max_value=100.0,
                ),
                "distance": st.column_config.NumberColumn("Distance", format="%.3f"),
                "npxg_p90": st.column_config.NumberColumn("npxG/90", format="%.2f"),
                "xt_p90": st.column_config.NumberColumn("Singh xT/90", format="%.2f"),
                "xgbuildup_p90": st.column_config.NumberColumn("xGBuildup/90", format="%.2f"),
                "vaep_p90": st.column_config.NumberColumn("VAEP/90", format="%.2f"),
            },
        )

        st.divider()
        st.markdown("#### 1x1 Scouting Overlay & Attribute Similarity Breakdown")
        neigh_labels = knn_table.apply(
            lambda r: (
                f"#{int(r['rank'])} [{r['team_name']}] {r['player_name']} "
                f"({r['similarity_pct']:.1f}% similar)"
            ),
            axis=1,
        )
        sel_neigh_idx = st.selectbox(
            "Inspect Neighbour in Detail",
            range(len(knn_table)),
            format_func=lambda i: neigh_labels.iloc[i],
        )
        cand_row = knn_table.iloc[sel_neigh_idx]

        col_pizza, col_deltas = st.columns([5, 5])
        with col_pizza:
            fig_comp = plot_comparison_pizza(pizza_data(target_row), pizza_data(cand_row))
            st.plotly_chart(fig_comp, width="stretch")

        with col_deltas:
            deltas = metric_deltas(target_row, cand_row)
            st.markdown(
                f"##### ✅ Closest Matching Attributes "
                f"(`{target_row['player_name']}` vs `{cand_row['player_name']}`)"
            )
            st.caption("Smallest absolute percentile differences across the 19 scouting metrics.")
            closest_df = deltas.head(7)[
                [
                    "category",
                    "label",
                    "target_p90",
                    "candidate_p90",
                    "target_pctile",
                    "candidate_pctile",
                    "pctile_diff",
                ]
            ].rename(
                columns={
                    "category": "Dimension",
                    "label": "Metric",
                    "target_p90": f"{target_row['player_name']} /90",
                    "candidate_p90": f"{cand_row['player_name']} /90",
                    "target_pctile": "Target Pct",
                    "candidate_pctile": "Match Pct",
                    "pctile_diff": "Δ Pct",
                }
            )
            st.dataframe(
                closest_df,
                hide_index=True,
                width="stretch",
                height="content",
                column_config={
                    f"{target_row['player_name']} /90": st.column_config.NumberColumn(
                        format="%.2f"
                    ),
                    f"{cand_row['player_name']} /90": st.column_config.NumberColumn(format="%.2f"),
                    "Target Pct": st.column_config.NumberColumn(format="%.0f"),
                    "Match Pct": st.column_config.NumberColumn(format="%.0f"),
                    "Δ Pct": st.column_config.NumberColumn(format="%+.1f"),
                },
            )

            st.markdown("##### ⚠️ Largest Attribute Divergences")
            st.caption("Metrics where the candidate diverges most from the target profile.")
            diverging_df = (
                deltas.sort_values("abs_pctile_diff", ascending=False)
                .head(6)[
                    [
                        "category",
                        "label",
                        "target_p90",
                        "candidate_p90",
                        "target_pctile",
                        "candidate_pctile",
                        "pctile_diff",
                    ]
                ]
                .rename(
                    columns={
                        "category": "Dimension",
                        "label": "Metric",
                        "target_p90": f"{target_row['player_name']} /90",
                        "candidate_p90": f"{cand_row['player_name']} /90",
                        "target_pctile": "Target Pct",
                        "candidate_pctile": "Match Pct",
                        "pctile_diff": "Δ Pct",
                    }
                )
            )
            st.dataframe(
                diverging_df,
                hide_index=True,
                width="stretch",
                height="content",
                column_config={
                    f"{target_row['player_name']} /90": st.column_config.NumberColumn(
                        format="%.2f"
                    ),
                    f"{cand_row['player_name']} /90": st.column_config.NumberColumn(format="%.2f"),
                    "Target Pct": st.column_config.NumberColumn(format="%.0f"),
                    "Match Pct": st.column_config.NumberColumn(format="%.0f"),
                    "Δ Pct": st.column_config.NumberColumn(format="%+.1f"),
                },
            )

with tab_arch:
    a_left, a_right = st.columns([1.05, 1.15])
    with a_left:
        st.markdown(f"#### Tactical Role Spectrum: **{target_row['player_name']}**")
        st.caption(
            "Each archetype is an explainable weighted average of within-pool percentiles "
            "(0–100). Players often exhibit hybrid profiles across two adjacent roles."
        )
        prof_df = player_archetype_profile(target_row)
        fig_arch = px.bar(
            prof_df.sort_values("Score", ascending=True),
            x="Score",
            y="Archetype",
            orientation="h",
            text="Score",
            color="Score",
            color_continuous_scale="Tealgrn",
            range_x=[0, 105],
            template="plotly_dark",
            height=380,
        )
        fig_arch.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig_arch.update_layout(
            coloraxis_showscale=False, yaxis_title=None, xaxis_title="Score (0–100)"
        )
        st.plotly_chart(fig_arch, width="stretch")

        st.dataframe(
            prof_df,
            hide_index=True,
            width="stretch",
            height="content",
            column_config={
                "Score": st.column_config.ProgressColumn(
                    "Score (0–100)", format="%.1f", min_value=0.0, max_value=100.0
                ),
            },
        )

    with a_right:
        st.markdown("#### Pool Distribution & Top Specialists by Archetype")
        dist_df = (
            scored_pool.groupby(["archetype", "position_group"], as_index=False)
            .size()
            .rename(columns={"size": "Players"})
        )
        fig_dist = px.bar(
            dist_df,
            x="Players",
            y="archetype",
            color="position_group",
            orientation="h",
            template="plotly_dark",
            height=380,
            labels={
                "archetype": "Primary Archetype",
                "position_group": "Position Group",
            },
        )
        fig_dist.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_dist, width="stretch")

        arch_names = [spec.name for spec in ARCHETYPES]
        default_arch_idx = (
            arch_names.index(str(target_row["archetype"]))
            if str(target_row["archetype"]) in arch_names
            else 0
        )
        chosen_arch = st.selectbox(
            "Inspect Top Specialists for Archetype:", arch_names, index=default_arch_idx
        )
        arch_col = f"arch_{chosen_arch}"
        leaders_arch = (
            scored_pool.sort_values(arch_col, ascending=False)
            .head(10)[
                [
                    "player_name",
                    "team_name",
                    "competition_name",
                    "primary_position",
                    "minutes",
                    arch_col,
                    "secondary_archetype",
                ]
            ]
            .rename(
                columns={
                    "player_name": "Player",
                    "team_name": "Team",
                    "competition_name": "League",
                    "primary_position": "Position",
                    "minutes": "Min",
                    arch_col: "Role Score",
                    "secondary_archetype": "Secondary Role",
                }
            )
        )
        st.dataframe(
            leaders_arch,
            hide_index=True,
            width="stretch",
            height="content",
            column_config={
                "Min": st.column_config.NumberColumn(format="%.0f"),
                "Role Score": st.column_config.ProgressColumn(
                    "Role Score", format="%.1f", min_value=0.0, max_value=100.0
                ),
            },
        )

with tab_ml:
    m1, m2, m3, m4 = st.columns([1.4, 1.6, 1.5, 1.3])
    n_clusters = int(m1.slider("Number of Clusters (k)", 2, 8, 5))
    ml_rep = m2.selectbox(
        "ML Feature Matrix",
        options=["zscore", "percentiles"],
        format_func=lambda r: (
            "Standardised Per-90 (Z-Score)" if r == "zscore" else "Percentiles (0–100, Scaled)"
        ),
    )
    color_mode = m3.selectbox(
        "Color PCA Biplot By",
        options=["cluster_label", "position_group", "archetype"],
        format_func=lambda c: {
            "cluster_label": "K-Means Cluster",
            "position_group": "Position Group",
            "archetype": "Domain Archetype",
        }[c],
    )
    show_arrows = m4.toggle("Show Biplot Loading Arrows", value=True)

    pca_res = fit_pca_profile(scored_pool, representation=ml_rep, by_group=by_group_pct)
    km_res = fit_kmeans_clusters(
        scored_pool, n_clusters=n_clusters, representation=ml_rep, by_group=by_group_pct
    )

    v1, v2, v3 = (r * 100.0 for r in pca_res.explained_variance_ratio)
    mk1, mk2, mk3, mk4 = st.columns(4)
    mk1.metric(
        "PCA Variance (PC1 + PC2)",
        f"{v1 + v2:.1f}%",
        f"PC1: {v1:.1f}% · PC2: {v2:.1f}% · PC3: {v3:.1f}%",
        delta_color="off",
    )
    mk2.metric(
        "Silhouette Score",
        f"{km_res.silhouette:.3f}",
        f"k = {n_clusters} clusters",
        delta_color="off",
    )
    mk3.metric(
        "Agreement vs Position Group",
        f"ARI {km_res.ari_position_group:.2f}",
        f"NMI {km_res.nmi_position_group:.2f}",
        delta_color="off",
    )
    mk4.metric(
        "Agreement vs Domain Archetype",
        f"ARI {km_res.ari_archetype:.2f}",
        f"NMI {km_res.nmi_archetype:.2f}",
        delta_color="off",
    )

    clustered_df = km_res.clustered
    fig_pca = px.scatter(
        clustered_df,
        x="pc1",
        y="pc2",
        color=color_mode,
        hover_name="player_name",
        hover_data={
            "team_name": True,
            "primary_position": True,
            "cluster_label": True,
            "archetype": True,
            "minutes": ":.0f",
            "pc1": ":.2f",
            "pc2": ":.2f",
        },
        labels={
            "pc1": f"PC1 ({v1:.1f}% variance)",
            "pc2": f"PC2 ({v2:.1f}% variance)",
            "cluster_label": "K-Means Cluster",
            "position_group": "Position Group",
            "archetype": "Domain Archetype",
        },
        template="plotly_dark",
        height=560,
    )
    fig_pca.update_traces(marker={"size": 7, "opacity": 0.68})

    # Highlight Target Player and Top-K k-NN neighbours on the PCA manifold
    target_pca = clustered_df[
        (clustered_df["player_id"] == int(target_row["player_id"]))
        & (clustered_df["team_id"] == int(target_row["team_id"]))
    ]
    if not neighbours.empty:
        neigh_keys = set(
            zip(
                neighbours["player_id"].astype(int), neighbours["team_id"].astype(int), strict=False
            )
        )
        neigh_pca = clustered_df[
            clustered_df.apply(
                lambda r: (int(r["player_id"]), int(r["team_id"])) in neigh_keys, axis=1
            )
        ]
        if not neigh_pca.empty:
            fig_pca.add_trace(
                go.Scatter(
                    x=neigh_pca["pc1"],
                    y=neigh_pca["pc2"],
                    mode="markers",
                    marker={
                        "size": 12,
                        "symbol": "diamond",
                        "color": "#00E5FF",
                        "line": {"width": 1.5, "color": "#FFFFFF"},
                    },
                    text=neigh_pca["player_name"],
                    name=f"Top {len(neigh_pca)} k-NN Neighbours",
                    hovertemplate=(
                        "<b>%{text}</b> (k-NN Neighbour)<br>"
                        "PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>"
                    ),
                )
            )

    if not target_pca.empty:
        fig_pca.add_trace(
            go.Scatter(
                x=target_pca["pc1"],
                y=target_pca["pc2"],
                mode="markers+text",
                marker={
                    "size": 16,
                    "symbol": "star",
                    "color": "#FFB800",
                    "line": {"width": 1.8, "color": "#FFFFFF"},
                },
                text=[str(target_row["player_name"])],
                textposition="top center",
                textfont={"size": 12, "color": "#FFB800"},
                name=f"Target: {target_row['player_name']}",
                hovertemplate=(
                    "<b>%{text}</b> (Target)<br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>"
                ),
            )
        )

    if show_arrows:
        scale_x = float(clustered_df["pc1"].abs().quantile(0.95)) * 1.8
        scale_y = float(clustered_df["pc2"].abs().quantile(0.95)) * 1.8
        scale = max(scale_x, scale_y, 2.0)
        top_loadings = pca_res.loadings.head(8)
        for _, lrow in top_loadings.iterrows():
            x_end = float(lrow["pc1"]) * scale
            y_end = float(lrow["pc2"]) * scale
            fig_pca.add_annotation(
                x=x_end,
                y=y_end,
                ax=0,
                ay=0,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1.4,
                arrowcolor="rgba(255, 255, 255, 0.55)",
                text=str(lrow["label"]),
                font={"size": 10.5, "color": "#E0E6ED"},
            )

    st.plotly_chart(fig_pca, width="stretch")

    st.markdown("#### K-Means Cluster Summary & Centroid Profiles")
    st.dataframe(
        km_res.summary.drop(columns=["cluster_id"]).rename(
            columns={
                "cluster_label": "Cluster",
                "players": "Players",
                "top_position_group": "Modal Position Group",
                "top_archetype": "Modal Archetype",
                "signature_metrics": "Top 3 Signature Metrics (Mean Percentile)",
                "mean_minutes": "Mean Min",
                "mean_xt_p90": "Mean Singh xT/90",
                "mean_vaep_p90": "Mean VAEP/90",
            }
        ),
        hide_index=True,
        width="stretch",
        height="content",
        column_config={
            "Mean Min": st.column_config.NumberColumn(format="%.0f"),
            "Mean Singh xT/90": st.column_config.NumberColumn(format="%.3f"),
            "Mean VAEP/90": st.column_config.NumberColumn(format="%.3f"),
        },
    )

    fig_cent = px.imshow(
        km_res.centroids_pctile,
        text_auto=".0f",
        aspect="auto",
        color_continuous_scale="Viridis",
        zmin=0,
        zmax=100,
        template="plotly_dark",
        height=340,
        labels={"x": "Scouting Metric", "y": "K-Means Cluster", "color": "Mean Pct"},
    )
    st.plotly_chart(fig_cent, width="stretch")

    st.markdown("#### K-Means Hyperparameter Diagnostics ($k = 2 \\dots 8$) & PCA Loadings")
    diag_df = evaluate_kmeans_range(
        scored_pool, k_values=(2, 3, 4, 5, 6, 7, 8), representation=ml_rep, by_group=by_group_pct
    )
    d_left, d_right = st.columns(2)
    with d_left:
        fig_sil = px.line(
            diag_df,
            x="k",
            y=[
                "silhouette",
                "ari_position_group",
                "nmi_position_group",
                "ari_archetype",
                "nmi_archetype",
            ],
            markers=True,
            template="plotly_dark",
            height=380,
            labels={"k": "Number of Clusters (k)", "value": "Score", "variable": "Metric"},
            title="Silhouette & Agreement vs Position / Archetypes",
        )
        st.plotly_chart(fig_sil, width="stretch")

    with d_right:
        fig_elbow = px.line(
            diag_df,
            x="k",
            y="inertia",
            markers=True,
            template="plotly_dark",
            height=380,
            labels={"k": "Number of Clusters (k)", "inertia": "Within-Cluster Inertia (SSE)"},
            title="K-Means Elbow Curve (Within-Cluster Sum of Squares)",
        )
        st.plotly_chart(fig_elbow, width="stretch")

    with st.expander("Inspect Full PCA Component Loadings (PC1, PC2, PC3)"):
        st.dataframe(
            pca_res.loadings.rename(
                columns={
                    "category": "Dimension",
                    "label": "Metric",
                    "column": "Column",
                    "pc1": f"PC1 ({v1:.1f}%)",
                    "pc2": f"PC2 ({v2:.1f}%)",
                    "pc3": f"PC3 ({v3:.1f}%)",
                    "loading_norm_2d": "2D Biplot Norm",
                }
            ),
            hide_index=True,
            width="stretch",
            height="content",
        )

methodology(
    """
    - **Why start with Euclidean & Cosine k-NN before ML (GEMINI.md §4)?** Distance in the 19-metric
      percentile or Z-score space requires zero black-box assumptions: every neighbour's similarity
      can be decomposed exactly into per-metric diffs (`Metric Deltas`).
    - **Euclidean vs Cosine distance**: Euclidean distance penalises differences in overall volume
      (comparing starters of similar output), whereas **Cosine similarity** normalises each player
      vector to unit length, matching players with the same *proportional style* even if one plays
      for a lower-possession team.
    - **Domain Archetypes vs Unsupervised K-Means**: Rule-based archetype scores provide transparent
      tactical anchors. Comparing K-Means clusters against `position_group` and `archetype` via
      **Adjusted Rand Index (ARI)** and **Normalized Mutual Information (NMI)** quantifies how much
      unsupervised geometry recovers known football roles versus hybrid tactical profiles.
    - **Limitations**: Single season (2015/16), minimum-minutes sensitivity, team-style bias in
      volume metrics, and absence of off-ball tracking data.
    """
)
