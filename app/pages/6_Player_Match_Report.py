"""Player match report: single-match KPIs, full stat breakdown and 5 tactical pitch maps."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from common import ALL_TEAMS, methodology, require, setup_page
from data_service import (
    MISSING_METRICS,
    load_event_match_ids,
    load_match_events,
    load_player_match_metrics,
    load_xt_grid,
)
from match_plots import (
    plot_defensive_actions_plotly,
    plot_heatmap_mpl,
    plot_passes_carries_mpl,
    plot_passes_carries_plotly,
    plot_player_shots_plotly,
)
from statsbomb_analytics.metrics.player_match import (
    match_player_detailed_stats,
    score_match_spatial_events,
)

setup_page(
    "Player Match Report",
    "🔍",
    "Individual match performance: action-valuation KPIs, detailed stats and 5 tactical pitch maps",
)

match_table = require(load_player_match_metrics(), MISSING_METRICS)
event_ids = load_event_match_ids()
total_matches = match_table["match_id"].nunique()
if 0 < len(event_ids) < total_matches:
    match_table = match_table[match_table["match_id"].isin(event_ids)].reset_index(drop=True)
    st.caption(
        f"📦 Lightweight deploy subset active: **{len(event_ids)}** featured matches with full "
        f"pitch event streams available (out of **{total_matches:,}** season matches)."
    )

col1, col2, col3, col4 = st.columns([1.3, 1.4, 1.8, 2.2])
leagues = sorted(match_table["competition_name"].dropna().unique())
league = col1.selectbox("Competition", leagues)
in_league = match_table[match_table["competition_name"] == league]

teams = [ALL_TEAMS, *sorted(in_league["team_name"].dropna().unique())]
team = col2.selectbox("Team", teams)
in_team = in_league if team == ALL_TEAMS else in_league[in_league["team_name"] == team]

players = (
    in_team.groupby(["player_id", "player_name", "team_name"], as_index=False)
    .agg(matches=("match_id", "nunique"), minutes=("minutes", "sum"))
    .sort_values(["player_name", "team_name"])
    .reset_index(drop=True)
)
require(players, "No players found for the selected filters.")


def _player_label(idx: int) -> str:
    r = players.iloc[idx]
    prefix = f"[{r['team_name']}] " if team == ALL_TEAMS else ""
    return f"{prefix}{r['player_name']} ({int(r['matches'])} matches · {r['minutes']:.0f}m)"


p_idx = col3.selectbox("Player", range(len(players)), format_func=_player_label)
chosen_player = players.iloc[p_idx]
player_id = int(chosen_player["player_id"])
player_name = str(chosen_player["player_name"])

player_matches = (
    in_team[
        (in_team["player_id"] == player_id) & (in_team["team_name"] == chosen_player["team_name"])
    ]
    .sort_values("match_date", ascending=False)
    .reset_index(drop=True)
)
require(player_matches, f"No matches found for {player_name}.")


def _match_label(idx: int) -> str:
    r = player_matches.iloc[idx]
    date = str(r["match_date"])[:10]
    return (
        f"{date} · {r['home_team']} {int(r['home_score'])}-{int(r['away_score'])} "
        f"{r['away_team']} ({r['minutes']:.0f}m)"
    )


m_idx = col4.selectbox("Match", range(len(player_matches)), format_func=_match_label)
selected = player_matches.iloc[m_idx]
match_id = int(selected["match_id"])

st.divider()
st.subheader(f"📊 {player_name} ({selected['team_name']}) · {_match_label(m_idx)}")

events = load_match_events(match_id)
xt_grid = load_xt_grid()
detailed = match_player_detailed_stats(events, player_id)
spatial = score_match_spatial_events(events, player_id, grid=xt_grid)

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("⏱ Minutes", f"{selected['minutes']:.0f}'")
k2.metric("⚽ Goals", int(selected["goals"]))
k3.metric("🎯 Assists", int(selected["assists"]))
k4.metric("xG (npxG)", f"{selected['xg']:.2f} ({selected['npxg']:.2f})")
k5.metric("Key Passes", int(selected["key_passes"]))
k6.metric("Position", str(selected["position"]))

st.markdown("##### Match Action-Valuation Summary")
v1, v2, v3, v4, v5 = st.columns(5)
v1.metric(
    "Singh xT",
    f"{selected['xt']:+.3f}",
    help=f"Passes: {selected['xt_pass']:+.3f} · Carries: {selected['xt_carry']:+.3f}",
)
v2.metric("xGChain", f"{selected['xgchain']:.3f}")
v3.metric("xGBuildup", f"{selected['xgbuildup']:.3f}")
v4.metric(
    "Action xT",
    f"{selected['axt']:+.3f}",
    help=f"Passes: {selected['axt_pass']:+.3f} · Carries: {selected['axt_carry']:+.3f}",
)
v5.metric(
    "VAEP (Out-of-Fold)",
    f"{selected['vaep']:+.3f}",
    help=(
        f"Offensive: {selected['vaep_offensive']:+.3f} · "
        f"Defensive: {selected['vaep_defensive']:+.3f}"
    ),
)

with st.expander("📋 Complete Match Statistics", expanded=False):
    c_att, c_pas, c_def, c_prg = st.columns(4)
    att = detailed["attack"]
    pas = detailed["passes"]
    dfn = detailed["defense"]
    prg = detailed["progression"]

    with c_att:
        st.markdown("#### ⚡ Attack & Finishing")
        st.write(f"**xG / npxG:** {att['xg']:.2f} / {att['npxg']:.2f}")
        st.write(f"**Shots (on target / total):** {att['shots_on_target']} / {att['shots']}")
        st.write(f"**Goals:** {att['goals']}")
        st.write(f"**Key Passes (Shot Assists):** {att['key_passes']}")
        st.write(f"**Assists:** {att['assists']}")
        st.write(
            f"**Dribbles (completed / total):** {att['dribbles_completed']} / {att['dribbles']}"
        )

    with c_pas:
        st.markdown("#### 🎯 Passes & Distribution")
        st.write(
            f"**Passes (completed / total):** {pas['passes_completed']} / {pas['passes']} "
            f"({pas['pass_pct']:.1f}%)"
        )
        st.write(f"**Progressive Passes:** {pas['progressive_passes']}")
        st.write(f"**Crosses (completed / total):** {pas['crosses_completed']} / {pas['crosses']}")
        st.write(f"**Switches of Play:** {pas['switches']}")
        st.write(f"**Through Balls:** {pas['through_balls']}")
        st.write(f"**On-Ball Touches:** {pas['touches']}")

    with c_def:
        st.markdown("#### 🛡️ Defense & Duels")
        st.write(f"**Tackles (won / total):** {dfn['tackles_won']} / {dfn['tackles']}")
        st.write(f"**Interceptions:** {dfn['interceptions']}")
        st.write(f"**Clearances:** {dfn['clearances']}")
        st.write(f"**Ball Recoveries:** {dfn['ball_recoveries']}")
        st.write(f"**Blocks:** {dfn['blocks']}")
        st.write(
            f"**Pressures (Counterpressures):** {dfn['pressures']} ({dfn['counterpressures']})"
        )
        st.write(f"**Fouls Committed:** {dfn['fouls_committed']}")

    with c_prg:
        st.markdown("#### 🏃 Carries & Progression")
        st.write(f"**Carries:** {prg['carries']}")
        st.write(f"**Total Carry Distance:** {prg['carry_distance']:.0f} yd")
        st.write(f"**Progressive Carries:** {prg['progressive_carries']}")
        st.write(f"**Actions Under Pressure:** {prg['under_pressure']}")

st.divider()
st.markdown("### ⚙️ Visualization Controls")
f1, f2, f3 = st.columns([2, 1, 1])
with f1:
    action_types = st.multiselect(
        "Action types:",
        options=["pass", "carry", "defensive", "shot"],
        default=["pass", "carry", "defensive", "shot"],
        format_func={
            "pass": "Passes",
            "carry": "Carries",
            "defensive": "Defensive Actions",
            "shot": "Shots",
        }.get,
    )
with f2:
    show_top = st.radio(
        "Display moves:",
        options=["All", "Top 5 xT"],
        horizontal=True,
        help="'Top 5 xT' shows only the 5 actions with largest positive or negative xT impact.",
    )
with f3:
    interactive_mode = st.toggle(
        "Interactive Mode (Plotly Tooltips)",
        value=True,
        help="Toggle between interactive Plotly pitch charts and static mplsoccer charts.",
    )

with st.expander("ℹ️ Coordinate System & Pitch Reference (StatsBomb 120×80)", expanded=False):
    st.markdown(
        """
        - **Pitch Dimensions**: 120 yards (length) × 80 yards (width).
        - **Attacking Direction**: Always **left to right** (`x = 0` own goal line → `x = 120`
          opponent goal line; `x = 60` halfway line).
        - **Vertical Axis (`y`)**: `y = 0` top touchline → `y = 40` centre of the pitch/goals →
          `y = 80` bottom touchline.
        - **xT Gain vs xT Loss on Maps**:
          - Completed passes/carries receive `xT[end] - xT[start]`. Moves into more dangerous
            zones appear in **xT Gain (> 0)**; backward/lateral moves to safer zones appear in
            **xT Loss (<= 0)**.
          - Incomplete passes receive `-xT[mirror(start)]` (the threat handed to the opponent at
            that location) and appear in **xT Loss**.
          - Defensive actions receive `+xT[mirror(start)]` (**xTP** — Expected Threat Prevented).
        """
    )

moves_df = spatial["moves"]
moves_df = moves_df[moves_df["event_type"].isin(action_types)].copy()
gain_df = moves_df[moves_df["xt_value"] > 0]
loss_df = moves_df[moves_df["xt_value"] <= 0]

col_gain, col_loss = st.columns(2)
with col_gain:
    if interactive_mode:
        st.plotly_chart(
            plot_passes_carries_plotly(
                gain_df, "📈 Passes & Carries (xT Gain)", show_top=show_top, is_gain=True
            ),
            width="stretch",
        )
    else:
        st.markdown("**📈 Passes & Carries (xT Gain)**")
        fig_g = plot_passes_carries_mpl(gain_df, show_top=show_top, is_gain=True)
        st.pyplot(fig_g, width="stretch")
        plt.close(fig_g)

with col_loss:
    if interactive_mode:
        st.plotly_chart(
            plot_passes_carries_plotly(
                loss_df,
                "📉 Passes & Carries (xT Loss / Turnovers)",
                show_top=show_top,
                is_gain=False,
            ),
            width="stretch",
        )
    else:
        st.markdown("**📉 Passes & Carries (xT Loss / Turnovers)**")
        fig_l = plot_passes_carries_mpl(loss_df, show_top=show_top, is_gain=False)
        st.pyplot(fig_l, width="stretch")
        plt.close(fig_l)

col_shots, col_def = st.columns(2)
with col_shots:
    shots_df = spatial["shots"] if "shot" in action_types else pd.DataFrame()
    st.plotly_chart(plot_player_shots_plotly(shots_df), width="stretch")

with col_def:
    def_df = spatial["defensive"] if "defensive" in action_types else pd.DataFrame()
    st.plotly_chart(plot_defensive_actions_plotly(def_df), width="stretch")

_, col_heat, _ = st.columns([1, 2, 1])
with col_heat:
    st.markdown("**🔥 Touch Heatmap (KDE)**")
    touches_df = spatial["touches"]
    fig_h = plot_heatmap_mpl(touches_df)
    st.pyplot(fig_h, width="stretch")
    plt.close(fig_h)
    st.caption(f"{len(touches_df)} on-ball touches recorded in this match.")

methodology(
    """
    - **KPI Singh xT** (`xt` in the top summary) follows Karun Singh's canonical definition
      (sum of `xT[end] - xT[start]` over **completed** passes and carries only) so that match
      totals sum to the player's season `xt` in the League Explorer and Scouting Radar.
    - **Tactical Pitch Maps (xT Gain / xT Loss / xTP)** additionally visualise the turnover cost
      of incomplete passes (`-xT[mirror(start)]`) and the threat prevented by defensive actions
      (`+xT[mirror(start)]`).
    - **No physical tracking data**: StatsBomb Open Data 2015/16 is event data (no GPS distance
      covered or sprint speed); the 4th stat column therefore reports on-ball carry volume,
      carry distance and progressive actions.
    """
)
