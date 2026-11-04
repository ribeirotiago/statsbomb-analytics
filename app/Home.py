"""StatsBomb Analytics: entry page and hierarchical navigation router of the Streamlit app."""

from __future__ import annotations

import streamlit as st

from common import CREDIT, methodology, setup_page
from data_service import load_matches, load_player_match_metrics, load_player_metrics


def overview_page() -> None:
    """Render the landing Overview page."""
    setup_page("StatsBomb Analytics", "⚽", "Exploratory football analytics on StatsBomb Open Data")

    st.markdown(
        """
        Exploratory and methodological analyses of the **2015/16** season across Europe's top 5
        leagues. The interface is organised into **three analytical sections** in the left sidebar:

        #### 🏟️ 1. Match Analysis
        - **Match Report**: score, team numbers (including match **PPDA**), cumulative **xG race**,
          interactive **Shot Map / Shot Table** tabs, and **Top 3 Contributors per Team** across all
          five action-valuation models.
        - **Pass Network**: average pass locations and connection volume until the first
          substitution.

        #### 🏃 2. Player Analysis
        - **Player Match Report**: single-match deep dive with action-valuation KPIs, 4-category
          stat breakdown and **5 tactical pitch maps** (Passes & Carries xT Gain / xT Loss,
          Shots, Defensive Actions with xTP, and KDE Touch Heatmap).
        - **Player Match Log & Valuation**: match-by-match trajectory across the season comparing
          how **Singh xT**, **xGChain**, **xGBuildup**, **Action xT** and **VAEP** evaluate each
          game, plus pass/carry and offensive/defensive decompositions.
        - **Scouting Radar**: 19-slice 4-dimension `Barpolar` pizza profile against positional
          peers, with 1x1 head-to-head comparison and a `ProgressColumn` breakdown table.
        - **Similar Players & Profiles**: baseline-first **Euclidean & Cosine k-NN** similarity with
          1x1 pizza overlay and attribute delta breakdown, **7 domain tactical archetypes**, and
          exploratory **PCA biplot & K-Means clustering** (Silhouette, Elbow, ARI/NMI agreement).
        - **Shot Map & Finishing**: season shots on the pitch by league, team or player
          (`Goals - xG`).

        #### 🏆 3. Season Analysis
        - **Action Valuation Comparison**: academic comparison of the five action-valuation
          paradigms — conceptual taxonomy, positional bias (`mean_percentile_by_group`), pairwise
          **Spearman rank correlation** (with/without goalkeepers) and **Rank Shifts** explorer.
        - **League Explorer**: full player per-90 table (`height=600`) and quadrant scatter plot.
        """
    )

    matches = load_matches()
    players = load_player_metrics()
    appearances = load_player_match_metrics()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Matches", f"{len(matches):,}")
    col2.metric("Players with minutes", f"{0 if players is None else len(players):,}")
    col3.metric("Player-match appearances", f"{0 if appearances is None else len(appearances):,}")
    col4.metric("Competitions", matches["competition_name"].nunique())

    methodology(
        """
        - **One season (15/16)**: no temporal evolution across seasons; the Champions League has a
          single match (the final) and is excluded from aggregate season tables.
        - **Small samples**: per-90 rates are inflated for players with few minutes. Season tables,
          radars and method comparisons default to a 300-minute minimum.
        - **Team style bias**: volume metrics (passes, Singh xT, Action xT, xGChain) favour
          possession-heavy teams.
        - **Cross-league comparisons** are not calibrated for league strength.
        - Detailed methods and validation: `docs/phase2_metrics.md` and `docs/phase3_app.md`.
        """
    )
    st.caption(CREDIT)


pg = st.navigation(
    {
        "Overview": [
            st.Page(overview_page, title="Home", icon="⚽", default=True),
        ],
        "Match Analysis": [
            st.Page("pages/1_Match_Report.py", title="Match Report", icon="📋"),
            st.Page("pages/3_Pass_Network.py", title="Pass Network", icon="🕸️"),
        ],
        "Player Analysis": [
            st.Page("pages/6_Player_Match_Report.py", title="Player Match Report", icon="🔍"),
            st.Page(
                "pages/7_Player_Match_Log.py",
                title="Player Match Log & Valuation",
                icon="📈",
            ),
            st.Page("pages/4_Scouting_Radar.py", title="Scouting Radar", icon="🧭"),
            st.Page(
                "pages/9_Similar_Players.py",
                title="Similar Players & Profiles",
                icon="🧬",
            ),
            st.Page("pages/2_Shot_Map.py", title="Shot Map & Finishing", icon="🎯"),
        ],
        "Season Analysis": [
            st.Page(
                "pages/8_Valuation_Comparison.py",
                title="Action Valuation Comparison",
                icon="⚖️",
            ),
            st.Page("pages/5_League_Explorer.py", title="League Explorer", icon="📊"),
        ],
    }
)
pg.run()
