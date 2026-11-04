"""Shared page setup: page config, data credit, methodology notes and the data-missing guard."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from data_service import match_label
from statsbomb_analytics.selection import matches_with_team, team_names

CREDIT = "Data: StatsBomb (Open Data, 2015/16). Academic, non-commercial project."


def setup_page(title: str, icon: str, subtitle: str | None = None) -> None:
    """Configure the page, render the title and the data credit in the sidebar."""
    st.set_page_config(page_title=f"{title} · StatsBomb Analytics", page_icon=icon, layout="wide")
    st.title(f"{icon} {title}")
    if subtitle:
        st.caption(subtitle)
    st.sidebar.caption(CREDIT)


def methodology(text: str) -> None:
    """Collapsed methodology/limitations note (project rule: every method documents its limits)."""
    with st.expander("Methodology and limitations"):
        st.markdown(text)


def require(df, message: str):
    """Stop the page with a clear message when the data is missing or empty."""
    if df is None or len(df) == 0:
        st.warning(message)
        st.stop()
    return df


ALL_TEAMS = "All teams"


def match_picker(matches: pd.DataFrame) -> tuple[pd.Series, str]:
    """Competition -> team -> match selectors. Returns the match and the chosen team name.

    The team selector keeps only the matches in which the team played; `ALL_TEAMS` keeps all.
    """
    col1, col2, col3 = st.columns([1, 1, 2])
    league = col1.selectbox("Competition", sorted(matches["competition_name"].unique()))
    pool = matches[matches["competition_name"] == league].reset_index(drop=True)
    team = col2.selectbox("Team", [ALL_TEAMS, *team_names(pool)])
    if team != ALL_TEAMS:
        pool = matches_with_team(pool, team)
    labels = pool.apply(match_label, axis=1)
    choice = col3.selectbox("Match", range(len(pool)), format_func=lambda i: labels.iloc[i])
    return pool.iloc[choice], team
