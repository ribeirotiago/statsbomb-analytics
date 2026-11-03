"""Shared page setup: page config, data credit, methodology notes and the data-missing guard."""

from __future__ import annotations

import streamlit as st

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
