"""Tactical pitch charts (Plotly and mplsoccer) for Match Report and Player Match Report.

Coordinates use the StatsBomb 120 x 80 pitch (attack left to right, y from 0 at top to 80 at
bottom).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.lines import Line2D
from mplsoccer import Pitch

from pitch import pitch_figure

PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0

PITCH_BG = "#2B4C3B"
PITCH_LINE = "rgba(255, 255, 255, 0.85)"
FIG_BG = "#0f0f1a"
TEXT_COLOR = "#F0F0F0"
SUBTEXT_COLOR = "#AAAAAA"

PASS_OK_COLOR = "#4CAF50"
PASS_FAIL_COLOR = "#F44336"
CARRY_COLOR = "#2196F3"

SHOT_GOAL_COLOR = "#FFD700"
SHOT_SAVE_COLOR = "#2196F3"
SHOT_BLOCK_COLOR = "#9E9E9E"
SHOT_POST_COLOR = "#FF9800"
SHOT_MISS_COLOR = "#F44336"

DEF_TACKLE_COLOR = "#4CAF50"
DEF_INTERCEPT_COLOR = "#2196F3"
DEF_CLEARANCE_COLOR = "#9C27B0"
DEF_RECOVERY_COLOR = "#FF9800"
DEF_BLOCK_COLOR = "#9E9E9E"
DEF_PRESSURE_COLOR = "#00BCD4"


def _title_html(title: str, subtitle: str) -> str:
    return (
        f"<b>{title}</b><br><span style='font-size:11.5px;color:{SUBTEXT_COLOR}'>{subtitle}</span>"
    )


def create_tactical_pitch_plotly(height: int = 430) -> go.Figure:
    """Create a dark-green 120x80 tactical pitch in Plotly with penalties and arcs."""
    fig = pitch_figure(half=False, height=height)
    fig.update_layout(
        plot_bgcolor=PITCH_BG,
        paper_bgcolor=FIG_BG,
        margin={"l": 10, "r": 10, "t": 55, "b": 48},
        legend={
            "orientation": "h",
            "x": 0.5,
            "xanchor": "center",
            "y": -0.12,
            "font": {"color": TEXT_COLOR, "size": 10},
            "bgcolor": "rgba(26, 26, 46, 0.85)",
            "bordercolor": "rgba(255, 255, 255, 0.2)",
            "borderwidth": 1,
        },
        hoverlabel={
            "bgcolor": "#1A1A2E",
            "font_size": 12,
            "font_family": "sans-serif",
            "font_color": "#FFFFFF",
        },
    )
    return fig


def _empty_legend(fig: go.Figure) -> None:
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker={"color": "rgba(0,0,0,0)", "opacity": 0, "size": 1},
            name=" ",
            showlegend=True,
            hoverinfo="skip",
        )
    )


def plot_match_shots_pitch(
    shots: pd.DataFrame,
    home_team_id: int,
    home_name: str,
    away_team_id: int,
    away_name: str,
) -> go.Figure:
    """Plot both teams' shots on a single 120x80 pitch (home attacks right, away attacks left)."""
    fig = create_tactical_pitch_plotly(height=480)
    if shots.empty:
        fig.update_layout(
            title={
                "text": _title_html("Match Shot Map", "No shots recorded"),
                "x": 0.5,
                "xanchor": "center",
            }
        )
        return fig

    teams = [
        (home_team_id, home_name, "#4cc9f0", False),
        (away_team_id, away_name, "#f72585", True),
    ]
    for team_id, team_name, color, mirror in teams:
        t_shots = shots[shots["team_id"] == team_id].dropna(subset=["x", "y"]).copy()
        if t_shots.empty:
            continue
        if mirror:
            t_shots["plot_x"] = PITCH_LENGTH - t_shots["x"].astype(float)
            t_shots["plot_y"] = PITCH_WIDTH - t_shots["y"].astype(float)
        else:
            t_shots["plot_x"] = t_shots["x"].astype(float)
            t_shots["plot_y"] = t_shots["y"].astype(float)

        for is_goal, suffix, symbol, line_col in (
            (False, "Shot", "circle", "white"),
            (True, "Goal", "star", "#FFD700"),
        ):
            part = t_shots[(t_shots["shot_outcome"] == "Goal") == is_goal]
            if part.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=part["plot_x"],
                    y=part["plot_y"],
                    mode="markers",
                    name=f"{team_name} · {suffix}",
                    marker={
                        "size": part["xg"].fillna(0.03) * 52 + (11 if is_goal else 7),
                        "color": "#FFD700" if is_goal else color,
                        "symbol": symbol,
                        "opacity": 0.9 if is_goal else 0.72,
                        "line": {"width": 1.4 if is_goal else 0.8, "color": line_col},
                    },
                    customdata=part[
                        ["player_name", "minute", "xg", "shot_outcome", "shot_body_part"]
                    ],
                    hovertemplate=(
                        f"<b>%{{customdata[0]}}</b> ({team_name})<br>"
                        "Min %{customdata[1]}' · xG <b>%{customdata[2]:.2f}</b><br>"
                        "%{customdata[3]} · %{customdata[4]}<extra></extra>"
                    ),
                )
            )

    subtitle = (
        f"{away_name} (attacks left ⬅)  vs  {home_name} (attacks right ➡) · "
        "Marker size = xG · ★ = Goal"
    )
    fig.update_layout(
        title={
            "text": _title_html("Shot Map by Team", subtitle),
            "x": 0.5,
            "xanchor": "center",
            "y": 0.98,
            "font": {"color": TEXT_COLOR, "size": 13},
        }
    )
    return fig


def plot_passes_carries_plotly(
    df_moves: pd.DataFrame,
    title: str,
    show_top: str = "All",
    is_gain: bool = True,
) -> go.Figure:
    """Plot passes and carries with directional arrows and xT hover tooltips."""
    fig = create_tactical_pitch_plotly()
    if df_moves.empty:
        fig.update_layout(
            title={
                "text": _title_html(title, "No actions in this slice"),
                "x": 0.5,
                "xanchor": "center",
                "font": {"color": TEXT_COLOR, "size": 13},
            }
        )
        _empty_legend(fig)
        return fig

    df = df_moves.copy()
    if show_top == "Top 5 xT":
        df = df.nlargest(5, "xt_value") if is_gain else df.nsmallest(5, "xt_value")

    total_xt = float(df["xt_value"].sum()) if "xt_value" in df.columns else 0.0

    for row in df.itertuples():
        is_pass = row.event_type == "pass"
        color = (PASS_OK_COLOR if row.outcome else PASS_FAIL_COLOR) if is_pass else CARRY_COLOR
        fig.add_annotation(
            x=row.x_end,
            y=row.y_end,
            ax=row.x_start,
            ay=row.y_start,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.0,
            arrowwidth=1.8,
            arrowcolor=color,
            opacity=0.85,
        )

    groups = [
        ("Completed Pass/Cross", PASS_OK_COLOR, (df["event_type"] == "pass") & df["outcome"]),
        ("Incomplete Pass/Cross", PASS_FAIL_COLOR, (df["event_type"] == "pass") & (~df["outcome"])),
        ("Carry", CARRY_COLOR, df["event_type"] == "carry"),
    ]
    has_traces = False
    for label, col, mask in groups:
        sub = df[mask]
        if sub.empty:
            continue
        has_traces = True
        dist = np.hypot(sub["x_end"] - sub["x_start"], sub["y_end"] - sub["y_start"])
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                line={"color": col, "width": 2.5},
                name=label,
                showlegend=True,
            )
        )
        custom = np.column_stack(
            [
                np.full(len(sub), label),
                sub["minute"].to_numpy(),
                sub["xt_value"].to_numpy(),
                sub["x_start"].to_numpy(),
                sub["y_start"].to_numpy(),
                sub["x_end"].to_numpy(),
                sub["y_end"].to_numpy(),
                dist.to_numpy(),
            ]
        )
        fig.add_trace(
            go.Scatter(
                x=sub["x_end"],
                y=sub["y_end"],
                mode="markers",
                marker={"size": 14, "color": "rgba(0,0,0,0)", "opacity": 0},
                customdata=custom,
                hovertemplate=(
                    "<b>%{customdata[0]}</b> (Min %{customdata[1]}')<br>"
                    "xT Impact: <b>%{customdata[2]:+.3f}</b><br>"
                    "Start: (%{customdata[3]:.1f}, %{customdata[4]:.1f}) → "
                    "End: (%{customdata[5]:.1f}, %{customdata[6]:.1f})<br>"
                    "Distance: %{customdata[7]:.1f} yd<extra></extra>"
                ),
                showlegend=False,
            )
        )

    if not has_traces:
        _empty_legend(fig)

    subtitle = f"Total xT in slice: <b>{total_xt:+.3f}</b> ({len(df)} actions)"
    fig.update_layout(
        title={
            "text": _title_html(title, subtitle),
            "x": 0.5,
            "xanchor": "center",
            "y": 0.98,
            "font": {"color": TEXT_COLOR, "size": 13},
        }
    )
    return fig


def plot_defensive_actions_plotly(df_def: pd.DataFrame) -> go.Figure:
    """Plot defensive actions on the 120x80 pitch with xTP (Threat Prevented) tooltips."""
    fig = create_tactical_pitch_plotly()
    if df_def.empty:
        fig.update_layout(
            title={
                "text": _title_html("🛡️ Defensive Actions", "No defensive actions recorded"),
                "x": 0.5,
                "xanchor": "center",
                "font": {"color": TEXT_COLOR, "size": 13},
            }
        )
        _empty_legend(fig)
        return fig

    total_xtp = float(df_def["xt_value"].fillna(0.0).sum())
    specs = [
        ("tackle", "Tackle", DEF_TACKLE_COLOR, "x", 12),
        ("interception", "Interception", DEF_INTERCEPT_COLOR, "diamond", 10),
        ("clearance", "Clearance", DEF_CLEARANCE_COLOR, "circle", 10),
        ("ball-recovery", "Ball Recovery", DEF_RECOVERY_COLOR, "triangle-up", 11),
        ("block", "Block", DEF_BLOCK_COLOR, "square", 10),
        ("pressure", "Pressure", DEF_PRESSURE_COLOR, "pentagon", 9),
    ]
    has_traces = False
    for subtype, label, color, symbol, size in specs:
        part = df_def[df_def["event_subtype"] == subtype]
        if part.empty:
            continue
        has_traces = True
        fig.add_trace(
            go.Scatter(
                x=part["x_start"],
                y=part["y_start"],
                mode="markers",
                marker={
                    "size": size,
                    "color": color,
                    "symbol": symbol,
                    "line": {"width": 1.1, "color": "#FFFFFF"},
                },
                name=label,
                customdata=part[["minute", "xt_value", "x_start", "y_start"]],
                hovertemplate=(
                    f"<b>{label}</b> (Min %{{customdata[0]}}')<br>"
                    "xTP (Threat Prevented): <b>+%{customdata[1]:.3f}</b><br>"
                    "Location: (%{customdata[2]:.1f}, %{customdata[3]:.1f})<extra></extra>"
                ),
            )
        )
    if not has_traces:
        _empty_legend(fig)

    subtitle = f"{len(df_def)} defensive actions · Total xTP: <b>+{total_xtp:.3f}</b>"
    fig.update_layout(
        title={
            "text": _title_html("🛡️ Defensive Actions", subtitle),
            "x": 0.5,
            "xanchor": "center",
            "y": 0.98,
            "font": {"color": TEXT_COLOR, "size": 13},
        }
    )
    return fig


def plot_player_shots_plotly(df_shots: pd.DataFrame) -> go.Figure:
    """Plot a single player's shots in a match with outcome symbols and xG tooltips."""
    fig = create_tactical_pitch_plotly()
    if df_shots.empty:
        fig.update_layout(
            title={
                "text": _title_html("🥅 Shots", "No shots recorded in this match"),
                "x": 0.5,
                "xanchor": "center",
                "font": {"color": TEXT_COLOR, "size": 13},
            }
        )
        _empty_legend(fig)
        return fig

    specs = [
        ("goal", "Goal", SHOT_GOAL_COLOR, "star", 16),
        ("save", "Saved", SHOT_SAVE_COLOR, "triangle-up", 12),
        ("block", "Blocked", SHOT_BLOCK_COLOR, "x", 12),
        ("post", "Post", SHOT_POST_COLOR, "diamond", 12),
        ("miss", "Off Target", SHOT_MISS_COLOR, "circle", 12),
    ]
    has_traces = False
    for subtype, label, color, symbol, size in specs:
        part = df_shots[df_shots["event_subtype"] == subtype]
        if part.empty:
            continue
        has_traces = True
        fig.add_trace(
            go.Scatter(
                x=part["x_start"],
                y=part["y_start"],
                mode="markers",
                marker={
                    "size": size,
                    "color": color,
                    "symbol": symbol,
                    "line": {"width": 1.2, "color": "#FFFFFF"},
                },
                name=label,
                customdata=part[["minute", "xg", "shot_body_part", "x_start", "y_start"]],
                hovertemplate=(
                    f"<b>{label}</b> (Min %{{customdata[0]}}')<br>"
                    "xG: <b>%{customdata[1]:.2f}</b> · %{customdata[2]}<br>"
                    "Location: (%{customdata[3]:.1f}, %{customdata[4]:.1f})<extra></extra>"
                ),
            )
        )
    if not has_traces:
        _empty_legend(fig)

    xg_tot = float(df_shots["xg"].sum())
    n_goals = int((df_shots["event_subtype"] == "goal").sum())
    subtitle = f"{n_goals} goal(s) from {len(df_shots)} shot(s) · Total xG: <b>{xg_tot:.2f}</b>"
    fig.update_layout(
        title={
            "text": _title_html("🥅 Shots", subtitle),
            "x": 0.5,
            "xanchor": "center",
            "y": 0.98,
            "font": {"color": TEXT_COLOR, "size": 13},
        }
    )
    return fig


def plot_heatmap_mpl(touches: pd.DataFrame) -> plt.Figure:
    """Draw a smooth KDE touch heatmap on a StatsBomb 120x80 pitch using mplsoccer."""
    pitch = Pitch(
        pitch_type="statsbomb",
        pitch_color=PITCH_BG,
        line_color="#ffffff",
        linewidth=1.2,
    )
    fig, ax = pitch.draw(figsize=(7.5, 4.8))
    fig.patch.set_facecolor(FIG_BG)
    if len(touches) >= 3:
        pitch.kdeplot(
            x=touches["x"],
            y=touches["y"],
            ax=ax,
            fill=True,
            levels=80,
            bw_adjust=0.25,
            cmap="YlOrRd",
            alpha=0.45,
            thresh=0.15,
        )
    elif not touches.empty:
        pitch.scatter(touches["x"], touches["y"], ax=ax, color="#FF9800", s=120, zorder=5)
    return fig


def plot_passes_carries_mpl(
    df_moves: pd.DataFrame, show_top: str = "All", is_gain: bool = True
) -> plt.Figure:
    """Static mplsoccer arrow map for passes and carries."""
    pitch = Pitch(pitch_type="statsbomb", pitch_color=PITCH_BG, line_color="#ffffff", linewidth=1.2)
    fig, ax = pitch.draw(figsize=(7.5, 4.8))
    fig.patch.set_facecolor(FIG_BG)
    df = df_moves.copy()
    if not df.empty and show_top == "Top 5 xT":
        df = df.nlargest(5, "xt_value") if is_gain else df.nsmallest(5, "xt_value")
    for row in df.itertuples():
        color = (
            (PASS_OK_COLOR if row.outcome else PASS_FAIL_COLOR)
            if row.event_type == "pass"
            else CARRY_COLOR
        )
        pitch.arrows(
            row.x_start,
            row.y_start,
            row.x_end,
            row.y_end,
            ax=ax,
            color=color,
            width=1.5,
            headwidth=4.0,
            headlength=4.0,
            alpha=0.85,
        )
    legend_elements = [
        Line2D([0], [0], color=PASS_OK_COLOR, lw=2, label="Completed Pass"),
        Line2D([0], [0], color=PASS_FAIL_COLOR, lw=2, label="Incomplete Pass"),
        Line2D([0], [0], color=CARRY_COLOR, lw=2, label="Carry"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower left",
        fontsize=7,
        facecolor="#1a1a2e",
        labelcolor="white",
    )
    return fig
