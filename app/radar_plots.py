"""Plotly Barpolar pizza charts for the Scouting Radar page (single player and 1x1 comparison)."""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

from statsbomb_analytics.metrics.scouting import CATEGORY_COLORS

TEXT_COLOR = "#F0F2F6"
SUBTEXT_COLOR = "#A3A8B8"


def _polar_layout(angles: list[float], params: list[str], barmode: str | None = None) -> dict:
    polar: dict[str, Any] = {
        "bgcolor": "rgba(0,0,0,0)",
        "radialaxis": {
            "range": [0, 100],
            "showticklabels": True,
            "tickvals": [25, 50, 75, 100],
            "ticktext": ["25", "50", "75", "100"],
            "gridcolor": "rgba(255, 255, 255, 0.12)",
            "linecolor": "rgba(255, 255, 255, 0.22)",
            "tickfont": {"size": 10, "color": SUBTEXT_COLOR},
        },
        "angularaxis": {
            "rotation": 90,
            "direction": "clockwise",
            "tickmode": "array",
            "tickvals": angles,
            "ticktext": [p.replace("\n", "<br>") for p in params],
            "gridcolor": "rgba(255, 255, 255, 0.10)",
            "linecolor": "rgba(255, 255, 255, 0.22)",
            "tickfont": {"size": 11.5, "color": TEXT_COLOR},
        },
    }
    if barmode:
        polar["barmode"] = barmode
    return polar


def plot_scouting_pizza(data: dict[str, Any]) -> go.Figure:
    """Interactive Barpolar pizza chart for a single player."""
    params = data["params"]
    values = data["values"]
    colors = data["colors"]
    p_name = data["player_name"]
    t_name = data["team_name"]
    c_name = data.get("competition_name", "")
    pos = data.get("position") or ""
    mins = int(round(data["minutes"]))

    metric_names = data.get("metric_names", params)
    categories = data.get("categories", ["General"] * len(params))
    p90_labels = data.get("p90_labels", [str(v) for v in values])
    raw_totals = data.get("raw_totals", ["—"] * len(params))

    n = len(params)
    slice_width = 360.0 / n if n > 0 else 360.0
    angles = [i * slice_width for i in range(n)]

    fig = go.Figure()

    for cat_name, cat_color in CATEGORY_COLORS.items():
        fig.add_trace(
            go.Scatterpolar(
                r=[None],
                theta=[None],
                mode="markers",
                marker={"size": 11, "color": cat_color, "symbol": "square"},
                name=cat_name,
                showlegend=True,
                hoverinfo="skip",
            )
        )

    customdata = [
        [m_name, cat, p90, raw, val]
        for m_name, cat, p90, raw, val in zip(
            metric_names, categories, p90_labels, raw_totals, values, strict=False
        )
    ]

    fig.add_trace(
        go.Barpolar(
            r=values,
            theta=angles,
            width=[slice_width] * n,
            marker_color=colors,
            marker_line_color="rgba(255, 255, 255, 0.2)",
            marker_line_width=1.5,
            opacity=0.88,
            customdata=customdata,
            hovertemplate=(
                "<b>%{customdata[0]}</b> (%{customdata[1]})<br>"
                "Percentile: <b>%{customdata[4]}</b><br>"
                "Per 90 min: <b>%{customdata[2]}</b><br>"
                "Season total: <b>%{customdata[3]}</b><extra></extra>"
            ),
            showlegend=False,
        )
    )

    text_r = [max(18.0, v * 0.72) for v in values]
    fig.add_trace(
        go.Scatterpolar(
            r=text_r,
            theta=angles,
            mode="text",
            text=[str(v) for v in values],
            textfont={"size": 12, "color": "#FFFFFF", "family": "Arial Black"},
            showlegend=False,
            hoverinfo="skip",
        )
    )

    pos_suffix = f" • {pos}" if pos else ""
    title_text = f"<b>{p_name} — {t_name}{pos_suffix}</b>"
    subtitle_text = f"Percentiles | {c_name} 2015/16 | {mins} min"
    sub_html = f"<span style='font-size:12.5px;color:{SUBTEXT_COLOR}'>{subtitle_text}</span>"

    fig.update_layout(
        title={
            "text": f"{title_text}<br>{sub_html}",
            "x": 0.5,
            "xanchor": "center",
            "y": 0.97,
            "yanchor": "top",
            "font": {"size": 16, "color": TEXT_COLOR},
        },
        height=700,
        polar=_polar_layout(angles, params),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": TEXT_COLOR},
        legend={
            "orientation": "h",
            "x": 0.5,
            "xanchor": "center",
            "y": -0.08,
            "font": {"color": TEXT_COLOR, "size": 11.5},
        },
        margin={"l": 50, "r": 50, "t": 95, "b": 60},
    )
    return fig


def plot_comparison_pizza(data_1: dict[str, Any], data_2: dict[str, Any]) -> go.Figure:
    """Interactive head-to-head Barpolar comparison pizza chart."""
    params_1 = data_1["params"]
    params_2 = data_2["params"]
    common_params = [p for p in params_1 if p in params_2]

    idx_1 = [params_1.index(p) for p in common_params]
    idx_2 = [params_2.index(p) for p in common_params]

    values_1 = [data_1["values"][i] for i in idx_1]
    values_2 = [data_2["values"][i] for i in idx_2]

    m_names_1 = data_1.get("metric_names", params_1)
    cats_1 = data_1.get("categories", ["General"] * len(params_1))
    p90_1 = data_1.get("p90_labels", [str(v) for v in params_1])
    raw_1 = data_1.get("raw_totals", ["—"] * len(params_1))

    m_names_2 = data_2.get("metric_names", params_2)
    cats_2 = data_2.get("categories", ["General"] * len(params_2))
    p90_2 = data_2.get("p90_labels", [str(v) for v in params_2])
    raw_2 = data_2.get("raw_totals", ["—"] * len(params_2))

    color_1 = "#FFB800"
    color_2 = "#00E5FF"

    n = len(common_params)
    slice_width = 360.0 / n if n > 0 else 360.0
    angles = [i * slice_width for i in range(n)]

    p1_name = data_1["player_name"]
    p2_name = data_2["player_name"]
    pos1 = data_1.get("position") or ""
    pos2 = data_2.get("position") or ""
    pos1_str = f" • {pos1}" if pos1 else ""
    pos2_str = f" • {pos2}" if pos2 else ""
    p1_label = f"{p1_name} ({int(round(data_1['minutes']))}m{pos1_str})"
    p2_label = f"{p2_name} ({int(round(data_2['minutes']))}m{pos2_str})"

    c1_trans = "rgba(255, 184, 0, 0.32)"
    c1_solid = "rgba(255, 184, 0, 0.88)"
    c2_trans = "rgba(0, 229, 255, 0.32)"
    c2_solid = "rgba(0, 229, 255, 0.88)"

    colors_1: list[str] = []
    colors_2: list[str] = []
    for v1, v2 in zip(values_1, values_2, strict=False):
        if v1 > v2:
            colors_1.append(c1_trans)
            colors_2.append(c2_solid)
        elif v2 > v1:
            colors_1.append(c1_solid)
            colors_2.append(c2_trans)
        else:
            colors_1.append("rgba(255, 184, 0, 0.60)")
            colors_2.append("rgba(0, 229, 255, 0.60)")

    customdata_1 = [
        [m_names_1[i], cats_1[i], p90_1[i], raw_1[i], values_1[idx]] for idx, i in enumerate(idx_1)
    ]
    customdata_2 = [
        [m_names_2[i], cats_2[i], p90_2[i], raw_2[i], values_2[idx]] for idx, i in enumerate(idx_2)
    ]

    fig = go.Figure()
    for values, colors, border, label, p_name, customdata in (
        (values_1, colors_1, color_1, p1_label, p1_name, customdata_1),
        (values_2, colors_2, color_2, p2_label, p2_name, customdata_2),
    ):
        fig.add_trace(
            go.Barpolar(
                r=values,
                theta=angles,
                name=label,
                width=[slice_width] * n,
                marker_color=colors,
                marker_line_color=border,
                marker_line_width=1.5,
                customdata=customdata,
                hovertemplate=(
                    f"<b>{p_name}</b><br>"
                    "<b>%{customdata[0]}</b> (%{customdata[1]})<br>"
                    "Percentile: <b>%{customdata[4]}</b><br>"
                    "Per 90 min: <b>%{customdata[2]}</b><br>"
                    "Season total: <b>%{customdata[3]}</b><extra></extra>"
                ),
            )
        )

    c_name = data_1.get("competition_name", "")
    sub_title_text = f"Head-to-Head Percentiles | {c_name} 2015/16"
    sub_html = f"<span style='font-size:12.5px;color:{SUBTEXT_COLOR}'>{sub_title_text}</span>"

    fig.update_layout(
        title={
            "text": f"<b>{p1_name} vs {p2_name}</b><br>{sub_html}",
            "x": 0.5,
            "xanchor": "center",
            "y": 0.97,
            "yanchor": "top",
            "font": {"size": 16, "color": TEXT_COLOR},
        },
        height=700,
        polar=_polar_layout(angles, common_params, barmode="overlay"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": TEXT_COLOR},
        legend={
            "orientation": "h",
            "x": 0.5,
            "xanchor": "center",
            "y": -0.08,
            "font": {"color": TEXT_COLOR, "size": 11.5},
        },
        margin={"l": 50, "r": 50, "t": 95, "b": 60},
    )
    return fig
