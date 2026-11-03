"""Plotly football pitch (StatsBomb coordinates: 120 x 80, attack left to right, y downwards)."""

from __future__ import annotations

import plotly.graph_objects as go

PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0
_LINE = {"color": "#9aa0a6", "width": 1.5}


def _rect(x0: float, y0: float, x1: float, y1: float) -> dict:
    return {"type": "rect", "x0": x0, "y0": y0, "x1": x1, "y1": y1, "line": _LINE, "layer": "below"}


def pitch_figure(half: bool = False, height: int = 560) -> go.Figure:
    """Empty pitch figure. `half=True` shows only the attacking half (x from 60 to 120)."""
    shapes = [
        _rect(0, 0, 120, 80),
        _rect(0, 18, 18, 62),
        _rect(102, 18, 120, 62),
        _rect(0, 30, 6, 50),
        _rect(114, 30, 120, 50),
        {"type": "line", "x0": 60, "y0": 0, "x1": 60, "y1": 80, "line": _LINE, "layer": "below"},
        {
            "type": "circle",
            "x0": 50.85,
            "y0": 30.85,
            "x1": 69.15,
            "y1": 49.15,
            "line": _LINE,
            "layer": "below",
        },
    ]
    fig = go.Figure(layout={"shapes": shapes})
    x_range = [58, 122] if half else [-2, 122]
    fig.update_xaxes(range=x_range, visible=False, constrain="domain")
    fig.update_yaxes(range=[82, -2], visible=False, scaleanchor="x", scaleratio=1)
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
        showlegend=True,
        legend={"orientation": "h", "y": -0.02},
    )
    return fig
