"""Pass network of one team in one match.

Assumptions and limitations:
- Only completed passes with a known recipient are counted.
- The network covers the period until the team's first substitution (the usual convention):
  later passes involve a different set of players, so the picture would mix line-ups.
- A node is the average location of the passes made by the player; it is a rough proxy of his
  role, not of his real position.
- Edges are undirected: the passes A->B and B->A are added together.
"""

from __future__ import annotations

import pandas as pd

EDGE_COLUMNS = ["player_a", "player_b", "passes"]
NODE_COLUMNS = ["player_id", "x", "y", "passes"]


def until_first_substitution(events: pd.DataFrame) -> pd.DataFrame:
    """Events of one team before its first `Substitution` (all events if there is none)."""
    subs = events.loc[events["type"] == "Substitution", "index"]
    if subs.empty:
        return events
    return events[events["index"] < subs.min()]


def pass_network(events: pd.DataFrame, min_passes: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Nodes (`player_id, x, y, passes`) and edges (`player_a, player_b, passes`).

    `events` are the events of one team in one match (`type`, `index`, `player_id`,
    `pass_recipient_id`, `pass_outcome`, `x`, `y`). Edges below `min_passes` are dropped, and
    nodes without any remaining edge are kept only if they made at least one pass.
    """
    events = until_first_substitution(events)
    passes = events[
        (events["type"] == "Pass")
        & events["pass_outcome"].isna()
        & events["pass_recipient_id"].notna()
        & events["player_id"].notna()
    ]
    nodes = passes.groupby("player_id", as_index=False).agg(
        x=("x", "mean"), y=("y", "mean"), passes=("x", "size")
    )[NODE_COLUMNS]
    a = passes[["player_id", "pass_recipient_id"]].astype("int64")
    pair = pd.DataFrame(
        {
            "player_a": a.min(axis=1),
            "player_b": a.max(axis=1),
        }
    )
    edges = pair.groupby(["player_a", "player_b"], as_index=False).size()
    edges = edges.rename(columns={"size": "passes"})
    edges = edges[edges["passes"] >= min_passes].reset_index(drop=True)
    return nodes, edges[EDGE_COLUMNS]
