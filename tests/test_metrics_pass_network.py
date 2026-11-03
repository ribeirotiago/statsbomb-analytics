from __future__ import annotations

import pandas as pd

from statsbomb_analytics.metrics.pass_network import pass_network, until_first_substitution


def _events() -> pd.DataFrame:
    rows = [
        # index, type, player, recipient, outcome, x, y
        (1, "Pass", 1, 2, None, 20.0, 40.0),
        (2, "Pass", 2, 1, None, 30.0, 30.0),
        (3, "Pass", 1, 2, None, 40.0, 50.0),
        (4, "Pass", 1, 3, None, 40.0, 50.0),  # single pass: below the edge threshold
        (5, "Pass", 2, 3, "Incomplete", 60.0, 40.0),  # failed pass is ignored
        (6, "Substitution", 1, None, None, None, None),
        (7, "Pass", 3, 1, None, 70.0, 40.0),  # after the first substitution
    ]
    return pd.DataFrame(
        rows,
        columns=["index", "type", "player_id", "pass_recipient_id", "pass_outcome", "x", "y"],
    )


def test_until_first_substitution_cuts_events() -> None:
    assert until_first_substitution(_events())["index"].max() == 5


def test_until_first_substitution_without_substitution() -> None:
    events = _events()
    events = events[events["type"] != "Substitution"]
    assert len(until_first_substitution(events)) == len(events)


def test_edges_are_undirected_and_thresholded() -> None:
    _, edges = pass_network(_events(), min_passes=3)
    assert edges.to_dict("records") == [{"player_a": 1, "player_b": 2, "passes": 3}]


def test_nodes_use_average_location_of_completed_passes() -> None:
    nodes, _ = pass_network(_events(), min_passes=1)
    node = nodes.set_index("player_id").loc[1]
    assert node["passes"] == 3
    assert node["x"] == (20.0 + 40.0 + 40.0) / 3


def test_players_with_only_failed_passes_are_not_nodes() -> None:
    nodes, _ = pass_network(_events(), min_passes=1)
    assert 3 not in set(nodes["player_id"])
