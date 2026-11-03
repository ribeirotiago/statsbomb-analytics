"""VAEP (Valuing Actions by Estimating Probabilities), implemented in-house with scikit-learn.

Method (Decroos et al., 2019), adapted to StatsBomb events without the `socceraction` package:

1. Actions: on-ball events of both teams (`ACTION_TYPES`), ordered by match, period and index.
2. State features: for the current action and the two previous ones (game state of the last
   3 actions): type, result, start/end coordinates, displacement, whether the same team acted,
   and elapsed time. Coordinates of the previous actions are mirrored into the current team's
   frame. The current action also has distance and angle to the goal.
3. Labels: the acting team *scores* / *concedes* a goal within the next 10 actions
   (the current action included).
4. Two gradient-boosting classifiers estimate P(score) and P(concede).
5. Action value = offensive + defensive value:
   `off = P_score(i) - P_score(i-1)`, `def = -(P_concede(i) - P_concede(i-1))`, where the
   previous state is taken from the acting team's perspective (scores/concedes swap when the
   team changes) and reset to 0 at the start of a period and after a goal.

Validation: matches are split into `n_folds` folds **by match**; every action is scored by a
model that never saw its match (out-of-fold), so values and metrics are free of leakage.
Metrics: Brier score vs. a constant baseline, ROC AUC and a calibration table.

Assumptions and limitations:
- The result of the action is a feature, so a goal gets P(score) = 1 by construction (as in
  the original paper); shots are therefore valued as `1 - previous state`.
- Own goals are not counted as goals (157 events in the season); the shootout (period 5) is
  excluded.
- Goals are rare (~1-2% of actions), so probabilities are small and per-action values noisy.
- Defenders and goalkeepers are undervalued: prevention (not conceding) leaves little trace in
  event data. Style bias: possession teams accumulate more actions.
- Per-90 rates are inflated by small minutes samples; the UI filters on minutes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score

from statsbomb_analytics.metrics.action_xt import calibration_table
from statsbomb_analytics.metrics.minutes import load_player_minutes
from statsbomb_analytics.storage.duckdb_store import query

FIELD_LENGTH = 120.0
FIELD_WIDTH = 80.0
HORIZON = 10
N_LAGS = 3
RANDOM_STATE = 123

ACTION_TYPES = (
    "Pass",
    "Carry",
    "Dribble",
    "Shot",
    "Clearance",
    "Interception",
    "Ball Recovery",
    "Block",
    "Miscontrol",
    "Dispossessed",
    "Foul Committed",
    "Duel",
    "Goal Keeper",
)

_TYPE_LIST = ", ".join(f"'{t}'" for t in ACTION_TYPES)
_WON = "('Won', 'Success', 'Success In Play', 'Success Out')"

VAEP_EVENTS_SQL = f"""
SELECT match_id, period, "index", team_id, player_id, type,
       (60 * minute + second) AS seconds, x, y,
       COALESCE(pass_end_x, carry_end_x, shot_end_x, x) AS end_x,
       COALESCE(pass_end_y, carry_end_y, shot_end_y, y) AS end_y,
       COALESCE(CASE type
           WHEN 'Pass' THEN pass_outcome IS NULL
           WHEN 'Dribble' THEN dribble_outcome = 'Complete'
           WHEN 'Shot' THEN shot_outcome = 'Goal'
           WHEN 'Duel' THEN duel_outcome IN {_WON}
           WHEN 'Interception' THEN interception_outcome IN {_WON}
           WHEN 'Ball Recovery' THEN NOT COALESCE(ball_recovery_failure, FALSE)
           WHEN 'Miscontrol' THEN FALSE
           WHEN 'Dispossessed' THEN FALSE
           WHEN 'Foul Committed' THEN FALSE
           ELSE TRUE END, FALSE) AS success,
       (type = 'Shot' AND shot_outcome = 'Goal') AS is_goal
FROM events
WHERE type IN ({_TYPE_LIST}) AND period <= 4 AND x IS NOT NULL AND player_id IS NOT NULL
ORDER BY match_id, period, "index"
"""

_LAG_FEATURES = ("type", "success", "x0", "y0", "x1", "y1", "dx", "dy", "same_team", "dt")


def label_actions(actions: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """`scores` / `concedes`: the acting team scores / concedes within the next `horizon` actions.

    `actions` must be ordered by match and play order and have `match_id, team_id, is_goal`.
    The window includes the current action and never crosses matches.
    """
    match = actions["match_id"].to_numpy()
    team = actions["team_id"].to_numpy()
    goal = actions["is_goal"].to_numpy(dtype=bool)
    n = len(actions)
    positions = np.arange(n)
    # Last row index of each match, used to cap the window.
    last = pd.Series(positions).groupby(match).transform("max").to_numpy()
    end = np.minimum(positions + horizon - 1, last)

    def window_count(flag: np.ndarray) -> np.ndarray:
        cumulative = np.concatenate([[0], np.cumsum(flag)])
        return cumulative[end + 1] - cumulative[positions]

    scores = np.zeros(n, dtype=bool)
    concedes = np.zeros(n, dtype=bool)
    for team_id in np.unique(team):
        own = team == team_id
        # Goals by `team_id` / by the opponent in each row's window (windows stay in one match).
        scored = window_count(goal & own) > 0
        against = window_count(goal & ~own) > 0
        scores[own] = scored[own]
        concedes[own] = against[own]
    out = actions.copy()
    out["scores"] = scores.astype(int)
    out["concedes"] = concedes.astype(int)
    return out


def build_features(actions: pd.DataFrame, n_lags: int = N_LAGS) -> pd.DataFrame:
    """Game-state features of the current action and the `n_lags - 1` previous ones."""
    n = len(actions)
    match = actions["match_id"].to_numpy()
    team = actions["team_id"].to_numpy()
    codes = pd.Categorical(actions["type"], categories=ACTION_TYPES).codes.astype(float)
    codes[codes < 0] = np.nan
    base = {
        "type": codes,
        "success": actions["success"].to_numpy(dtype=float),
        "x0": actions["x"].to_numpy(dtype=float),
        "y0": actions["y"].to_numpy(dtype=float),
        "x1": actions["end_x"].to_numpy(dtype=float),
        "y1": actions["end_y"].to_numpy(dtype=float),
        "seconds": actions["seconds"].to_numpy(dtype=float),
    }
    features: dict[str, np.ndarray] = {}
    for lag in range(n_lags):
        idx = np.arange(n) - lag
        valid = (idx >= 0) & (match[np.clip(idx, 0, None)] == match)
        src = np.clip(idx, 0, None)
        same = team[src] == team
        values = {k: np.where(valid, v[src], np.nan) for k, v in base.items()}
        for axis, length in (("x0", FIELD_LENGTH), ("x1", FIELD_LENGTH)):
            values[axis] = np.where(same, values[axis], length - values[axis])
        for axis, width in (("y0", FIELD_WIDTH), ("y1", FIELD_WIDTH)):
            values[axis] = np.where(same, values[axis], width - values[axis])
        values["dx"] = values["x1"] - values["x0"]
        values["dy"] = values["y1"] - values["y0"]
        values["same_team"] = np.where(valid, same.astype(float), np.nan)
        values["dt"] = np.where(valid, base["seconds"] - values["seconds"], np.nan)
        for name in _LAG_FEATURES:
            features[f"{name}_a{lag}"] = values[name]
    out = pd.DataFrame(features, index=actions.index)
    for suffix, (x, y) in {"start": ("x0_a0", "y0_a0"), "end": ("x1_a0", "y1_a0")}.items():
        dx = FIELD_LENGTH - out[x]
        dy = FIELD_WIDTH / 2 - out[y]
        out[f"dist_goal_{suffix}"] = np.hypot(dx, dy)
        out[f"angle_goal_{suffix}"] = np.arctan2(np.abs(dy), dx)
    return out


def categorical_mask(columns: list[str]) -> list[bool]:
    """Boolean mask of categorical (action type) columns for the gradient-boosting model."""
    return [c.startswith("type_") for c in columns]


def make_classifier() -> HistGradientBoostingClassifier:
    """Regularised gradient boosting; the same configuration for both labels."""
    return HistGradientBoostingClassifier(
        max_iter=100,
        learning_rate=0.1,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=RANDOM_STATE,
    )


def match_folds(match_ids: np.ndarray, n_folds: int, seed: int = RANDOM_STATE) -> np.ndarray:
    """Fold id (0..n_folds-1) per action, assigned per match so no match spans two folds."""
    matches = np.sort(np.unique(match_ids))
    rng = np.random.default_rng(seed)
    assignment = dict(zip(matches, rng.permutation(len(matches)) % n_folds, strict=True))
    return np.array([assignment[m] for m in match_ids]) if len(match_ids) else np.array([])


def out_of_fold_probabilities(
    features: pd.DataFrame, labels: pd.Series, folds: np.ndarray, n_folds: int
) -> np.ndarray:
    """Probability of `labels` per action from a model that did not train on its fold."""
    mask = categorical_mask(list(features.columns))
    probabilities = np.zeros(len(features))
    for fold in range(n_folds):
        test = folds == fold
        model = make_classifier().set_params(categorical_features=mask)
        model.fit(features[~test], labels[~test])
        probabilities[test] = model.predict_proba(features[test])[:, 1]
    return probabilities


def action_values(
    actions: pd.DataFrame, p_score: np.ndarray, p_concede: np.ndarray
) -> pd.DataFrame:
    """Offensive, defensive and total VAEP per action from the probability estimates.

    `actions` ordered by match and play order, with `match_id, period, team_id, is_goal`.
    """
    n = len(actions)
    match = actions["match_id"].to_numpy()
    period = actions["period"].to_numpy()
    team = actions["team_id"].to_numpy()
    goal = actions["is_goal"].to_numpy(dtype=bool)

    prev = np.clip(np.arange(n) - 1, 0, None)
    has_prev = (np.arange(n) > 0) & (match[prev] == match) & (period[prev] == period)
    has_prev &= ~goal[prev]  # state resets after a goal
    same = team[prev] == team
    prev_score = np.where(same, p_score[prev], p_concede[prev])
    prev_concede = np.where(same, p_concede[prev], p_score[prev])
    prev_score = np.where(has_prev, prev_score, 0.0)
    prev_concede = np.where(has_prev, prev_concede, 0.0)

    out = actions[["match_id", "team_id", "player_id", "type"]].copy()
    out["p_score"] = p_score
    out["p_concede"] = p_concede
    out["offensive"] = p_score - prev_score
    out["defensive"] = -(p_concede - prev_concede)
    out["vaep"] = out["offensive"] + out["defensive"]
    return out


def classifier_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Brier score (and constant-rate baseline), AUC and base rate."""
    base_rate = float(np.mean(y_true))
    return {
        "brier": float(brier_score_loss(y_true, y_prob)),
        "brier_baseline": float(brier_score_loss(y_true, np.full(len(y_true), base_rate))),
        "auc": float(roc_auc_score(y_true, y_prob)),
        "base_rate": base_rate,
    }


def player_vaep(values: pd.DataFrame) -> pd.DataFrame:
    """Season VAEP per (team, player), with offensive/defensive split and action count."""
    return values.groupby(["team_id", "player_id"], as_index=False).agg(
        vaep=("vaep", "sum"),
        vaep_offensive=("offensive", "sum"),
        vaep_defensive=("defensive", "sum"),
        vaep_actions=("vaep", "size"),
    )


def fit_vaep(
    actions: pd.DataFrame, n_folds: int = 5
) -> tuple[pd.DataFrame, dict[str, dict[str, float]], dict[str, pd.DataFrame]]:
    """Out-of-fold VAEP: action values, metrics and calibration tables per label."""
    labelled = label_actions(actions.reset_index(drop=True))
    features = build_features(labelled)
    folds = match_folds(labelled["match_id"].to_numpy(), n_folds)
    probabilities: dict[str, np.ndarray] = {}
    metrics: dict[str, dict[str, float]] = {}
    calibration: dict[str, pd.DataFrame] = {}
    for label in ("scores", "concedes"):
        y = labelled[label]
        probabilities[label] = out_of_fold_probabilities(features, y, folds, n_folds)
        metrics[label] = classifier_metrics(y.to_numpy(), probabilities[label])
        calibration[label] = calibration_table(y.to_numpy(), probabilities[label])
    values = action_values(labelled, probabilities["scores"], probabilities["concedes"])
    return values, metrics, calibration


def build_player_vaep(
    processed_dir: Path, n_folds: int = 5
) -> tuple[pd.DataFrame, dict[str, dict[str, float]], dict[str, pd.DataFrame]]:
    """Player table with minutes and per-90 VAEP, plus metrics and calibration tables."""
    actions = query(processed_dir, VAEP_EVENTS_SQL)
    values, metrics, calibration = fit_vaep(actions, n_folds)
    table = load_player_minutes(processed_dir).merge(
        player_vaep(values), on=["team_id", "player_id"], how="left"
    )
    for col in ("vaep", "vaep_offensive", "vaep_defensive"):
        table[col] = table[col].fillna(0.0)
        table[f"{col}_p90"] = table[col] / table["minutes"].where(table["minutes"] > 0) * 90.0
    return table, metrics, calibration
