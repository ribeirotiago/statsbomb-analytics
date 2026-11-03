"""Action-based xT: value of each pass/carry from possession chains.

Idea from the Soccermatics course (`football/soccermatics/4_valuingActions`), rewritten for
StatsBomb events. For every pass or carry of the possession team:

    xT_action = P(possession ends in a shot | action geometry) * E[xG | shot-ending action]

- P(shot) is a logistic regression on polynomial (degree 3) features of the start/end
  coordinates and their distance to the pitch's central axis (`x0, x1, c0, c1`).
- E[xG] is a linear regression fitted only on actions of shot-ending possessions, with the
  non-penalty xG of the possession as target.
- The train/test split is done **by match** (never by action) and the P(shot) model is
  validated on held-out matches with the Brier score and ROC AUC, plus a calibration table.

Assumptions and limitations:
- The model uses geometry only (no game state, defenders or action type), so it values where
  an action happens rather than how hard it was. Failed passes/carries end the possession and
  are valued with the same model (the value of a failed action is its predicted threat, not a
  penalty); the player value therefore sums over *successful* actions only, as in `xt.py`.
- Every action of a shot-ending possession carries the same label, so the model learns
  "typical possession paths to a shot" (favours possession teams and attacking zones).
- Penalties are excluded from the possession value (see `chains.possession_values`).
- Per-90 rates are inflated by small minutes samples; the UI filters on minutes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from statsbomb_analytics.metrics.chains import possession_values
from statsbomb_analytics.metrics.minutes import load_player_minutes
from statsbomb_analytics.storage.duckdb_store import query

FIELD_LENGTH = 120.0
FIELD_WIDTH = 80.0
FEATURES = ("x0", "x1", "c0", "c1")
POLY_DEGREE = 3
TEST_FRACTION = 0.2
RANDOM_STATE = 123

ACTION_EVENTS_SQL = """
SELECT match_id, possession, possession_team_id, team_id, player_id, type,
       x, y, COALESCE(pass_end_x, carry_end_x) AS end_x,
       COALESCE(pass_end_y, carry_end_y) AS end_y,
       (type = 'Carry' OR pass_outcome IS NULL) AS success,
       shot_xg, shot_type
FROM events
WHERE type IN ('Pass', 'Carry', 'Shot')
  AND team_id = possession_team_id
  AND x IS NOT NULL
"""

_POSSESSION = ["match_id", "possession", "team_id"]


@dataclass
class ActionXTModel:
    """Fitted action-based xT models and held-out validation metrics."""

    shot_model: Pipeline
    xg_model: Pipeline
    metrics: dict[str, float]


def build_features(actions: pd.DataFrame) -> pd.DataFrame:
    """Geometry features: start/end x and distance to the central axis (all scaled to [0, 1])."""
    centre = FIELD_WIDTH / 2
    return pd.DataFrame(
        {
            "x0": actions["x"].to_numpy(dtype=float) / FIELD_LENGTH,
            "x1": actions["end_x"].to_numpy(dtype=float) / FIELD_LENGTH,
            "c0": np.abs(actions["y"].to_numpy(dtype=float) - centre) / centre,
            "c1": np.abs(actions["end_y"].to_numpy(dtype=float) - centre) / centre,
        },
        index=actions.index,
    )


def label_actions(events: pd.DataFrame) -> pd.DataFrame:
    """Passes/carries with `shot_end` (possession has a non-penalty shot) and `chain_xg`."""
    values = possession_values(events)
    actions = events[events["type"].isin(["Pass", "Carry"])].dropna(
        subset=["end_x", "end_y", "player_id"]
    )
    actions = actions.merge(values, on=_POSSESSION, how="left")
    actions["shot_end"] = actions["chain_xg"].notna().astype(int)
    actions["chain_xg"] = actions["chain_xg"].fillna(0.0)
    return actions.reset_index(drop=True)


def split_by_match(
    actions: pd.DataFrame, test_fraction: float = TEST_FRACTION, seed: int = RANDOM_STATE
) -> tuple[pd.Index, pd.Index]:
    """Train/test row labels split on `match_id`, so a match is never in both sets."""
    matches = np.sort(actions["match_id"].unique())
    rng = np.random.default_rng(seed)
    test_matches = set(
        rng.choice(matches, size=max(1, int(len(matches) * test_fraction)), replace=False)
    )
    is_test = actions["match_id"].isin(test_matches)
    return actions.index[~is_test], actions.index[is_test]


def calibration_table(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Mean predicted vs observed rate per quantile bin of predicted probability."""
    df = pd.DataFrame({"y": y_true, "p": y_prob})
    df["bin"] = pd.qcut(df["p"], q=bins, duplicates="drop")
    table = df.groupby("bin", observed=True).agg(predicted=("p", "mean"), observed=("y", "mean"))
    return table.reset_index(drop=True)


def _poly_pipeline(estimator) -> Pipeline:
    return make_pipeline(
        PolynomialFeatures(POLY_DEGREE, include_bias=False), StandardScaler(), estimator
    )


def fit_action_xt(actions: pd.DataFrame) -> ActionXTModel:
    """Fit P(shot) and E[xG] on train matches and report held-out metrics."""
    train, test = split_by_match(actions)
    features = build_features(actions)[list(FEATURES)]

    shot_model = _poly_pipeline(LogisticRegression(max_iter=1000))
    shot_model.fit(features.loc[train], actions.loc[train, "shot_end"])
    prob_test = shot_model.predict_proba(features.loc[test])[:, 1]
    y_test = actions.loc[test, "shot_end"].to_numpy()
    base_rate = float(actions.loc[train, "shot_end"].mean())

    shot_train = actions.loc[train][actions.loc[train, "shot_end"] == 1]
    xg_model = _poly_pipeline(LinearRegression())
    xg_model.fit(features.loc[shot_train.index], shot_train["chain_xg"])

    metrics = {
        "brier": float(brier_score_loss(y_test, prob_test)),
        "brier_baseline": float(brier_score_loss(y_test, np.full(len(y_test), base_rate))),
        "auc": float(roc_auc_score(y_test, prob_test)),
        "base_rate": base_rate,
        "n_train": float(len(train)),
        "n_test": float(len(test)),
    }
    return ActionXTModel(shot_model, xg_model, metrics)


def score_actions(actions: pd.DataFrame, model: ActionXTModel) -> pd.DataFrame:
    """Add `shot_prob`, `xg_pred` and `axt = shot_prob * xg_pred` (xG clipped at 0)."""
    features = build_features(actions)[list(FEATURES)]
    out = actions.copy()
    out["shot_prob"] = model.shot_model.predict_proba(features)[:, 1]
    out["xg_pred"] = np.clip(model.xg_model.predict(features), 0.0, None)
    out["axt"] = out["shot_prob"] * out["xg_pred"]
    return out


def player_action_xt(scored: pd.DataFrame) -> pd.DataFrame:
    """Season action-based xT per (team, player), successful actions only, split by type."""
    df = scored[scored["success"].astype(bool)].copy()
    df["axt_pass"] = df["axt"].where(df["type"] == "Pass", 0.0)
    df["axt_carry"] = df["axt"].where(df["type"] == "Carry", 0.0)
    return df.groupby(["team_id", "player_id"], as_index=False)[
        ["axt", "axt_pass", "axt_carry"]
    ].sum()


def build_player_action_xt(processed_dir: Path) -> tuple[pd.DataFrame, ActionXTModel]:
    """Player table with minutes and per-90 action-based xT, plus the fitted model."""
    events = query(processed_dir, ACTION_EVENTS_SQL)
    actions = label_actions(events)
    model = fit_action_xt(actions)
    table = load_player_minutes(processed_dir).merge(
        player_action_xt(score_actions(actions, model)), on=["team_id", "player_id"], how="left"
    )
    for col in ("axt", "axt_pass", "axt_carry"):
        table[col] = table[col].fillna(0.0)
        table[f"{col}_p90"] = table[col] / table["minutes"].where(table["minutes"] > 0) * 90.0
    return table, model
