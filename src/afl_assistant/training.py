"""Model training for match-winner and top-player prediction.

Two families of models are produced:

``match_winner_model.joblib``
    A binary classifier that estimates the probability that a team wins its
    next match. Logistic regression and gradient boosting are compared on the
    validation season; the better model is persisted.

``top_player_model.joblib``
    A dictionary of three gradient-boosting regressors, one per player stat
    (disposals, goals, fantasy points). Each model predicts a player's
    *next-game* output from their previous five games, and the predicted value
    is used to rank players.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import (
    MATCH_FEATURES,
    MATCH_MODEL_PATH,
    PLAYER_FEATURES,
    PLAYER_TARGETS,
    PLAYER_MODEL_PATH,
)
from .data import load_player_game, load_test, load_train, load_val, match_feature_frame


def _classification_metrics(y_true, y_pred, y_proba) -> Dict[str, float]:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1": round(float(f1_score(y_true, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 4),
        "brier": round(float(brier_score_loss(y_true, y_proba)), 4),
    }


def train_match_winner() -> Dict:
    """Compare LR and gradient boosting on validation, persist the winner."""
    train, val, test = load_train(), load_val(), load_test()
    X_train, y_train = match_feature_frame(train), train["home_win"]
    X_val, y_val = match_feature_frame(val), val["home_win"]
    X_test, y_test = match_feature_frame(test), test["home_win"]

    models = {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=150,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            random_state=42,
        ),
    }

    results: Dict[str, Dict] = {}
    best_name, best_model, best_acc = None, None, -1.0
    for name, model in models.items():
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        val_proba = model.predict_proba(X_val)[:, 1]
        accuracy = accuracy_score(y_val, val_pred)
        results[name] = _classification_metrics(y_val, val_pred, val_proba)
        if accuracy > best_acc:
            best_name, best_model, best_acc = name, model, accuracy

    test_pred = best_model.predict(X_test)
    test_proba = best_model.predict_proba(X_test)[:, 1]
    results["selected"] = best_name
    results["test"] = _classification_metrics(y_test, test_pred, test_proba)
    results["train_rows"] = int(len(train))
    results["val_rows"] = int(len(val))
    results["test_rows"] = int(len(test))

    MATCH_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MATCH_MODEL_PATH)
    (MATCH_MODEL_PATH.parent / "match_metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    return results


def _regression_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
    }


def train_top_player() -> Dict:
    """Train next-game regressors from a player's prior five games."""
    games = load_player_game().copy()
    games = games.dropna(subset=PLAYER_FEATURES + list(PLAYER_TARGETS.values()))
    if games.empty:
        raise ValueError("Not enough player history to train the player models")

    latest_season = int(games["year"].max())
    train = games[games["year"] < latest_season]
    test = games[games["year"] == latest_season]

    X_train = train[PLAYER_FEATURES]
    X_test = test[PLAYER_FEATURES]

    models: Dict[str, GradientBoostingRegressor] = {}
    metrics: Dict[str, Dict] = {}
    for stat, target in PLAYER_TARGETS.items():
        model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            random_state=42,
        )
        model.fit(X_train, train[target])
        prediction = model.predict(X_test)
        metrics[stat] = _regression_metrics(test[target], prediction)
        models[stat] = model

    metrics["train_rows"] = int(len(train))
    metrics["test_rows"] = int(len(test))
    metrics["test_season"] = latest_season

    PLAYER_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, PLAYER_MODEL_PATH)
    (PLAYER_MODEL_PATH.parent / "top_player_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return metrics


if __name__ == "__main__":
    print("Training match-winner model...")
    print(json.dumps(train_match_winner(), indent=2))
    print("Training top-player models...")
    print(json.dumps(train_top_player(), indent=2))
