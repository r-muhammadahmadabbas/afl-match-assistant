"""Inference for match winners and top players."""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Optional

import joblib
import pandas as pd

from .config import (
    MATCH_FEATURES,
    MATCH_MODEL_PATH,
    PLAYER_FEATURES,
    PLAYER_MODEL_PATH,
    PLAYER_TARGETS,
)
from .data import (
    load_home_matches,
    load_player_game,
    load_player_info,
    normalize_team_name,
)

STAT_COLUMNS = dict(PLAYER_TARGETS)

# Prior-form column used when a model artifact is unavailable.
FALLBACK_FORM = {
    "disposals": "avg_disposals_5",
    "goals": "avg_goals_5",
    "fantasy_points": "avg_fantasy_5",
}


@lru_cache(maxsize=1)
def _match_model():
    if not MATCH_MODEL_PATH.exists():
        return None
    return joblib.load(MATCH_MODEL_PATH)


@lru_cache(maxsize=1)
def _player_models():
    if not PLAYER_MODEL_PATH.exists():
        return None
    return joblib.load(PLAYER_MODEL_PATH)


def _recent_form(team: str, n: int = 5) -> Dict[str, float]:
    matches = load_home_matches()
    subset = matches[matches["team_name"] == team].sort_values("match_date").tail(n)
    if subset.empty:
        return {
            "win_streak": 0,
            "avg_score": 90.0,
            "avg_conceded": 90.0,
            "avg_margin": 0.0,
            "form": 0.5,
        }

    streak = 0
    for won in reversed(subset["home_win"].tolist()):
        if won == 1:
            streak += 1
        else:
            break

    return {
        "win_streak": streak,
        "avg_score": float(subset["team_score"].mean()),
        "avg_conceded": float(subset["opponent_score"].mean()),
        "avg_margin": float(subset["margin"].mean()),
        "form": float(subset["home_win"].mean()),
    }


def build_match_features(team_a: str, team_b: str) -> Dict[str, float]:
    """Derive a feature row from recent form and head-to-head history."""
    a, b = normalize_team_name(team_a), normalize_team_name(team_b)
    form_a, form_b = _recent_form(a), _recent_form(b)

    matches = load_home_matches()
    h2h = matches[
        ((matches["team_name"] == a) & (matches["opponent"] == b))
        | ((matches["team_name"] == b) & (matches["opponent"] == a))
    ]
    a_wins = int(((h2h["team_name"] == a) & (h2h["home_win"] == 1)).sum())
    meetings = int(len(h2h))
    h2h_rate = a_wins / meetings if meetings else 0.5
    h2h_margin = float(h2h[h2h["team_name"] == a]["margin"].mean()) if meetings else 0.0

    return {
        "win_streak": form_a["win_streak"],
        "avg_score_3": form_a["avg_score"],
        "avg_conceded_3": form_a["avg_conceded"],
        "avg_margin_3": form_a["avg_margin"],
        "avg_score_5": form_a["avg_score"],
        "avg_conceded_5": form_a["avg_conceded"],
        "form_5": (form_a["form"] + form_b["form"]) / 2,
        "h2h_win_rate": h2h_rate,
        "h2h_avg_margin": h2h_margin,
        "h2h_meetings": meetings,
        "days_rest": 7,
        "is_home": 1,
        "venue_experience": min(meetings, 20),
    }


def predict_match_winner(
    team_a: str,
    team_b: str,
    features: Optional[Dict[str, float]] = None,
) -> Dict:
    """Predict the winner of ``team_a`` (home) vs ``team_b`` (away)."""
    a, b = normalize_team_name(team_a), normalize_team_name(team_b)
    feature_row = features or build_match_features(a, b)
    model = _match_model()

    if model is None:
        return {
            "team_a": a,
            "team_b": b,
            "winner": a,
            "probability": 0.57,
            "confidence": "low",
            "features": feature_row,
            "note": "Model artifact missing; using home-ground baseline.",
        }

    matrix = pd.DataFrame([feature_row])[MATCH_FEATURES]
    probability = float(model.predict_proba(matrix)[0, 1])
    confidence = (
        "high" if probability >= 0.65 else "medium" if probability >= 0.55 else "low"
    )

    return {
        "team_a": a,
        "team_b": b,
        "winner": a if probability >= 0.5 else b,
        "probability": round(probability, 3),
        "confidence": confidence,
        "features": feature_row,
    }


def predict_top_player(
    team: Optional[str] = None,
    stat_type: str = "disposals",
    top_k: int = 5,
) -> List[Dict]:
    """Rank players by their model-predicted output for the next game.

    The regressors are trained on a player's previous five games, so every
    prediction is made strictly from pre-match information.
    """
    if stat_type not in PLAYER_TARGETS:
        raise ValueError(f"stat_type must be one of {sorted(PLAYER_TARGETS)}")

    games = load_player_game()
    if team:
        team = normalize_team_name(team)
        games = games[games["team"] == team]

    if games.empty:
        return []

    # Only rank players who appeared in the most recent season in the dataset.
    latest_season = int(games["year"].max())
    games = games[games["year"] == latest_season]

    latest = (
        games.dropna(subset=PLAYER_FEATURES)
        .sort_values(["player_id", "year", "round_number"])
        .groupby("player_id")
        .tail(1)
    )
    if latest.empty:
        return []

    models = _player_models()
    if models and stat_type in models:
        values = models[stat_type].predict(latest[PLAYER_FEATURES])
    else:
        values = latest[FALLBACK_FORM[stat_type]].fillna(0).to_numpy()

    ranked = pd.DataFrame(
        {
            "player_id": latest["player_id"].to_numpy(),
            "team": latest["team"].to_numpy(),
            "predicted_value": values,
        }
    ).nlargest(top_k, "predicted_value")

    info = load_player_info()
    names = (
        info.set_index("id")["player_name"].to_dict() if not info.empty else {}
    )

    return [
        {
            "player_id": int(row.player_id),
            "player": str(names.get(row.player_id, row.player_id)),
            "team": row.team,
            "stat": stat_type,
            "predicted_value": round(float(row.predicted_value), 1),
        }
        for row in ranked.itertuples()
    ]
