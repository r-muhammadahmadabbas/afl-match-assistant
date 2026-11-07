"""Rebuild the processed AFL datasets from the raw CSV exports.

Usage:
    python scripts/prepare_data.py

The raw files are expected under ``data/raw``. Processed artifacts are written
to ``data/processed`` and are what the application loads at runtime.

Design notes
------------
Every predictive feature is computed from information available *before* the
match being described:

* rolling form uses ``shift(1)`` so the current match never enters its own
  feature window;
* ``win_streak`` is the streak *entering* the match, not after it;
* head-to-head history is accumulated in chronological order across the whole
  league, so a team's row never sees a later meeting between the same pair.

This is the difference between a 60-70% sports model and a suspicious 100%
model that has simply leaked the result into a feature.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

ROUND_STATS_CANDIDATES = [
    RAW / "afl_players_round_by_round_stats_raw.csv",
]
MATCHES_CANDIDATES = [
    RAW / "team_matches_home_away_raw.csv",
]

TEAM_FIXES = {
    "w. bulldogs": "Western Bulldogs",
    "western bulldogs": "Western Bulldogs",
}

PLAYER_KEEP = [
    "player_id",
    "team",
    "year",
    "round",
    "career_game_count",
    "fantasy_points",
    "score",
    "margin",
    "disposals",
    "goals",
    "kicks",
    "marks",
    "handballs",
    "tackles",
    "brownlow_votes",
]


def _find(candidates: list[Path], pattern: str, label: str) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = sorted(RAW.glob(pattern))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"Missing {label}. Put the raw CSV in data/raw/ (see data/README.md)."
    )


def _clean_team(name: object) -> str:
    text = re.sub(r"\s+", " ", str(name)).strip()
    return TEAM_FIXES.get(text.lower(), text)


FINALS_ROUNDS = {"EF": 24, "QF": 24, "SF": 25, "PF": 26, "GF": 27}


def _round_number(value: object) -> int:
    """Turn ``R5`` / ``QF`` style round labels into an ordinal number."""
    text = str(value).upper().strip()
    match = re.search(r"\d+", text)
    if match:
        return int(match.group())
    return FINALS_ROUNDS.get(text, 0)


def _rolling_prior(group: pd.Series, window: int) -> pd.Series:
    """Prior-match rolling mean (shifted to avoid leaking the current match)."""
    return group.shift(1).rolling(window, min_periods=1).mean()


def build_player_games(min_year: int = 2012) -> Path:
    """Build a lean per-player, per-game table with prior-form features."""
    path = _find(
        ROUND_STATS_CANDIDATES,
        "*round_by_round*.csv",
        "round-by-round player stats",
    )
    df = pd.read_csv(path)

    required = {"player_id", "team", "year", "round", "disposals", "goals", "fantasy_points"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Raw round stats missing columns: {sorted(missing)}")

    df["team"] = df["team"].map(_clean_team)
    df["match_date"] = pd.to_datetime(df.get("match_date"), errors="coerce")
    df["round_number"] = df["round"].map(_round_number)
    df = df.sort_values(["player_id", "year", "round_number"]).reset_index(drop=True)

    df["disposals_target"] = df["disposals"]
    df["goals_target"] = df["goals"]
    df["fantasy_target"] = df["fantasy_points"]

    grouped = df.groupby("player_id", sort=False)
    df["avg_disposals_5"] = grouped["disposals"].transform(lambda s: _rolling_prior(s, 5))
    df["avg_goals_5"] = grouped["goals"].transform(lambda s: _rolling_prior(s, 5))
    df["avg_fantasy_5"] = grouped["fantasy_points"].transform(
        lambda s: _rolling_prior(s, 5)
    )
    df["avg_margin_5"] = grouped["margin"].transform(lambda s: _rolling_prior(s, 5))

    keep = [c for c in PLAYER_KEEP if c in df.columns]
    keep += [
        "round_number",
        "disposals_target",
        "goals_target",
        "fantasy_target",
        "avg_disposals_5",
        "avg_goals_5",
        "avg_fantasy_5",
        "avg_margin_5",
    ]
    out_df = df[keep + ["match_date"]].copy()
    out_df = out_df[out_df["year"] >= min_year]
    out_df = out_df.dropna(subset=["player_id", "year"])
    out_df = out_df.sort_values(["year", "round_number", "player_id"])

    out = OUT / "player_game_eda.csv"
    out_df.to_csv(out, index=False)
    return out


def build_match_features() -> dict:
    path = _find(
        MATCHES_CANDIDATES,
        "*team_matches*.csv",
        "team home/away match results",
    )
    df = pd.read_csv(path)

    df["team_name"] = df["team_name"].map(_clean_team)
    df["opponent"] = df["opponent"].map(_clean_team)
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df = df.dropna(subset=["match_date"]).copy()

    # A draw is not a win.
    df["home_win"] = (df["result"] == "W").astype(int)
    df["margin"] = pd.to_numeric(df["margin"], errors="coerce").fillna(0)

    # Chronological order is mandatory for leak-free history features.
    df = df.sort_values(["match_date", "id"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Per-team rolling form (previous matches only)
    # ------------------------------------------------------------------
    team_sorted = df.sort_values(["team_name", "match_date"]).copy()
    grouped = team_sorted.groupby("team_name", sort=False)
    for window in (3, 5):
        team_sorted[f"avg_score_{window}"] = grouped["team_score"].transform(
            lambda s, w=window: _rolling_prior(s, w)
        )
        team_sorted[f"avg_conceded_{window}"] = grouped["opponent_score"].transform(
            lambda s, w=window: _rolling_prior(s, w)
        )
        team_sorted[f"avg_margin_{window}"] = grouped["margin"].transform(
            lambda s, w=window: _rolling_prior(s, w)
        )
    team_sorted["form_5"] = grouped["home_win"].transform(
        lambda s: _rolling_prior(s, 5)
    )
    team_sorted["days_rest"] = (
        grouped["match_date"].diff().dt.days.fillna(7).clip(upper=60)
    )
    team_sorted["venue_experience"] = team_sorted.groupby(["team_name", "venue"]).cumcount()

    engineered = [
        "avg_score_3",
        "avg_conceded_3",
        "avg_margin_3",
        "avg_score_5",
        "avg_conceded_5",
        "avg_margin_5",
        "form_5",
        "days_rest",
        "venue_experience",
    ]
    for column in engineered:
        df[column] = team_sorted[column].reindex(df.index)

    # ------------------------------------------------------------------
    # Streak entering the match + head-to-head history in true date order
    # ------------------------------------------------------------------
    streaks: list[int] = []
    current_streak: dict[str, int] = {}

    h2h_win_rate: list[float] = []
    h2h_margin: list[float] = []
    h2h_meetings: list[int] = []
    history: dict[tuple[str, str], list[int]] = {}
    margin_history: dict[tuple[str, str], list[float]] = {}

    for _, row in df.iterrows():
        team, opponent = row["team_name"], row["opponent"]
        key = tuple(sorted((team, opponent)))

        streaks.append(current_streak.get(team, 0))

        prior = history.get(key, [])
        h2h_meetings.append(len(prior))
        h2h_win_rate.append(float(np.mean(prior)) if prior else 0.5)
        h2h_margin.append(
            float(np.mean(margin_history.get(key, [0.0]))) if prior else 0.0
        )

        won = int(row["home_win"])
        current_streak[team] = current_streak.get(team, 0) + 1 if won else 0
        history.setdefault(key, []).append(won)
        margin_history.setdefault(key, []).append(
            float(row["margin"]) if won else -float(row["margin"])
        )

    df["win_streak"] = streaks
    df["h2h_win_rate"] = h2h_win_rate
    df["h2h_avg_margin"] = h2h_margin
    df["h2h_meetings"] = h2h_meetings

    df["is_home"] = (df["home_away"] == "H").astype(int)
    df["season_stage"] = df["round"].map(_round_number).astype(int)

    home = df[df["home_away"] == "H"].copy()
    home.to_csv(OUT / "home_matches_eda.csv", index=False)
    df.to_csv(OUT / "all_matches.csv", index=False)

    feature_cols = [
        "team_name",
        "opponent",
        "match_date",
        "year",
        "round",
        "home_win",
        "margin",
        "win_streak",
        "avg_score_3",
        "avg_conceded_3",
        "avg_margin_3",
        "avg_score_5",
        "avg_conceded_5",
        "form_5",
        "h2h_win_rate",
        "h2h_avg_margin",
        "h2h_meetings",
        "days_rest",
        "is_home",
        "venue_experience",
        "season_stage",
        "team_score",
        "opponent_score",
        "venue",
    ]
    features = df[feature_cols].fillna(0).sort_values(["match_date", "team_name"])

    # Chronological split. The most recent season is the test set, the season
    # before it is validation, and everything older is training data.
    years = sorted(features["year"].dropna().unique())
    if len(years) < 3:
        raise ValueError("Need at least three seasons to build train/val/test splits")
    val_year, test_year = int(years[-2]), int(years[-1])

    features[features["year"] < val_year].to_csv(OUT / "train.csv", index=False)
    features[features["year"] == val_year].to_csv(OUT / "val.csv", index=False)
    features[features["year"] == test_year].to_csv(OUT / "test.csv", index=False)
    features.to_csv(OUT / "feature_table.csv", index=False)

    return {
        "rows": int(len(features)),
        "train_rows": int((features["year"] < val_year).sum()),
        "val_rows": int((features["year"] == val_year).sum()),
        "test_rows": int((features["year"] == test_year).sum()),
        "val_year": val_year,
        "test_year": test_year,
    }


def main() -> None:
    summary = {"player_games": str(build_player_games())}
    summary.update(build_match_features())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
