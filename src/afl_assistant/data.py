"""Dataset loading and team-name normalization."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd

from .config import (
    MATCH_FEATURES,
    PLAYER_INFO_PATH,
    PROCESSED_DIR,
    TEAM_ALIASES,
)


def _read(name: str) -> pd.DataFrame:
    path = PROCESSED_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"Missing processed dataset: {path}. Run scripts/prepare_data.py or "
            "see data/README.md."
        )
    return pd.read_csv(path)


@lru_cache(maxsize=1)
def load_feature_table() -> pd.DataFrame:
    return _read("feature_table.csv")


@lru_cache(maxsize=1)
def load_train() -> pd.DataFrame:
    return _read("train.csv")


@lru_cache(maxsize=1)
def load_val() -> pd.DataFrame:
    return _read("val.csv")


@lru_cache(maxsize=1)
def load_test() -> pd.DataFrame:
    return _read("test.csv")


@lru_cache(maxsize=1)
def load_home_matches() -> pd.DataFrame:
    df = _read("home_matches_eda.csv")
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    return df.sort_values("match_date")


@lru_cache(maxsize=1)
def load_player_game() -> pd.DataFrame:
    return _read("player_game_eda.csv")


@lru_cache(maxsize=1)
def load_player_info() -> pd.DataFrame:
    if not PLAYER_INFO_PATH.exists():
        return pd.DataFrame(columns=["id", "player_name"])
    df = pd.read_csv(PLAYER_INFO_PATH)
    df["player_name"] = df["player_name"].astype(str).str.strip()
    return df


def match_feature_frame(df: pd.DataFrame, features=None) -> pd.DataFrame:
    """Select and fill the model feature matrix."""
    features = features or MATCH_FEATURES
    return df[features].fillna(0)


def normalize_team_name(name: str) -> str:
    """Resolve nicknames / partial names to the canonical AFL team name."""
    if not name:
        return name

    needle = name.lower().strip()
    if needle in TEAM_ALIASES:
        return TEAM_ALIASES[needle]

    teams = set(TEAM_ALIASES.values())
    for team in teams:
        if needle == team.lower():
            return team

    for team in teams:
        if needle in team.lower() or team.lower() in needle:
            return team

    for alias, team in TEAM_ALIASES.items():
        if alias in needle:
            return team

    return name


def resolve_teams_in_text(text: str) -> list[str]:
    """Return canonical team names mentioned in a free-text query.

    Matches full names ("Collingwood Magpies"), the distinctive part of a name
    ("Collingwood", "Carlton") and nicknames ("pies", "blues").
    """
    lowered = (text or "").lower()
    found: list[str] = []
    for team in set(TEAM_ALIASES.values()):
        if team.lower() in lowered:
            found.append(team)
            continue
        # Match the city / distinctive portion, e.g. "collingwood".
        if any(
            len(word) >= 5 and word in lowered.split()
            for word in team.lower().split()
        ):
            found.append(team)

    for alias, team in TEAM_ALIASES.items():
        if alias in lowered.split() and team not in found:
            found.append(team)
    return found


def latest_season(df: Optional[pd.DataFrame] = None) -> int:
    data = df if df is not None else load_home_matches()
    return int(data["year"].max())
