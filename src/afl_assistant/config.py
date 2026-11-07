"""Configuration and team metadata for the AFL assistant."""
from __future__ import annotations

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
DOCS_DIR = PROJECT_ROOT / "docs"
LOG_DIR = Path(os.getenv("LOG_DIR", PROJECT_ROOT / "logs"))

MATCH_MODEL_PATH = Path(
    os.getenv("MATCH_MODEL_PATH", MODELS_DIR / "match_winner_model.joblib")
)
PLAYER_MODEL_PATH = Path(
    os.getenv("PLAYER_MODEL_PATH", MODELS_DIR / "top_player_model.joblib")
)
PLAYER_INFO_PATH = Path(
    os.getenv("PLAYER_INFO_PATH", RAW_DIR / "afl_players_info_raw.csv")
)

MATCH_FEATURES = [
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
]

PLAYER_FEATURES = [
    "year",
    "round_number",
    "career_game_count",
    "avg_disposals_5",
    "avg_goals_5",
    "avg_fantasy_5",
    "avg_margin_5",
]

PLAYER_TARGETS = {
    "disposals": "disposals_target",
    "goals": "goals_target",
    "fantasy_points": "fantasy_target",
}

TEAM_ALIASES = {
    "pies": "Collingwood Magpies",
    "magpies": "Collingwood Magpies",
    "blues": "Carlton Blues",
    "hawks": "Hawthorn Hawks",
    "tigers": "Richmond Tigers",
    "bulldogs": "Western Bulldogs",
    "swans": "Sydney Swans",
    "eagles": "West Coast Eagles",
    "dockers": "Fremantle Dockers",
    "crows": "Adelaide Crows",
    "lions": "Brisbane Lions",
    "suns": "Gold Coast Suns",
    "giants": "Greater Western Sydney Giants",
    "kangaroos": "North Melbourne Kangaroos",
    "demons": "Melbourne Demons",
    "saints": "St Kilda Saints",
    "bombers": "Essendon Bombers",
    "power": "Port Adelaide Power",
}

OFF_TOPIC_KEYWORDS = (
    "weather",
    "mausam",
    "politics",
    "siasat",
    "cricket",
    "khana",
    "movie",
    "film",
    "song",
    "stock market",
    "crypto",
)

REFUSAL_RESPONSE = (
    "Main sirf AFL ki madad kar sakta hoon. Kya aap koi AFL team, player, "
    "ya match ke baare mein jaanna chahte hain?"
)
