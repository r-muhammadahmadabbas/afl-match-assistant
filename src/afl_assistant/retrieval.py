"""Structured retrieval helpers used by the chat agent."""
from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from .data import load_home_matches, load_player_game, load_player_info, normalize_team_name


def get_team_record(team_name: str, year: Optional[int] = None) -> Dict:
    """Win/loss record for a team (optionally for a single season)."""
    team = normalize_team_name(team_name)
    matches = load_home_matches()
    subset = matches[matches["team_name"] == team]
    if year is not None:
        subset = subset[subset["year"] == year]

    if subset.empty:
        return {"error": f"No matches found for {team}"}

    wins = int((subset["home_win"] == 1).sum())
    total = int(len(subset))
    return {
        "team": team,
        "year": year,
        "wins": wins,
        "losses": total - wins,
        "games": total,
        "win_rate": round(wins / total, 3),
    }


def get_head_to_head(team_a: str, team_b: str) -> Dict:
    """Historical head-to-head record between two teams."""
    a = normalize_team_name(team_a)
    b = normalize_team_name(team_b)
    matches = load_home_matches()
    h2h = matches[
        ((matches["team_name"] == a) & (matches["opponent"] == b))
        | ((matches["team_name"] == b) & (matches["opponent"] == a))
    ]
    a_wins = int(((h2h["team_name"] == a) & (h2h["home_win"] == 1)).sum())
    b_wins = int(((h2h["team_name"] == b) & (h2h["home_win"] == 1)).sum())
    return {
        "team_a": a,
        "team_b": b,
        "team_a_wins": a_wins,
        "team_b_wins": b_wins,
        "total_meetings": int(len(h2h)),
        "team_a_win_rate": round(a_wins / len(h2h), 3) if len(h2h) else 0.0,
    }


def get_player_stats(player_name: str) -> Dict:
    """Return career averages for a player matched by partial name."""
    players = load_player_info()
    if players.empty:
        return {"error": "Player database is unavailable"}

    mask = players["player_name"].str.contains(player_name, case=False, na=False)
    matches = players[mask]
    if matches.empty:
        return {"error": f"Player not found: {player_name}"}

    player = matches.iloc[0]
    stats = load_player_game()
    player_stats = stats[stats["player_id"] == player["id"]]
    if player_stats.empty:
        return {"error": f"No match stats for {player['player_name']}"}

    return {
        "player": player["player_name"],
        "games": int(len(player_stats)),
        "avg_disposals": round(float(player_stats["disposals_target"].mean()), 1),
        "avg_goals": round(float(player_stats["goals_target"].mean()), 1),
        "avg_fantasy": round(float(player_stats["fantasy_target"].mean()), 1),
    }


def search_players(term: str, limit: int = 5) -> List[Dict]:
    """Lightweight name search used for clarification prompts."""
    players = load_player_info()
    if players.empty:
        return []
    mask = players["player_name"].str.contains(term, case=False, na=False)
    rows = players[mask].head(limit)
    return [
        {"id": int(r["id"]), "player": r["player_name"]}
        for _, r in rows.iterrows()
    ]
