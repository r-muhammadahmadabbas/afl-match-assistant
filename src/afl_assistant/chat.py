"""Domain-locked AFL chat agent."""
from __future__ import annotations

import re
from typing import Dict, Optional

from .data import resolve_teams_in_text
from .guardrails import is_goodbye, is_greeting, is_off_topic, refusal_response
from .memory import ConversationMemory
from .predict import predict_match_winner
from .retrieval import get_head_to_head, get_player_stats, get_team_record


class AFLChatAgent:
    """Deterministic, retrieval-grounded assistant restricted to AFL topics."""

    def __init__(self, memory: Optional[ConversationMemory] = None) -> None:
        self.memory = memory or ConversationMemory()

    def reset(self) -> None:
        self.memory.clear()

    def chat(self, message: str) -> Dict[str, str]:
        self.memory.add("user", message)
        response, intent = self._respond(message)
        self.memory.add("assistant", response)
        return {"response": response, "intent": intent}

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------
    def _respond(self, message: str) -> tuple[str, str]:
        if is_off_topic(message):
            return refusal_response(), "off_topic"
        if is_greeting(message):
            return (
                "Assalam-o-Alaikum! Main AFL assistant hoon. Team record, player "
                "stats, head-to-head ya match prediction ke liye pooch sakte hain.",
                "greeting",
            )
        if is_goodbye(message):
            return "Khuda Hafiz! AFL ke baare mein kabhi bhi pooch lein.", "goodbye"

        lowered = message.lower()
        if any(token in lowered for token in ("predict", "who will win", "forecast", "odds")):
            return self._prediction(message), "prediction"
        if any(token in lowered for token in ("head to head", "h2h", "matchup", "record against")):
            return self._head_to_head(message), "head_to_head"
        if any(token in lowered for token in ("player", "disposals", "goals", "fantasy", "stats")):
            return self._player(message), "player_stats"
        if any(token in lowered for token in ("record", "win", "loss", "season", "ladder")):
            return self._team_record(message), "team_record"
        return (
            "Main AFL expert hoon. Aap kisi bhi team, player, ya match ke baare "
            "mein pooch sakte hain. Jaise: 'Collingwood ka record kya hai?'",
            "general",
        )

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------
    def _team_record(self, message: str) -> str:
        teams = resolve_teams_in_text(message)
        if not teams:
            return "Team ka naam batayein. Jaise: 'Collingwood ka record kya hai?'"
        record = get_team_record(teams[0])
        if "error" in record:
            return record["error"]
        return (
            f"{record['team']}: {record['wins']} wins, {record['losses']} losses "
            f"({record['win_rate']:.0%} win rate across {record['games']} matches)."
        )

    def _head_to_head(self, message: str) -> str:
        teams = resolve_teams_in_text(message)
        if len(teams) < 2:
            return "Head-to-head ke liye do teams ke naam batayein."
        h2h = get_head_to_head(teams[0], teams[1])
        return (
            f"{h2h['team_a']} vs {h2h['team_b']}: "
            f"{h2h['team_a_wins']}-{h2h['team_b_wins']} in {h2h['total_meetings']} meetings."
        )

    def _player(self, message: str) -> str:
        name = self._extract_player_name(message)
        if not name:
            return "Player ka naam batayein, phir main us ke stats dikha doonga."
        stats = get_player_stats(name)
        if "error" in stats:
            return stats["error"]
        return (
            f"{stats['player']}: {stats['games']} games, "
            f"{stats['avg_disposals']} disposals, {stats['avg_goals']} goals, "
            f"{stats['avg_fantasy']} fantasy points per game."
        )

    def _prediction(self, message: str) -> str:
        teams = resolve_teams_in_text(message)
        if len(teams) < 2:
            return "Prediction ke liye do teams ke naam batayein. Jaise: 'Collingwood vs Carlton'."
        result = predict_match_winner(teams[0], teams[1])
        return (
            f"Prediction: {result['winner']} has a {result['probability']:.0%} win "
            f"probability (confidence: {result['confidence']}). "
            "This is based on historical data, not a certainty."
        )

    @staticmethod
    def _extract_player_name(message: str) -> str:
        # Prefer a quoted name, otherwise fall back to the words between
        # "player"/"stats" markers.
        quoted = re.search(r"['\"]([^'\"]{2,})['\"]", message)
        if quoted:
            return quoted.group(1)
        match = re.search(
            r"(?:player|stats of|stats for)\s+([A-Z][\w'.-]+(?:\s+[A-Z][\w'.-]+)*)",
            message,
        )
        return match.group(1) if match else ""
