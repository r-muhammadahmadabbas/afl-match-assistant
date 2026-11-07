"""Guardrails that keep the assistant inside the AFL domain."""
from __future__ import annotations

from .config import OFF_TOPIC_KEYWORDS, REFUSAL_RESPONSE


def is_off_topic(query: str) -> bool:
    lowered = (query or "").lower()
    return any(keyword in lowered for keyword in OFF_TOPIC_KEYWORDS)


def refusal_response() -> str:
    return REFUSAL_RESPONSE


def is_prediction_query(query: str) -> bool:
    lowered = (query or "").lower()
    return any(
        token in lowered
        for token in ("predict", "will win", "who will", "forecast", "odds", "chance")
    )


def is_greeting(query: str) -> bool:
    lowered = (query or "").lower().strip()
    return lowered in {"hi", "hello", "hey", "salam", "assalam-o-alaikum", "assalam o alaikum"}


def is_goodbye(query: str) -> bool:
    lowered = (query or "").lower()
    return any(token in lowered for token in ("bye", "goodbye", "see you", "khuda hafiz"))
