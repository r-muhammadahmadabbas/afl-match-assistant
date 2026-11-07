"""Conversation memory with bounded history."""
from __future__ import annotations

from typing import Dict, List


class ConversationMemory:
    """Keeps a bounded, ordered transcript for multi-turn conversations."""

    def __init__(self, max_turns: int = 12, context_turns: int = 5) -> None:
        self.max_turns = max_turns
        self.context_turns = context_turns
        self.history: List[Dict[str, str]] = []

    def add(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.max_turns:
            self.history = self.history[-self.max_turns :]

    def context(self) -> str:
        recent = self.history[-self.context_turns :]
        return "\n".join(f"{m['role']}: {m['content']}" for m in recent)

    def last_user_message(self) -> str:
        for item in reversed(self.history):
            if item["role"] == "user":
                return item["content"]
        return ""

    def clear(self) -> None:
        self.history.clear()
