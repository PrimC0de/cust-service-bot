"""Conversation formatting for recovery and composition prompts."""

from __future__ import annotations

from app.models import ConversationExchange


def format_history(history: tuple[ConversationExchange, ...]) -> str:
    if not history:
        return "(no previous conversation)"
    return "\n\n".join(
        f"User: {exchange.user}\nAssistant: {exchange.assistant}"
        for exchange in history
    )
