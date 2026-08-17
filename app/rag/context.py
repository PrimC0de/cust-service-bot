"""Conversation-aware retrieval query construction."""

from __future__ import annotations

from app.models import ConversationExchange


def build_initial_query(current: str, previous: ConversationExchange | None) -> str:
    if previous is None:
        return current
    return (
        f"Current user message:\n{current}\n\n"
        f"Immediately preceding exchange (context only):\n"
        f"User: {previous.user}\nAssistant: {previous.assistant}"
    )


def format_history(history: tuple[ConversationExchange, ...]) -> str:
    if not history:
        return "(no previous conversation)"
    return "\n\n".join(
        f"User: {exchange.user}\nAssistant: {exchange.assistant}"
        for exchange in history
    )
