"""Shared data structures used across the bot."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Literal


@dataclass(frozen=True)
class ConversationMessage:
    role: Literal["user", "assistant"]
    content: str


@dataclass(frozen=True)
class ConversationExchange:
    user: str
    assistant: str


@dataclass(frozen=True)
class UnresolvedClarification:
    original_query: str
    reformulated_query: str
    question: str


@dataclass(frozen=True)
class PendingMessage:
    message_id: int
    text: str


@dataclass(frozen=True)
class PendingBatch:
    batch_id: str
    user_id: int
    chat_id: int
    messages: tuple[PendingMessage, ...]
    profile: ModelProfileSnapshot | None = None

    @property
    def text(self) -> str:
        return "\n\n".join(message.text for message in self.messages)


@dataclass
class UserConversationState:
    history: deque[ConversationExchange]
    pending: list[PendingMessage] = field(default_factory=list)
    last_active: float = 0.0
    generation: int = 0
    debounce_task: asyncio.Task[None] | None = None
    state_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    processing_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    processing: bool = False
    completed_batches: set[str] = field(default_factory=set)
    unresolved: UnresolvedClarification | None = None


@dataclass(frozen=True)
class ParsedSection:
    path: tuple[str, ...]
    text: str


@dataclass(frozen=True)
class ParsedDocument:
    title: str
    category: str
    sub_category: str
    source: str
    sections: tuple[ParsedSection, ...]


@dataclass(frozen=True)
class KnowledgeChunk:
    chunk_id: int
    text: str
    embedding_text: str
    category: str
    sub_category: str
    source: str
    document_title: str
    section_path: tuple[str, ...]


@dataclass(frozen=True)
class RetrievalHit:
    chunk: KnowledgeChunk
    dense_score: float


@dataclass(frozen=True)
class ReformulationResult:
    query: str
    clarification: str
    answer: str | None = None


@dataclass(frozen=True)
class CompositionResult:
    kind: Literal["answer", "clarify"]
    text: str


@dataclass(frozen=True)
class ModelProfileSnapshot:
    name: str
    reformulate_model: str
    compose_model: str
    attempts: int
    backup_profile: str | None = None


@dataclass(frozen=True)
class PipelineReply:
    text: str
    kind: Literal["answer", "clarify", "insufficient"]
    unresolved: UnresolvedClarification | None = None


BatchProcessor = Callable[
    [
        PendingBatch,
        tuple[ConversationExchange, ...],
        UnresolvedClarification | None,
    ],
    Awaitable[PipelineReply],
]
MessageDeliverer = Callable[[str], Awaitable[None]]
