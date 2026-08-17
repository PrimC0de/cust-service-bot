"""In-memory DM batching and conversation history."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import deque
from collections.abc import Callable

from telegram.error import NetworkError, RetryAfter

from app.models import (
    BatchProcessor,
    ConversationExchange,
    MessageDeliverer,
    ModelProfileSnapshot,
    PendingBatch,
    PendingMessage,
    PipelineReply,
    UserConversationState,
)


logger = logging.getLogger(__name__)


class ConversationStore:
    """Owns ephemeral per-user state for a single polling process."""

    def __init__(
        self,
        *,
        debounce_seconds: float = 3.0,
        history_exchanges: int = 4,
        idle_ttl_seconds: float = 86_400,
        cleanup_interval_seconds: float = 3_600,
        delivery_attempts: int = 3,
        profile_snapshot: Callable[[], ModelProfileSnapshot] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.debounce_seconds = debounce_seconds
        self.history_exchanges = history_exchanges
        self.idle_ttl_seconds = idle_ttl_seconds
        self.cleanup_interval_seconds = cleanup_interval_seconds
        self.delivery_attempts = delivery_attempts
        self.profile_snapshot = profile_snapshot
        self.clock = clock
        self.states: dict[int, UserConversationState] = {}
        self._states_lock = asyncio.Lock()
        self._tasks: set[asyncio.Task[None]] = set()

    async def submit(
        self,
        *,
        user_id: int,
        chat_id: int,
        message_id: int,
        text: str,
        process: BatchProcessor,
        deliver: MessageDeliverer,
        unavailable: Callable[[str], str],
    ) -> None:
        state = await self._state_for(user_id)
        async with state.state_lock:
            state.pending.append(PendingMessage(message_id=message_id, text=text))
            state.last_active = self.clock()
            state.generation += 1
            generation = state.generation
            if state.debounce_task and not state.debounce_task.done():
                state.debounce_task.cancel()
            task = asyncio.create_task(
                self._debounce_and_dispatch(
                    state=state,
                    generation=generation,
                    user_id=user_id,
                    chat_id=chat_id,
                    process=process,
                    deliver=deliver,
                    unavailable=unavailable,
                ),
                name=f"debounce-{user_id}-{generation}",
            )
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
            state.debounce_task = task

    async def _state_for(self, user_id: int) -> UserConversationState:
        async with self._states_lock:
            state = self.states.get(user_id)
            now = self.clock()
            if state is None or (
                now - state.last_active > self.idle_ttl_seconds
                and not state.pending
                and not state.processing
            ):
                state = UserConversationState(
                    history=deque(maxlen=self.history_exchanges),
                    last_active=now,
                )
                self.states[user_id] = state
            return state

    async def _debounce_and_dispatch(
        self,
        *,
        state: UserConversationState,
        generation: int,
        user_id: int,
        chat_id: int,
        process: BatchProcessor,
        deliver: MessageDeliverer,
        unavailable: Callable[[str], str],
    ) -> None:
        try:
            await asyncio.sleep(self.debounce_seconds)
            async with state.state_lock:
                if generation != state.generation or not state.pending:
                    return
                messages = tuple(state.pending)
                state.pending.clear()
                state.debounce_task = None
                batch = PendingBatch(
                    batch_id=uuid.uuid4().hex,
                    user_id=user_id,
                    chat_id=chat_id,
                    messages=messages,
                    profile=self.profile_snapshot() if self.profile_snapshot else None,
                )

            async with state.processing_lock:
                state.processing = True
                try:
                    history = tuple(state.history)
                    await self._dispatch(
                        state,
                        batch,
                        history,
                        process,
                        deliver,
                        unavailable,
                    )
                finally:
                    state.processing = False
                    state.last_active = self.clock()
        except asyncio.CancelledError:
            return

    async def _dispatch(
        self,
        state: UserConversationState,
        batch: PendingBatch,
        history: tuple[ConversationExchange, ...],
        process: BatchProcessor,
        deliver: MessageDeliverer,
        unavailable: Callable[[str], str],
    ) -> None:
        try:
            reply = await process(batch, history, state.unresolved)
        except Exception as error:
            logger.warning(
                "batch_id=%s stage=pipeline profile=%s attempt=1 error_type=%s",
                batch.batch_id,
                batch.profile.name if batch.profile else "unset",
                type(error).__name__,
            )
            await self._deliver(batch, unavailable(batch.text), deliver, "failure_delivery")
            return

        if batch.batch_id in state.completed_batches:
            return
        if await self._deliver(batch, reply.text, deliver, "reply_delivery"):
            state.completed_batches.add(batch.batch_id)
            state.history.append(ConversationExchange(user=batch.text, assistant=reply.text))
            state.unresolved = reply.unresolved

    async def _deliver(
        self,
        batch: PendingBatch,
        text: str,
        deliver: MessageDeliverer,
        stage: str,
    ) -> bool:
        for attempt in range(1, self.delivery_attempts + 1):
            try:
                await deliver(text)
                return True
            except Exception as error:
                logger.warning(
                    "batch_id=%s stage=%s profile=%s attempt=%s error_type=%s",
                    batch.batch_id,
                    stage,
                    batch.profile.name if batch.profile else "unset",
                    attempt,
                    type(error).__name__,
                )
                if not isinstance(error, (NetworkError, RetryAfter)):
                    break
        return False

    async def cleanup_expired(self) -> int:
        now = self.clock()
        removed = 0
        async with self._states_lock:
            for user_id, state in list(self.states.items()):
                if (
                    now - state.last_active > self.idle_ttl_seconds
                    and not state.pending
                    and not state.processing
                ):
                    if state.debounce_task and not state.debounce_task.done():
                        state.debounce_task.cancel()
                    del self.states[user_id]
                    removed += 1
        return removed

    async def cleanup_loop(self) -> None:
        while True:
            await asyncio.sleep(self.cleanup_interval_seconds)
            await self.cleanup_expired()

    async def close(self) -> None:
        tasks = [task for task in self._tasks if not task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
