import asyncio
import unittest

from telegram.error import NetworkError

from app.bot.conversation import ConversationStore
from app.models import ModelProfileSnapshot, PipelineReply, UnresolvedClarification


PROFILE = ModelProfileSnapshot(
    name="test",
    reformulate_model="reformulate",
    compose_model="compose",
    attempts=1,
)


class ConversationStoreTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.store = ConversationStore(
            debounce_seconds=0.02,
            history_exchanges=4,
            idle_ttl_seconds=86_400,
            delivery_attempts=3,
            profile_snapshot=lambda: PROFILE,
        )

    async def asyncTearDown(self):
        await self.store.close()

    async def test_debounce_resets_and_combines_bubbles(self):
        processed = []
        delivered = []

        async def process(batch, history, _unresolved):
            processed.append((batch, history))
            return PipelineReply("answer", "answer")

        async def deliver(text):
            delivered.append(text)

        common = dict(
            user_id=1,
            chat_id=1,
            process=process,
            deliver=deliver,
            unavailable=lambda _: "unavailable",
        )
        await self.store.submit(message_id=1, text="first", **common)
        await asyncio.sleep(0.01)
        await self.store.submit(message_id=2, text="second", **common)
        await asyncio.sleep(0.05)

        self.assertEqual(processed[0][0].text, "first\n\nsecond")
        self.assertEqual(delivered, ["answer"])

    async def test_pipeline_failure_is_not_retried(self):
        ids = []
        delivered = []

        async def process(batch, _history, _unresolved):
            ids.append(batch.batch_id)
            raise RuntimeError("provider down")

        async def deliver(text):
            delivered.append(text)

        await self.store.submit(
            user_id=1,
            chat_id=1,
            message_id=1,
            text="hello",
            process=process,
            deliver=deliver,
            unavailable=lambda _: "unavailable",
        )
        await asyncio.sleep(0.06)
        self.assertEqual(len(ids), 1)
        self.assertEqual(len(set(ids)), 1)
        self.assertEqual(delivered, ["unavailable"])

    async def test_transient_delivery_retries_cached_reply_only(self):
        processed = []
        deliveries = []

        async def process(batch, _history, _unresolved):
            processed.append(batch.batch_id)
            return PipelineReply("answer", "answer")

        async def deliver(text):
            deliveries.append(text)
            if len(deliveries) == 1:
                raise NetworkError("temporary")

        await self.store.submit(
            user_id=1,
            chat_id=1,
            message_id=1,
            text="hello",
            process=process,
            deliver=deliver,
            unavailable=lambda _: "unavailable",
        )
        await asyncio.sleep(0.06)
        self.assertEqual(len(processed), 1)
        self.assertEqual(deliveries, ["answer", "answer"])

    async def test_successful_replies_store_and_clear_clarification(self):
        pending = UnresolvedClarification("original", "rewritten", "Which one?")
        replies = [PipelineReply("Which one?", "clarify", pending), PipelineReply("answer", "answer")]
        seen = []

        async def process(_batch, _history, unresolved):
            seen.append(unresolved)
            return replies.pop(0)

        async def deliver(_text):
            pass

        common = dict(
            user_id=1,
            chat_id=1,
            process=process,
            deliver=deliver,
            unavailable=lambda _: "unavailable",
        )
        await self.store.submit(message_id=1, text="ambiguous", **common)
        await asyncio.sleep(0.04)
        await self.store.submit(message_id=2, text="clarification", **common)
        await asyncio.sleep(0.04)
        self.assertEqual(seen, [None, pending])
        self.assertIsNone(self.store.states[1].unresolved)

    async def test_users_are_isolated_and_same_user_follow_up_is_ordered(self):
        started = asyncio.Event()
        release = asyncio.Event()
        events = []

        async def process(batch, history, _unresolved):
            events.append(("start", batch.user_id, batch.text, len(history)))
            if batch.text == "slow":
                started.set()
                await release.wait()
            events.append(("end", batch.user_id, batch.text, len(history)))
            return PipelineReply(f"reply:{batch.text}", "answer")

        async def deliver(_text):
            pass

        def submit(user, message_id, text):
            return self.store.submit(
                user_id=user,
                chat_id=user,
                message_id=message_id,
                text=text,
                process=process,
                deliver=deliver,
                unavailable=lambda _: "unavailable",
            )

        await submit(1, 1, "slow")
        await started.wait()
        await submit(1, 2, "follow-up")
        await submit(2, 1, "other-user")
        await asyncio.sleep(0.04)
        self.assertIn(("end", 2, "other-user", 0), events)
        self.assertNotIn(("start", 1, "follow-up", 1), events)
        release.set()
        await asyncio.sleep(0.04)
        self.assertIn(("start", 1, "follow-up", 1), events)

    async def test_history_keeps_four_global_exchanges_and_expires(self):
        now = [100.0]
        store = ConversationStore(
            debounce_seconds=0,
            history_exchanges=4,
            idle_ttl_seconds=10,
            profile_snapshot=lambda: PROFILE,
            clock=lambda: now[0],
        )

        async def process(batch, _history, _unresolved):
            return PipelineReply(f"a{batch.message_id if hasattr(batch, 'message_id') else ''}", "answer")

        async def deliver(_text):
            pass

        for number in range(5):
            await store.submit(
                user_id=1,
                chat_id=1,
                message_id=number,
                text=f"topic-{number}",
                process=process,
                deliver=deliver,
                unavailable=lambda _: "unavailable",
            )
            await asyncio.sleep(0.01)
        self.assertEqual(len(store.states[1].history), 4)
        self.assertEqual(store.states[1].history[0].user, "topic-1")
        now[0] += 11
        self.assertEqual(await store.cleanup_expired(), 1)
        await store.close()

    async def test_in_flight_batch_keeps_its_profile_snapshot(self):
        selected = [PROFILE]
        started = asyncio.Event()
        release = asyncio.Event()
        seen = []
        store = ConversationStore(
            debounce_seconds=0.01,
            profile_snapshot=lambda: selected[0],
        )

        async def process(batch, _history, _unresolved):
            seen.append(batch.profile.name)
            started.set()
            await release.wait()
            return PipelineReply("done", "answer")

        async def deliver(_text):
            pass

        await store.submit(
            user_id=1,
            chat_id=1,
            message_id=1,
            text="question",
            process=process,
            deliver=deliver,
            unavailable=lambda _: "unavailable",
        )
        await started.wait()
        selected[0] = ModelProfileSnapshot(
            name="replacement",
            reformulate_model="f",
            compose_model="c",
            attempts=1,
        )
        release.set()
        await asyncio.sleep(0.01)
        self.assertEqual(seen, ["test"])
        await store.close()
