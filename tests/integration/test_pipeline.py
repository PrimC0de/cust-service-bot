import unittest

from app.models import (
    CompositionResult,
    ConversationExchange,
    KnowledgeChunk,
    ModelProfileSnapshot,
    ReformulationResult,
    RetrievalHit,
    UnresolvedClarification,
)
from app.providers.router import ProviderFailure
from app.rag.generation import GenerationService
from app.rag.pipeline import RAGPipeline


PRIMARY = ModelProfileSnapshot("primary", "rewrite", "compose", 2, "backup")
BACKUP = ModelProfileSnapshot("backup", "rewrite", "compose", 3)
CHUNK = KnowledgeChunk(0, "KB fact", "KB fact", "cat", "sub", "source", "Doc", ("Section",))
STRONG = (RetrievalHit(CHUNK, 0.8),)
WEAK = (RetrievalHit(CHUNK, 0.4),)


class FakeRetriever:
    confidence_threshold = 0.7

    def __init__(self, results):
        self.results = list(results)
        self.queries = []

    async def search(self, query, _k, *, batch_id):
        self.queries.append(query)
        return self.results.pop(0)


class FakeGeneration:
    def __init__(
        self,
        composition=None,
        fail_compose=None,
        fail_reformulate=None,
        reformulation=None,
    ):
        self.composition = list(composition or [CompositionResult("answer", "grounded")])
        self.fail_compose = fail_compose
        self.fail_reformulate = fail_reformulate
        self.reformulation = reformulation or ReformulationResult(
            "rewritten query", "What do you mean?"
        )
        self.calls = []

    async def reformulate(self, profile, current, query, history, *, batch_id):
        self.calls.append(("reformulate", profile.name))
        if profile.name == self.fail_reformulate:
            raise ProviderFailure("failed")
        return self.reformulation

    async def compose(self, profile, current, history, hits, *, batch_id):
        self.calls.append(("compose", profile.name))
        if profile.name == self.fail_compose:
            raise ProviderFailure("failed")
        return self.composition.pop(0)


class FakeRouter:
    def __init__(self, response):
        self.response = response
        self.messages = None

    async def json(self, _profile, _model, messages, **_kwargs):
        self.messages = messages
        return self.response


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    def pipeline(self, retriever, generation):
        return RAGPipeline(retriever, generation, {"primary": PRIMARY, "backup": BACKUP})

    async def test_address_only_message_skips_retrieval(self):
        retriever = FakeRetriever([])
        generation = FakeGeneration()
        reply = await self.pipeline(retriever, generation).run(
            "Halo, Bang!", (), PRIMARY, batch_id="b-address"
        )
        self.assertEqual(reply.text, "Siap, bang 👋 Ada yang bisa dibantu?")
        self.assertEqual(retriever.queries, [])
        self.assertEqual(generation.calls, [])

    async def test_address_substring_is_not_treated_as_an_address(self):
        retriever = FakeRetriever([STRONG])
        generation = FakeGeneration()
        reply = await self.pipeline(retriever, generation).run(
            "Bosan nih", (), PRIMARY, batch_id="b-not-address"
        )
        self.assertEqual(reply.text, "grounded")
        self.assertEqual(retriever.queries, ["Bosan nih"])
        self.assertEqual(generation.calls, [("compose", "primary")])

    async def test_strong_current_message_composes_without_history_in_retrieval(self):
        retriever = FakeRetriever([STRONG])
        generation = FakeGeneration()
        reply = await self.pipeline(retriever, generation).run(
            "How do I deposit USDT?",
            (ConversationExchange("old", "answer"),),
            PRIMARY,
            batch_id="b1",
        )
        self.assertEqual(reply.kind, "answer")
        self.assertEqual(retriever.queries, ["How do I deposit USDT?"])
        self.assertEqual(generation.calls, [("compose", "primary")])

    async def test_weak_retrieval_reformulates_once_then_composes(self):
        retriever = FakeRetriever([WEAK, STRONG])
        generation = FakeGeneration()
        reply = await self.pipeline(retriever, generation).run(
            "two days", (), PRIMARY, batch_id="b2"
        )
        self.assertEqual(reply.kind, "answer")
        self.assertEqual(retriever.queries, ["two days", "rewritten query"])
        self.assertEqual(generation.calls, [("reformulate", "primary"), ("compose", "primary")])

    async def test_weak_retrieval_can_answer_from_explicit_user_history(self):
        retriever = FakeRetriever([WEAK])
        generation = FakeGeneration(
            reformulation=ReformulationResult("", "", "Your name is Alan.")
        )
        reply = await self.pipeline(retriever, generation).run(
            "What is my name?",
            (ConversationExchange("My name is Alan.", "Nice to meet you."),),
            PRIMARY,
            batch_id="b-context",
        )
        self.assertEqual(reply.text, "Your name is Alan.")
        self.assertEqual(reply.kind, "answer")
        self.assertEqual(retriever.queries, ["What is my name?"])
        self.assertEqual(generation.calls, [("reformulate", "primary")])

    async def test_second_weak_result_asks_one_clarification(self):
        reply = await self.pipeline(FakeRetriever([WEAK, WEAK]), FakeGeneration()).run(
            "two days", (), PRIMARY, batch_id="b3"
        )
        self.assertEqual(reply.kind, "clarify")
        self.assertEqual(reply.unresolved.original_query, "two days")

    async def test_pending_clarification_searches_current_then_combined(self):
        pending = UnresolvedClarification("two days", "withdrawal duration", "For what?")
        retriever = FakeRetriever([WEAK, STRONG])
        generation = FakeGeneration()
        reply = await self.pipeline(retriever, generation).run(
            "withdrawal", (), PRIMARY, batch_id="b4", unresolved=pending
        )
        self.assertEqual(reply.kind, "answer")
        self.assertEqual(retriever.queries[0], "withdrawal")
        self.assertEqual(retriever.queries[1], "rewritten query")
        self.assertIn(("reformulate", "primary"), generation.calls)

    async def test_pending_clarification_stops_after_final_weak_result(self):
        pending = UnresolvedClarification("two days", "withdrawal duration", "For what?")
        reply = await self.pipeline(FakeRetriever([WEAK, WEAK]), FakeGeneration()).run(
            "unclear", (), PRIMARY, batch_id="b5", unresolved=pending
        )
        self.assertEqual(reply.kind, "insufficient")
        self.assertIsNone(reply.unresolved)

    async def test_strong_new_topic_clears_pending_clarification(self):
        pending = UnresolvedClarification("two days", "withdrawal duration", "For what?")
        retriever = FakeRetriever([STRONG])
        reply = await self.pipeline(retriever, FakeGeneration()).run(
            "How do I deposit USDT?", (), PRIMARY, batch_id="b6", unresolved=pending
        )
        self.assertEqual(reply.kind, "answer")
        self.assertIsNone(reply.unresolved)
        self.assertEqual(len(retriever.queries), 1)

    async def test_conflicting_strong_chunks_use_composer_clarification(self):
        generation = FakeGeneration([CompositionResult("clarify", "Which region?")])
        reply = await self.pipeline(FakeRetriever([STRONG]), generation).run(
            "What is the limit?", (), PRIMARY, batch_id="b7"
        )
        self.assertEqual(reply.kind, "clarify")
        self.assertIsNotNone(reply.unresolved)

    async def test_composition_failover_reuses_retrieval(self):
        retriever = FakeRetriever([STRONG])
        generation = FakeGeneration(fail_compose="primary")
        reply = await self.pipeline(retriever, generation).run(
            "question", (), PRIMARY, batch_id="b8"
        )
        self.assertEqual(reply.kind, "answer")
        self.assertEqual(len(retriever.queries), 1)
        self.assertEqual(generation.calls, [("compose", "primary"), ("compose", "backup")])

    async def test_reformulation_failover_retries_only_reformulation(self):
        generation = FakeGeneration(fail_reformulate="primary")
        reply = await self.pipeline(FakeRetriever([WEAK, STRONG]), generation).run(
            "fragment", (), PRIMARY, batch_id="b9"
        )
        self.assertEqual(reply.kind, "answer")
        self.assertEqual(
            generation.calls,
            [("reformulate", "primary"), ("reformulate", "backup"), ("compose", "primary")],
        )


class GenerationTests(unittest.IsolatedAsyncioTestCase):
    async def test_reformulation_accepts_answer_from_explicit_user_history(self):
        router = FakeRouter(
            {"answer": "Your name is Alan.", "query": "", "clarification": ""}
        )
        result = await GenerationService(router).reformulate(
            PRIMARY,
            "What is my name?",
            "What is my name?",
            (ConversationExchange("My name is Alan.", "Nice to meet you."),),
            batch_id="b-generation-context",
        )
        self.assertEqual(result.answer, "Your name is Alan.")
        self.assertIn("User: My name is Alan.", router.messages[1]["content"])

    async def test_composer_receives_separate_conversation_and_kb_evidence(self):
        router = FakeRouter({"kind": "answer", "text": "Your name is Alan."})
        result = await GenerationService(router).compose(
            PRIMARY,
            "What is my name?",
            (ConversationExchange("My name is Alan.", "Nice to meet you."),),
            STRONG,
            batch_id="b-generation-compose",
        )
        self.assertEqual(result.text, "Your name is Alan.")
        self.assertIn("Facts explicitly stated by the user", router.messages[0]["content"])
        self.assertIn("Knowledge chunks:", router.messages[1]["content"])

    async def test_composer_prompt_mirrors_only_clear_address_terms(self):
        router = FakeRouter({"kind": "answer", "text": "Siap, bang."})
        await GenerationService(router).compose(
            PRIMARY, "Bang, bantu dong", (), STRONG, batch_id="b-tone"
        )
        prompt = router.messages[0]["content"]
        self.assertIn("natural Gen-Z conversational style", prompt)
        self.assertIn("imperfect grammar are allowed", prompt)
        self.assertIn("Do not sound formal", prompt)
        self.assertIn('"bosan" does not mean', prompt)
        self.assertNotIn("Always address the customer", prompt)
