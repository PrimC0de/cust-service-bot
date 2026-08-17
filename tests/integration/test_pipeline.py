import unittest

from app.models import (
    ConversationExchange,
    EvidenceDecision,
    HybridRetrievalResult,
    KnowledgeChunk,
    ModelProfileSnapshot,
    RetrievalHit,
)
from app.providers.router import ProviderFailure
from app.rag.pipeline import RAGPipeline


PRIMARY = ModelProfileSnapshot(
    "kimi", "url", "KIMI_API_KEY", "rerank", "rewrite", "compose", 2, "backup"
)
BACKUP = ModelProfileSnapshot(
    "backup", "url", "OPENAI_API_KEY", "rerank", "rewrite", "compose", 3
)
CHUNK = KnowledgeChunk(0, "KB fact", "KB fact", "cat", "sub", "source", "Doc", ("Section",))
RESULT = HybridRetrievalResult((RetrievalHit(CHUNK, bm25_score=1.0),), False)


class FakeRetriever:
    def __init__(self, results=None):
        self.results = list(results or [RESULT])
        self.queries = []

    async def search(self, query, *, batch_id):
        self.queries.append(query)
        return self.results.pop(0) if self.results else RESULT


class FakeGeneration:
    def __init__(self, decisions, fail_profile=None, fail_compose_profile=None):
        self.decisions = list(decisions)
        self.fail_profile = fail_profile
        self.fail_compose_profile = fail_compose_profile
        self.profiles = []

    async def decide(self, profile, query, hits, *, batch_id):
        self.profiles.append(("rerank", profile.name))
        if profile.name == self.fail_profile:
            raise ProviderFailure("failed")
        return self.decisions.pop(0)

    async def reformulate(self, profile, current, query, history, *, batch_id):
        self.profiles.append(("reformulate", profile.name))
        return "rewritten query"

    async def compose(self, profile, current, hits, ranked_ids, *, batch_id):
        self.profiles.append(("compose", profile.name))
        if profile.name == self.fail_compose_profile:
            raise ProviderFailure("composition failed")
        return "grounded answer"


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_answer_and_initial_query_uses_only_previous_exchange(self):
        retriever = FakeRetriever()
        generation = FakeGeneration([EvidenceDecision("answer", (0,))])
        pipeline = RAGPipeline(retriever, generation, {"kimi": PRIMARY, "backup": BACKUP})
        history = (
            ConversationExchange("old topic", "old answer"),
            ConversationExchange("recent topic", "recent answer"),
        )
        reply = await pipeline.run("two days", history, PRIMARY, batch_id="b1")
        self.assertEqual(reply.kind, "answer")
        self.assertNotIn("old topic", retriever.queries[0])
        self.assertIn("recent topic", retriever.queries[0])

    async def test_clarify_does_not_compose(self):
        generation = FakeGeneration([EvidenceDecision("clarify", clarification="What does that refer to?")])
        pipeline = RAGPipeline(FakeRetriever(), generation, {"kimi": PRIMARY, "backup": BACKUP})
        reply = await pipeline.run("two days", (), PRIMARY, batch_id="b2")
        self.assertEqual(reply.kind, "clarify")
        self.assertEqual(reply.text, "What does that refer to?")
        self.assertNotIn(("compose", "kimi"), generation.profiles)

    async def test_weak_reformulates_once_then_is_deterministic(self):
        generation = FakeGeneration([EvidenceDecision("weak"), EvidenceDecision("weak")])
        retriever = FakeRetriever([RESULT, RESULT])
        pipeline = RAGPipeline(retriever, generation, {"kimi": PRIMARY, "backup": BACKUP})
        reply = await pipeline.run("unknown", (), PRIMARY, batch_id="b3")
        self.assertEqual(reply.kind, "insufficient")
        self.assertEqual(retriever.queries[-1], "rewritten query")
        self.assertEqual(generation.profiles.count(("reformulate", "kimi")), 1)

    async def test_failover_restarts_reranking_before_composition(self):
        generation = FakeGeneration([EvidenceDecision("answer", (0,))], fail_profile="kimi")
        pipeline = RAGPipeline(FakeRetriever(), generation, {"kimi": PRIMARY, "backup": BACKUP})
        reply = await pipeline.run("question", (), PRIMARY, batch_id="b4")
        self.assertEqual(reply.kind, "answer")
        self.assertEqual(
            generation.profiles,
            [("rerank", "kimi"), ("rerank", "backup"), ("compose", "backup")],
        )

    async def test_kimi_composition_failure_reranks_with_openai(self):
        generation = FakeGeneration(
            [EvidenceDecision("answer", (0,)), EvidenceDecision("answer", (0,))],
            fail_compose_profile="kimi",
        )
        pipeline = RAGPipeline(FakeRetriever(), generation, {"kimi": PRIMARY, "backup": BACKUP})
        reply = await pipeline.run("question", (), PRIMARY, batch_id="b5")
        self.assertEqual(reply.text, "grounded answer")
        self.assertEqual(
            generation.profiles,
            [
                ("rerank", "kimi"),
                ("compose", "kimi"),
                ("rerank", "backup"),
                ("compose", "backup"),
            ],
        )
