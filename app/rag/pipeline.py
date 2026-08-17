"""Evidence-first hybrid RAG pipeline."""

from __future__ import annotations

from app.models import (
    ConversationExchange,
    EvidenceDecision,
    ModelProfileSnapshot,
    PipelineReply,
)
from app.providers.router import ProviderFailure
from app.rag.context import build_initial_query
from app.rag.generation import GenerationService, insufficient_response
from app.rag.retrieval.hybrid import HybridRetriever


class RAGPipeline:
    def __init__(
        self,
        retriever: HybridRetriever,
        generation: GenerationService,
        profiles: dict[str, ModelProfileSnapshot],
    ):
        self.retriever = retriever
        self.generation = generation
        self.profiles = profiles
        self.agreement_bypass_enabled = False

    async def run(
        self,
        current: str,
        history: tuple[ConversationExchange, ...],
        profile: ModelProfileSnapshot,
        *,
        batch_id: str,
    ) -> PipelineReply:
        previous = history[-1] if history else None
        query = build_initial_query(current, previous)
        retrieval = await self.retriever.search(query, batch_id=batch_id)

        try:
            return await self._run_profile(
                current,
                history,
                query,
                retrieval,
                profile,
                batch_id=batch_id,
                recovered=False,
            )
        except ProviderFailure:
            if not profile.backup_profile:
                raise
            backup = self.profiles[profile.backup_profile]
            return await self._run_profile(
                current,
                history,
                query,
                retrieval,
                backup,
                batch_id=batch_id,
                recovered=False,
            )

    async def _run_profile(
        self,
        current,
        history,
        query,
        retrieval,
        profile,
        *,
        batch_id,
        recovered,
    ) -> PipelineReply:
        if not retrieval.hits:
            decision = EvidenceDecision(status="weak")
        else:
            decision = await self.generation.decide(
                profile,
                query,
                retrieval.hits,
                batch_id=batch_id,
            )

        if decision.status == "clarify":
            return PipelineReply(text=decision.clarification or "Could you clarify?", kind="clarify")

        if decision.status == "weak":
            if recovered:
                return PipelineReply(text=insufficient_response(current), kind="insufficient")
            rewritten = await self.generation.reformulate(
                profile,
                current,
                query,
                history,
                batch_id=batch_id,
            )
            retry_retrieval = await self.retriever.search(rewritten, batch_id=batch_id)
            return await self._run_profile(
                current,
                history,
                rewritten,
                retry_retrieval,
                profile,
                batch_id=batch_id,
                recovered=True,
            )

        answer = await self.generation.compose(
            profile,
            current,
            retrieval.hits,
            decision.ranked_chunk_ids,
            batch_id=batch_id,
        )
        return PipelineReply(text=answer, kind="answer")
