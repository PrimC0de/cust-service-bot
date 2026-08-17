"""Calibrated dense-only RAG pipeline."""

from __future__ import annotations

from app.models import (
    CompositionResult,
    ConversationExchange,
    ModelProfileSnapshot,
    PipelineReply,
    ReformulationResult,
    RetrievalHit,
    UnresolvedClarification,
)
from app.providers.router import ProviderFailure
from app.rag.generation import GenerationService, insufficient_response
from app.rag.retrieval.dense import DenseRetriever


class RAGPipeline:
    def __init__(
        self,
        retriever: DenseRetriever,
        generation: GenerationService,
        profiles: dict[str, ModelProfileSnapshot],
        *,
        top_k: int = 4,
    ):
        self.retriever = retriever
        self.generation = generation
        self.profiles = profiles
        self.top_k = top_k

    async def run(
        self,
        current: str,
        history: tuple[ConversationExchange, ...],
        profile: ModelProfileSnapshot,
        *,
        batch_id: str,
        unresolved: UnresolvedClarification | None = None,
    ) -> PipelineReply:
        hits = await self.retriever.search(current, self.top_k, batch_id=batch_id)
        if self._strong(hits):
            return await self._compose_reply(
                profile,
                current,
                history,
                hits,
                retrieval_query=current,
                batch_id=batch_id,
                clarification_already_used=unresolved is not None,
            )

        weak_query = current
        if unresolved is not None:
            weak_query = (
                f"Unresolved question: {unresolved.reformulated_query}\n\n"
                f"User clarification: {current}"
            )

        reformulated = await self._reformulate(
            profile, current, weak_query, history, batch_id=batch_id
        )
        if reformulated.answer:
            return PipelineReply(reformulated.answer, "answer")
        recovered = await self.retriever.search(
            reformulated.query, self.top_k, batch_id=batch_id
        )
        if self._strong(recovered):
            return await self._compose_reply(
                profile,
                current,
                history,
                recovered,
                retrieval_query=reformulated.query,
                batch_id=batch_id,
                clarification_already_used=unresolved is not None,
            )

        if unresolved is not None:
            return PipelineReply(insufficient_response(current), "insufficient")
        pending = UnresolvedClarification(
            current, reformulated.query, reformulated.clarification
        )
        return PipelineReply(reformulated.clarification, "clarify", pending)

    def _strong(self, hits: tuple[RetrievalHit, ...]) -> bool:
        threshold = self.retriever.confidence_threshold
        return bool(hits) and threshold is not None and hits[0].dense_score >= threshold

    def _backup(self, profile: ModelProfileSnapshot) -> ModelProfileSnapshot | None:
        return self.profiles.get(profile.backup_profile) if profile.backup_profile else None

    async def _reformulate(
        self,
        profile: ModelProfileSnapshot,
        current: str,
        query: str,
        history: tuple[ConversationExchange, ...],
        *,
        batch_id: str,
    ) -> ReformulationResult:
        try:
            return await self.generation.reformulate(
                profile, current, query, history, batch_id=batch_id
            )
        except ProviderFailure:
            backup = self._backup(profile)
            if backup is None:
                raise
            return await self.generation.reformulate(
                backup, current, query, history, batch_id=batch_id
            )

    async def _compose_reply(
        self,
        profile: ModelProfileSnapshot,
        current: str,
        history: tuple[ConversationExchange, ...],
        hits: tuple[RetrievalHit, ...],
        *,
        retrieval_query: str,
        batch_id: str,
        clarification_already_used: bool,
    ) -> PipelineReply:
        try:
            result = await self.generation.compose(
                profile, current, history, hits, batch_id=batch_id
            )
        except ProviderFailure:
            backup = self._backup(profile)
            if backup is None:
                raise
            result = await self.generation.compose(
                backup, current, history, hits, batch_id=batch_id
            )
        return self._composition_reply(
            result, current, retrieval_query, clarification_already_used
        )

    @staticmethod
    def _composition_reply(
        result: CompositionResult,
        current: str,
        retrieval_query: str,
        clarification_already_used: bool,
    ) -> PipelineReply:
        if result.kind == "answer":
            return PipelineReply(result.text, "answer")
        if clarification_already_used:
            return PipelineReply(insufficient_response(current), "insufficient")
        pending = UnresolvedClarification(current, retrieval_query, result.text)
        return PipelineReply(result.text, "clarify", pending)
