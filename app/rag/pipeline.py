"""Calibrated dense-only RAG pipeline."""

from __future__ import annotations

import logging
import re
import unicodedata

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


logger = logging.getLogger(__name__)


ADDRESS_ONLY = re.compile(
    r"^\s*(?:(?:hai|halo|hello|hi|hey)\s*[,!]*\s*)?"
    r"(?P<term>bang|bos|boss|bro|kak|min|admin|gan|sis|mas|mbak)\s*[!?.,]*\s*$",
    re.IGNORECASE,
)
AMOUNT = re.compile(
    r"(?ix)(?<!\w)(?:"
    r"rp\.?\s*\d+(?:[.,]\d+)*|"
    r"\d+(?:[.,]\d+)*\s*(?:k|m|b|rb|ribu|jt|juta)\b|"
    r"(?:nominal|jumlah|amount)\s*[:=]?\s*\d+(?:[.,]\d+)*"
    r")"
)


def normalize_input(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def extract_amounts(text: str) -> str:
    matches = dict.fromkeys(match.group(0).strip() for match in AMOUNT.finditer(text))
    return ", ".join(matches) if matches else "none"


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
        address = ADDRESS_ONLY.fullmatch(current)
        if address:
            self._log_diagnostics(
                batch_id,
                "initial",
                current,
                (),
                "direct_address_reply",
                "none",
            )
            return PipelineReply(
                f"Siap, {address.group('term').lower()} 👋 Ada yang bisa dibantu?",
                "answer",
            )

        hits = await self.retriever.search(current, self.top_k, batch_id=batch_id)
        if self._strong(hits):
            self._log_diagnostics(
                batch_id, "initial", current, hits, "grounded_composition", "none"
            )
            return await self._compose_reply(
                profile,
                current,
                history,
                hits,
                retrieval_query=current,
                batch_id=batch_id,
                clarification_already_used=unresolved is not None,
            )
        self._log_diagnostics(
            batch_id,
            "initial",
            current,
            hits,
            "reformulate_and_retry",
            "no_retrieval_hits" if not hits else "top_score_below_confidence_threshold",
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
            self._log_diagnostics(
                batch_id,
                "recovery",
                reformulated.query,
                recovered,
                "grounded_composition",
                "initial_retrieval_below_confidence_threshold",
            )
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
            self._log_diagnostics(
                batch_id,
                "recovery",
                reformulated.query,
                recovered,
                "insufficient_information",
                "reformulated_query_below_threshold_after_clarification",
            )
            return PipelineReply(insufficient_response(current), "insufficient")
        self._log_diagnostics(
            batch_id,
            "recovery",
            reformulated.query,
            recovered,
            "clarification",
            "reformulated_query_below_confidence_threshold",
        )
        pending = UnresolvedClarification(
            current, reformulated.query, reformulated.clarification
        )
        return PipelineReply(reformulated.clarification, "clarify", pending)

    def _strong(self, hits: tuple[RetrievalHit, ...]) -> bool:
        threshold = self.retriever.confidence_threshold
        return bool(hits) and threshold is not None and hits[0].dense_score >= threshold

    def _log_diagnostics(
        self,
        batch_id: str,
        stage: str,
        query: str,
        hits: tuple[RetrievalHit, ...],
        workflow: str,
        fallback_reason: str,
    ) -> None:
        examples = "\n".join(
            f"{index}. {' | '.join(hit.chunk.embedding_text.splitlines())}"
            for index, hit in enumerate(hits[:3], 1)
        ) or "(none)"
        intent = hits[0].chunk.sub_category if hits else "none"
        confidence = f"{hits[0].dense_score:.4f}" if hits else "none"
        logger.info(
            "RAG DIAGNOSTIC batch_id=%s stage=%s\n\n"
            "INPUT\n%r\n\n"
            "NORMALIZED INPUT\n%r\n\n"
            "RETRIEVED EXAMPLES\n%s\n\n"
            "CLASSIFIER\nintent = %s\nconfidence = %s\n\n"
            "EXTRACTED ENTITIES\namount = %s\n\n"
            "WORKFLOW SELECTED\n%s\n\n"
            "FALLBACK REASON\n%s",
            batch_id,
            stage,
            query,
            normalize_input(query),
            examples,
            intent,
            confidence,
            extract_amounts(query),
            workflow,
            fallback_reason,
        )

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
