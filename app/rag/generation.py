"""Structured evidence decisions and grounded generation."""

from __future__ import annotations

from app.models import EvidenceDecision, ModelProfileSnapshot, RetrievalHit
from app.providers.router import ProviderFailure, ProviderRouter
from app.rag.context import format_history
from app.rag.prompts import (
    COMPOSE_SYSTEM,
    COMPOSE_USER,
    REFORMULATE_SYSTEM,
    REFORMULATE_USER,
    RERANK_SYSTEM,
    RERANK_USER,
)


class GenerationService:
    def __init__(self, router: ProviderRouter, *, top_n: int = 3):
        self.router = router
        self.top_n = top_n

    async def decide(
        self,
        profile: ModelProfileSnapshot,
        query: str,
        hits: tuple[RetrievalHit, ...],
        *,
        batch_id: str,
    ) -> EvidenceDecision:
        candidate_ids = {hit.chunk.chunk_id for hit in hits}
        candidates = "\n\n".join(
            f"[chunk_id={hit.chunk.chunk_id}] "
            f"{hit.chunk.document_title} > {' > '.join(hit.chunk.section_path)}\n"
            f"{hit.chunk.text}"
            for hit in hits
        )
        data = await self.router.json(
            profile,
            profile.rerank_model,
            [
                {"role": "system", "content": RERANK_SYSTEM},
                {"role": "user", "content": RERANK_USER.format(query=query, candidates=candidates)},
            ],
            batch_id=batch_id,
            stage="rerank",
        )

        status = data.get("status")
        if status not in {"answer", "clarify", "weak"}:
            raise ProviderFailure("Reranker returned an invalid evidence status")
        ranked = tuple(
            chunk_id
            for chunk_id in data.get("ranked_chunk_ids", [])
            if isinstance(chunk_id, int) and chunk_id in candidate_ids
        )[: self.top_n]
        clarification = data.get("clarification")

        if status == "answer" and not ranked:
            return EvidenceDecision(status="weak")
        if status == "clarify" and not isinstance(clarification, str):
            raise ProviderFailure("Reranker omitted its clarification question")
        return EvidenceDecision(status=status, ranked_chunk_ids=ranked, clarification=clarification)

    async def reformulate(
        self,
        profile: ModelProfileSnapshot,
        current: str,
        query: str,
        history,
        *,
        batch_id: str,
    ) -> str:
        data = await self.router.json(
            profile,
            profile.reformulate_model,
            [
                {"role": "system", "content": REFORMULATE_SYSTEM},
                {
                    "role": "user",
                    "content": REFORMULATE_USER.format(
                        history=format_history(history),
                        current=current,
                        query=query,
                    ),
                },
            ],
            batch_id=batch_id,
            stage="reformulate",
        )
        rewritten = data.get("query")
        if not isinstance(rewritten, str) or not rewritten.strip():
            raise ProviderFailure("Reformulator returned an empty query")
        return rewritten.strip()

    async def compose(
        self,
        profile: ModelProfileSnapshot,
        current: str,
        hits: tuple[RetrievalHit, ...],
        ranked_ids: tuple[int, ...],
        *,
        batch_id: str,
    ) -> str:
        by_id = {hit.chunk.chunk_id: hit for hit in hits}
        selected = [by_id[chunk_id] for chunk_id in ranked_ids if chunk_id in by_id]
        if not selected:
            raise ProviderFailure("No grounded chunks were selected for composition")
        context = "\n\n".join(
            f"--- {hit.chunk.document_title} > {' > '.join(hit.chunk.section_path)} ---\n"
            f"{hit.chunk.text}"
            for hit in selected
        )
        answer = await self.router.text(
            profile,
            profile.compose_model,
            [
                {"role": "system", "content": COMPOSE_SYSTEM},
                {"role": "user", "content": COMPOSE_USER.format(current=current, context=context)},
            ],
            batch_id=batch_id,
            stage="compose",
        )
        return answer[:4000]


TRADITIONAL_MARKERS = set("後發臺灣這個為與專業處理問題聯絡號碼資訊轉帳銀行額戶體驗應該說明還請")


def detect_language(text: str) -> str:
    if any(character in TRADITIONAL_MARKERS for character in text):
        return "zh-Hant"
    if any("\u3400" <= character <= "\u9fff" for character in text):
        return "zh-Hans"
    return "en"


INSUFFICIENT_RESPONSES = {
    "en": "I don't have enough verified information to answer that right now.",
    "zh-Hans": "目前没有足够的已验证资料来回答这个问题。",
    "zh-Hant": "目前沒有足夠的已驗證資料來回答這個問題。",
}

UNAVAILABLE_RESPONSES = {
    "en": "The service is unavailable right now. Please try again later.",
    "zh-Hans": "服务目前不可用，请稍后再试。",
    "zh-Hant": "服務目前不可用，請稍後再試。",
}


def insufficient_response(text: str) -> str:
    return INSUFFICIENT_RESPONSES[detect_language(text)]


def unavailable_response(text: str) -> str:
    return UNAVAILABLE_RESPONSES[detect_language(text)]


service_unavailable_response = unavailable_response
