"""History-aware query recovery and grounded generation."""

from __future__ import annotations

from app.models import CompositionResult, ModelProfileSnapshot, ReformulationResult, RetrievalHit
from app.providers.router import ProviderFailure, ProviderRouter
from app.rag.context import format_history
from app.rag.prompts import (
    COMPOSE_SYSTEM,
    COMPOSE_USER,
    REFORMULATE_SYSTEM,
    REFORMULATE_USER,
)


class GenerationService:
    def __init__(self, router: ProviderRouter):
        self.router = router

    async def reformulate(
        self,
        profile: ModelProfileSnapshot,
        current: str,
        query: str,
        history,
        *,
        batch_id: str,
    ) -> ReformulationResult:
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
            max_tokens=200,
        )
        answer = data.get("answer")
        if isinstance(answer, str) and answer.strip():
            return ReformulationResult("", "", answer.strip()[:4000])
        rewritten = data.get("query")
        clarification = data.get("clarification")
        if not isinstance(rewritten, str) or not rewritten.strip():
            raise ProviderFailure("Reformulator returned an empty query")
        if not isinstance(clarification, str) or not clarification.strip():
            raise ProviderFailure("Reformulator returned an empty clarification")
        return ReformulationResult(rewritten.strip(), clarification.strip())

    async def compose(
        self,
        profile: ModelProfileSnapshot,
        current: str,
        history,
        hits: tuple[RetrievalHit, ...],
        *,
        batch_id: str,
    ) -> CompositionResult:
        if not hits:
            raise ProviderFailure("No grounded chunks were selected for composition")
        context = "\n\n".join(
            f"--- {hit.chunk.document_title} > {' > '.join(hit.chunk.section_path)} ---\n"
            f"{hit.chunk.text}"
            for hit in hits
        )
        data = await self.router.json(
            profile,
            profile.compose_model,
            [
                {"role": "system", "content": COMPOSE_SYSTEM},
                {
                    "role": "user",
                    "content": COMPOSE_USER.format(
                        history=format_history(history), current=current, context=context
                    ),
                },
            ],
            batch_id=batch_id,
            stage="compose",
            max_tokens=800,
        )
        kind = data.get("kind")
        text = data.get("text")
        if kind not in {"answer", "clarify"} or not isinstance(text, str) or not text.strip():
            raise ProviderFailure("Composer returned an invalid result")
        return CompositionResult(kind, text.strip()[:4000])


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
