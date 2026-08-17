"""Global dense and lexical candidate retrieval."""

from __future__ import annotations

from app.models import HybridRetrievalResult, KnowledgeChunk, RetrievalHit
from app.rag.retrieval.bm25 import BM25Retriever
from app.rag.retrieval.dense import DenseRetriever


class HybridRetriever:
    def __init__(
        self,
        chunks: list[KnowledgeChunk],
        dense: DenseRetriever,
        bm25: BM25Retriever,
        *,
        top_k: int = 10,
    ):
        self.chunks = {chunk.chunk_id: chunk for chunk in chunks}
        self.dense = dense
        self.bm25 = bm25
        self.top_k = top_k

    async def search(self, query: str, *, batch_id: str = "startup") -> HybridRetrievalResult:
        dense_results = await self.dense.search(query, self.top_k, batch_id=batch_id)
        bm25_results = self.bm25.search(query, self.top_k)

        dense_scores = dict(dense_results)
        bm25_scores = dict(bm25_results)
        fused: dict[int, float] = {}
        for rank, (chunk_id, _) in enumerate(dense_results):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (60 + rank + 1)
        for rank, (chunk_id, _) in enumerate(bm25_results):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (60 + rank + 1)

        ordered = sorted(fused, key=fused.get, reverse=True)[: self.top_k]
        hits = tuple(
            RetrievalHit(
                chunk=self.chunks[chunk_id],
                dense_score=dense_scores.get(chunk_id),
                bm25_score=bm25_scores.get(chunk_id),
                fused_score=fused[chunk_id],
            )
            for chunk_id in ordered
            if chunk_id in self.chunks
        )
        return HybridRetrievalResult(hits=hits, dense_available=bool(dense_results))
