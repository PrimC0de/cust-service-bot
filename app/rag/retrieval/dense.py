"""Hosted query embeddings and calibrated local FAISS search."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import faiss
import numpy as np
from openai import AsyncOpenAI

from app.models import KnowledgeChunk, RetrievalHit
from app.providers.router import is_transient_error
from app.rag.ingestion.indexer import manifest_compatible

logger = logging.getLogger(__name__)


class DenseRetrievalError(RuntimeError):
    """Dense retrieval could not run, rather than merely finding weak evidence."""


def load_chunks(path: Path) -> list[KnowledgeChunk]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    records = raw.values() if isinstance(raw, dict) else raw
    return [
        KnowledgeChunk(
            chunk_id=int(item["chunk_id"]),
            text=item["text"],
            embedding_text=item.get("embedding_text", item["text"]),
            category=item["category"],
            sub_category=item["sub_category"],
            source=item["source"],
            document_title=item.get("document_title", ""),
            section_path=tuple(item.get("section_path", ())),
        )
        for item in records
    ]


class DenseRetriever:
    def __init__(
        self,
        client: AsyncOpenAI | None,
        model: str,
        chunks: list[KnowledgeChunk],
        index_path: Path,
        manifest_path: Path,
    ):
        self.client = client
        self.model = model
        self.chunks = {chunk.chunk_id: chunk for chunk in chunks}
        self.index = None
        self.confidence_threshold: float | None = None

        if client is None or not index_path.exists() or not manifest_path.exists():
            return

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not manifest_compatible(manifest, model) or manifest.get("chunk_count") != len(chunks):
            logger.warning("Dense index manifest is incompatible")
            return

        index = faiss.read_index(str(index_path))
        if index.ntotal != len(chunks) or manifest.get("dimensions") != index.d:
            logger.warning("Dense index manifest mismatch")
            return
        self.index = index
        self.confidence_threshold = float(manifest["confidence_threshold"])

    @property
    def available(self) -> bool:
        return self.client is not None and self.index is not None and self.confidence_threshold is not None

    async def search(
        self, query: str, k: int = 4, *, batch_id: str = "startup"
    ) -> tuple[RetrievalHit, ...]:
        if not self.available:
            raise DenseRetrievalError("Dense retrieval is unavailable")

        last_error: Exception | None = None
        for attempt in range(2):
            try:
                response = await self.client.embeddings.create(model=self.model, input=[query])
                vector = np.asarray([response.data[0].embedding], dtype=np.float32)
                if vector.shape[1] != self.index.d:
                    raise ValueError(
                        f"Query embedding dimension {vector.shape[1]} does not match index {self.index.d}"
                    )
                faiss.normalize_L2(vector)
                scores, ids = self.index.search(vector, min(k, self.index.ntotal))
                return tuple(
                    RetrievalHit(self.chunks[int(chunk_id)], float(score))
                    for chunk_id, score in zip(ids[0], scores[0], strict=True)
                    if chunk_id >= 0 and int(chunk_id) in self.chunks
                )
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "batch_id=%s stage=query_embedding profile=openrouter attempt=%d error_type=%s",
                    batch_id,
                    attempt + 1,
                    type(exc).__name__,
                )
                if not is_transient_error(exc):
                    break

        logger.error(
            "batch_id=%s stage=dense_retrieval profile=openrouter attempt=%d error_type=%s",
            batch_id,
            attempt + 1,
            type(last_error).__name__,
        )
        raise DenseRetrievalError("Query embedding failed") from last_error
