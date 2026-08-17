"""Hosted query embeddings and local FAISS search."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import faiss
import numpy as np
from openai import AsyncOpenAI

from app.models import KnowledgeChunk
from app.providers.router import is_transient_error

logger = logging.getLogger(__name__)


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

        if client is None or not index_path.exists() or not manifest_path.exists():
            return

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("schema_version") != 1
            or manifest.get("chunking_method") != "structure-aware-v1"
            or manifest.get("embedding_model") != model
            or manifest.get("chunk_count") != len(chunks)
        ):
            logger.warning("Dense index manifest is incompatible; using BM25 only")
            return

        index = faiss.read_index(str(index_path))
        if index.ntotal != len(chunks) or manifest.get("dimensions") != index.d:
            logger.warning("Dense index manifest mismatch; using BM25 only")
            return
        self.index = index

    @property
    def available(self) -> bool:
        return self.client is not None and self.index is not None

    async def search(self, query: str, k: int, *, batch_id: str = "startup") -> list[tuple[int, float]]:
        if not self.available:
            return []

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
                return [
                    (int(chunk_id), float(score))
                    for chunk_id, score in zip(ids[0], scores[0], strict=True)
                    if chunk_id >= 0 and int(chunk_id) in self.chunks
                ]
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "batch_id=%s stage=query_embedding profile=openai attempt=%d error_type=%s",
                    batch_id,
                    attempt + 1,
                    type(exc).__name__,
                )
                if not is_transient_error(exc):
                    break

        logger.warning(
            "batch_id=%s stage=dense_retrieval profile=openai attempt=2 error_type=%s",
            batch_id,
            type(last_error).__name__,
        )
        return []
