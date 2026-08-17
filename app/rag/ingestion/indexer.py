"""Write ordered chunk metadata and an optional hosted-embedding FAISS index."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np
from openai import AsyncOpenAI

from app.models import KnowledgeChunk
from app.rag.ingestion.chunker import CHUNKING_METHOD


SCHEMA_VERSION = 1


def _chunk_record(chunk: KnowledgeChunk) -> dict:
    record = asdict(chunk)
    record["section_path"] = list(chunk.section_path)
    return record


async def _embed(
    client: AsyncOpenAI,
    model: str,
    chunks: list[KnowledgeChunk],
    batch_size: int,
) -> np.ndarray:
    vectors: list[list[float]] = []
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        response = await client.embeddings.create(
            model=model,
            input=[chunk.embedding_text for chunk in batch],
        )
        vectors.extend(item.embedding for item in sorted(response.data, key=lambda item: item.index))
    matrix = np.asarray(vectors, dtype=np.float32)
    faiss.normalize_L2(matrix)
    return matrix


async def build_indexes(
    chunks: list[KnowledgeChunk],
    *,
    indexes_dir: Path,
    embedding_client: AsyncOpenAI | None,
    embedding_model: str,
    embedding_batch_size: int = 100,
) -> dict:
    if not chunks:
        raise ValueError("Cannot index an empty chunk collection")
    indexes_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = indexes_dir / "chunks.json"
    index_path = indexes_dir / "index.faiss"
    manifest_path = indexes_dir / "manifest.json"

    # Invalidate the old dense pair before replacing its ordered chunk corpus.
    # If hosted embedding fails, startup can still safely consume chunks via BM25.
    index_path.unlink(missing_ok=True)
    manifest_path.unlink(missing_ok=True)

    chunks_path.write_text(
        json.dumps([_chunk_record(chunk) for chunk in chunks], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    dimensions: int | None = None
    if embedding_client is not None:
        vectors = await _embed(
            embedding_client,
            embedding_model,
            chunks,
            embedding_batch_size,
        )
        dimensions = int(vectors.shape[1])
        index = faiss.IndexFlatIP(dimensions)
        index.add(vectors)
        faiss.write_index(index, str(index_path))

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "chunking_method": CHUNKING_METHOD,
        "embedding_provider": "openai",
        "embedding_model": embedding_model,
        "dimensions": dimensions,
        "chunk_count": len(chunks),
        "dense_available": dimensions is not None,
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def manifest_compatible(manifest: dict, embedding_model: str) -> bool:
    return (
        manifest.get("schema_version") == SCHEMA_VERSION
        and manifest.get("chunking_method") == CHUNKING_METHOD
        and manifest.get("embedding_model") == embedding_model
    )
