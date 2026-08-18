"""Build calibrated dense-retrieval artifacts."""

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


SCHEMA_VERSION = 2


def _chunk_record(chunk: KnowledgeChunk) -> dict:
    record = asdict(chunk)
    record["section_path"] = list(chunk.section_path)
    return record


def load_evaluation_cases(path: Path) -> list[dict]:
    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("Retrieval evaluation must be a non-empty JSON list")
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("query"), str):
            raise ValueError("Every retrieval case requires a query")
        expected = case.get("expected_source")
        if expected is not None and not isinstance(expected, str):
            raise ValueError("expected_source must be a string or null")
    if not any(case.get("expected_source") for case in cases):
        raise ValueError("Retrieval evaluation requires supported cases")
    if not any(case.get("expected_source") is None for case in cases):
        raise ValueError("Retrieval evaluation requires out-of-KB cases")
    return cases


async def embed_texts(
    client: AsyncOpenAI, model: str, texts: list[str], batch_size: int
) -> np.ndarray:
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        response = await client.embeddings.create(
            model=model, input=texts[start : start + batch_size]
        )
        vectors.extend(item.embedding for item in sorted(response.data, key=lambda item: item.index))
    return np.asarray(vectors, dtype=np.float32)


def calibrate_confidence(
    chunk_vectors: np.ndarray,
    chunks: list[KnowledgeChunk],
    query_vectors: np.ndarray,
    cases: list[dict],
    *,
    top_k: int = 4,
) -> dict:
    # ponytail: exact ranking is simplest for this small calibration corpus.
    similarities = query_vectors @ chunk_vectors.T
    ids = np.argsort(-similarities, axis=1)[:, : min(top_k, len(chunks))]
    scores = np.take_along_axis(similarities, ids, axis=1)
    positive_scores: list[float] = []
    negative_scores: list[float] = []
    for case, case_scores, case_ids in zip(cases, scores, ids, strict=True):
        best_score = float(case_scores[0])
        expected = case.get("expected_source")
        if expected is None:
            negative_scores.append(best_score)
            continue
        sources = {chunks[int(identifier)].source for identifier in case_ids if identifier >= 0}
        if expected not in sources:
            raise ValueError(f"Expected source absent from top {top_k}: {case['query']}")
        positive_scores.append(best_score)

    minimum_positive = min(positive_scores)
    maximum_negative = max(negative_scores)
    if maximum_negative >= minimum_positive:
        raise ValueError("Unsafe calibration: supported and out-of-KB scores overlap")
    return {
        "confidence_threshold": minimum_positive,
        "positive_cases": len(positive_scores),
        "negative_cases": len(negative_scores),
        "minimum_positive_score": minimum_positive,
        "maximum_negative_score": maximum_negative,
    }


def build_indexes(
    chunks: list[KnowledgeChunk],
    *,
    indexes_dir: Path,
    embedding_model: str,
    evaluation_cases: list[dict],
    vectors: np.ndarray,
) -> dict:
    if not chunks:
        raise ValueError("Cannot index an empty chunk collection")
    if not evaluation_cases:
        raise ValueError("Retrieval evaluation cases are required")

    vectors = np.asarray(vectors, dtype=np.float32).copy()
    faiss.normalize_L2(vectors)
    chunk_count = len(chunks)
    chunk_vectors = np.ascontiguousarray(vectors[:chunk_count])
    query_vectors = np.ascontiguousarray(vectors[chunk_count:])
    dimensions = int(chunk_vectors.shape[1])
    index = faiss.IndexFlatIP(dimensions)
    index.add(chunk_vectors)
    calibration = calibrate_confidence(
        chunk_vectors, chunks, query_vectors, evaluation_cases
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "chunking_method": CHUNKING_METHOD,
        "embedding_provider": "openai",
        "embedding_model": embedding_model,
        "dimensions": dimensions,
        "chunk_count": chunk_count,
        "dense_available": True,
        "confidence_threshold": calibration["confidence_threshold"],
        "calibration": {
            key: value for key, value in calibration.items() if key != "confidence_threshold"
        },
        "built_at": datetime.now(timezone.utc).isoformat(),
    }

    indexes_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = indexes_dir / "chunks.json"
    index_path = indexes_dir / "index.faiss"
    manifest_path = indexes_dir / "manifest.json"
    temporary_chunks = indexes_dir / "chunks.json.tmp"
    temporary_index = indexes_dir / "index.tmp.faiss"
    temporary_manifest = indexes_dir / "manifest.json.tmp"
    temporary_chunks.write_text(
        json.dumps([_chunk_record(chunk) for chunk in chunks], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    faiss.write_index(index, str(temporary_index))
    temporary_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary_chunks.replace(chunks_path)
    temporary_index.replace(index_path)
    temporary_manifest.replace(manifest_path)
    return manifest


def manifest_compatible(manifest: dict, embedding_model: str) -> bool:
    threshold = manifest.get("confidence_threshold")
    return (
        manifest.get("schema_version") == SCHEMA_VERSION
        and manifest.get("chunking_method") == CHUNKING_METHOD
        and manifest.get("embedding_provider") == "openai"
        and manifest.get("embedding_model") == embedding_model
        and manifest.get("dense_available") is True
        and isinstance(manifest.get("dimensions"), int)
        and isinstance(threshold, (int, float))
        and -1.0 <= threshold <= 1.0
    )
