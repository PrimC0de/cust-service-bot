"""Thin command line entry point for knowledge ingestion."""

from __future__ import annotations

import argparse
import asyncio

from openai import AsyncOpenAI

from app.config.settings import Settings
from app.rag.ingestion.chunker import chunk_documents
from app.rag.ingestion.indexer import build_indexes, embed_texts, load_evaluation_cases
from app.rag.ingestion.parser import parse_documents


def ingest(settings: Settings, rebuild: bool) -> None:
    if settings.chunks_path.exists() and not rebuild:
        print("Indexes already exist. Use --rebuild to replace them.")
        return

    documents = parse_documents(settings.knowledge_dir, settings.taxonomy_path)
    chunks = chunk_documents(
        documents,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    if not settings.openrouter_api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required for dense ingestion")
    cases = load_evaluation_cases(settings.retrieval_evaluation_path)
    print(f"Embedding {len(chunks)} chunks and {len(cases)} evaluation queries...", flush=True)
    client = AsyncOpenAI(
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        max_retries=0,
    )
    vectors = asyncio.run(
        embed_texts(
            client,
            settings.embedding_model,
            [chunk.embedding_text for chunk in chunks] + [case["query"] for case in cases],
            settings.embedding_batch_size,
        )
    )
    print("Calibrating and writing FAISS artifacts...", flush=True)
    manifest = build_indexes(
        chunks,
        indexes_dir=settings.indexes_dir,
        embedding_model=settings.embedding_model,
        evaluation_cases=cases,
        vectors=vectors,
    )
    print(
        f"Indexed {len(chunks)} chunks from {len(documents)} documents "
        f"(dense, confidence threshold {manifest['confidence_threshold']:.4f})."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build retrieval artifacts from the knowledge base")
    parser.add_argument("--rebuild", action="store_true", help="replace existing artifacts")
    args = parser.parse_args()
    ingest(Settings(), args.rebuild)


if __name__ == "__main__":
    main()
