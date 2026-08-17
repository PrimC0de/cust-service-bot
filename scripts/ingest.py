"""Thin command line entry point for knowledge ingestion."""

from __future__ import annotations

import argparse
import asyncio

from openai import AsyncOpenAI

from app.config.settings import Settings
from app.rag.ingestion.chunker import chunk_documents
from app.rag.ingestion.indexer import build_indexes
from app.rag.ingestion.parser import parse_documents


async def ingest(settings: Settings, rebuild: bool) -> None:
    if settings.chunks_path.exists() and not rebuild:
        print("Indexes already exist. Use --rebuild to replace them.")
        return

    documents = parse_documents(settings.knowledge_dir, settings.taxonomy_path)
    chunks = chunk_documents(
        documents,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    client = None
    if settings.openai_api_key:
        client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            max_retries=0,
        )
    else:
        print("OPENAI_API_KEY is absent; writing a BM25-only chunk index.")

    manifest = await build_indexes(
        chunks,
        indexes_dir=settings.indexes_dir,
        embedding_client=client,
        embedding_model=settings.embedding_model,
        embedding_batch_size=settings.embedding_batch_size,
    )
    mode = "dense + BM25" if manifest["dense_available"] else "BM25 only"
    print(f"Indexed {len(chunks)} chunks from {len(documents)} documents ({mode}).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build retrieval artifacts from the knowledge base")
    parser.add_argument("--rebuild", action="store_true", help="replace existing artifacts")
    args = parser.parse_args()
    asyncio.run(ingest(Settings(), args.rebuild))


if __name__ == "__main__":
    main()
