"""Offline ingestion: load knowledge docs, chunk, embed, build FAISS index."""

import argparse
import json
import sys
from pathlib import Path

import faiss
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

import config


def load_taxonomy() -> dict:
    with open(config.INTENTS_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_documents(taxonomy: dict) -> list[dict]:
    """Walk knowledge/ and return doc records with metadata."""
    valid = {}
    for cat in taxonomy["categories"]:
        for sub in cat["sub_intents"]:
            valid[(cat["slug"], sub["slug"])] = True

    docs = []
    for path in sorted(config.KNOWLEDGE_DIR.rglob("*")):
        if path.suffix not in (".txt", ".md") or not path.is_file():
            continue
        category = path.parent.name
        sub_category = path.stem
        rel_source = str(path.relative_to(config.BASE_DIR))

        if (category, sub_category) not in valid:
            print(f"WARN: orphan file not in taxonomy: {rel_source}", file=sys.stderr)

        docs.append({
            "text": path.read_text(encoding="utf-8").strip(),
            "category": category,
            "sub_category": sub_category,
            "source": rel_source,
        })

    if not docs:
        raise SystemExit(f"No documents found in {config.KNOWLEDGE_DIR}")
    return docs


def chunk_documents(docs: list[dict]) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    chunks = []
    for doc in docs:
        for piece in splitter.split_text(doc["text"]):
            if piece.strip():
                chunks.append({
                    "text": piece.strip(),
                    "category": doc["category"],
                    "sub_category": doc["sub_category"],
                    "source": doc["source"],
                })
    return chunks


def embed_texts(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    vectors = model.encode(texts, batch_size=config.EMBED_BATCH_SIZE, show_progress_bar=False, convert_to_numpy=True)
    vectors = vectors.astype(np.float32)
    faiss.normalize_L2(vectors)
    return vectors


def build_index(chunks: list[dict], vectors: np.ndarray) -> None:
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    chunks_json = {str(i): chunk for i, chunk in enumerate(chunks)}
    faiss.write_index(index, str(config.INDEX_PATH))
    with open(config.CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks_json, f, ensure_ascii=False, indent=2)

    print(f"Indexed {len(chunks)} chunks -> {config.INDEX_PATH}, {config.CHUNKS_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS index from knowledge/")
    parser.add_argument("--rebuild", action="store_true", help="Overwrite existing index")
    args = parser.parse_args()

    if config.INDEX_PATH.exists() and not args.rebuild:
        print("Index already exists. Use --rebuild to overwrite.")
        return

    taxonomy = load_taxonomy()
    docs = load_documents(taxonomy)
    print(f"Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"Split into {len(chunks)} chunks")

    print(f"Loading local embedding model ({config.LOCAL_EMBEDDING_MODEL})...")
    model = SentenceTransformer(config.LOCAL_EMBEDDING_MODEL)
    vectors = embed_texts(model, [c["text"] for c in chunks])
    build_index(chunks, vectors)


if __name__ == "__main__":
    main()
