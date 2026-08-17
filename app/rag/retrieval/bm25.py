"""Small multilingual BM25 index built from chunk metadata."""

from __future__ import annotations

import re
import unicodedata

import numpy as np
from rank_bm25 import BM25Okapi

from app.models import KnowledgeChunk

LATIN_TOKEN = re.compile(r"[a-z0-9]+(?:[-_./][a-z0-9]+)*", re.IGNORECASE)
CJK_RUN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")


def tokenize(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).lower()
    tokens = LATIN_TOKEN.findall(normalized)
    for run in CJK_RUN.findall(normalized):
        tokens.extend(run)
        tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    return tokens


class BM25Retriever:
    def __init__(self, chunks: list[KnowledgeChunk]):
        self.chunks = chunks
        corpus = [tokenize(chunk.embedding_text) for chunk in chunks]
        self.index = BM25Okapi(corpus) if corpus else None

    @property
    def available(self) -> bool:
        return self.index is not None

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        if self.index is None:
            return []
        scores = self.index.get_scores(tokenize(query))
        order = np.argsort(scores)[::-1][: min(k, len(self.chunks))]
        return [(self.chunks[int(i)].chunk_id, float(scores[int(i)])) for i in order]
