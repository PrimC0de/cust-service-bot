"""RAG pipeline: classify, scoped retrieve, rerank, compose."""

import json
import logging
from dataclasses import dataclass

import faiss
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer

import config
from prompts import CLASSIFY_SYSTEM, CLASSIFY_USER, COMPOSE_SYSTEM, COMPOSE_USER, RERANK_SYSTEM, RERANK_USER

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    chunk_id: int
    text: str
    category: str
    sub_category: str
    source: str
    score: float = 0.0


def load_taxonomy() -> dict:
    with open(config.INTENTS_PATH, encoding="utf-8") as f:
        return json.load(f)


def format_taxonomy(taxonomy: dict) -> str:
    lines = []
    for cat in taxonomy["categories"]:
        lines.append(f"Category: {cat['slug']} ({cat['name']})")
        for sub in cat["sub_intents"]:
            lines.append(f"  - {sub['slug']}: {sub['name']} — {sub['classification_hint']}")
    return "\n".join(lines)


def validate_intent(taxonomy: dict, category: str, sub_category: str) -> tuple[str, str]:
    """Ensure slugs exist; fallback to first sub-intent in category or first overall."""
    cat_map = {c["slug"]: c for c in taxonomy["categories"]}
    if category in cat_map:
        sub_slugs = {s["slug"] for s in cat_map[category]["sub_intents"]}
        if sub_category in sub_slugs:
            return category, sub_category
        return category, cat_map[category]["sub_intents"][0]["slug"]
    first_cat = taxonomy["categories"][0]
    return first_cat["slug"], first_cat["sub_intents"][0]["slug"]


class RAGStore:
    def __init__(self, groq_client: Groq | None = None, embed_model: SentenceTransformer | None = None):
        self.groq_client = groq_client or Groq(api_key=config.GROQ_API_KEY)
        self.embed_model = embed_model or SentenceTransformer(config.LOCAL_EMBEDDING_MODEL)
        self.taxonomy = load_taxonomy()
        self.taxonomy_text = format_taxonomy(self.taxonomy)

        if not config.INDEX_PATH.exists() or not config.CHUNKS_PATH.exists():
            raise FileNotFoundError(
                "Index not found. Run `python ingest.py` first."
            )

        self.index = faiss.read_index(str(config.INDEX_PATH))
        with open(config.CHUNKS_PATH, encoding="utf-8") as f:
            raw = json.load(f)
        self.chunks: dict[int, Chunk] = {
            int(k): Chunk(chunk_id=int(k), **v) for k, v in raw.items()
        }

        self.by_category: dict[str, list[int]] = {}
        self.by_sub: dict[tuple[str, str], list[int]] = {}
        for cid, chunk in self.chunks.items():
            self.by_category.setdefault(chunk.category, []).append(cid)
            self.by_sub.setdefault((chunk.category, chunk.sub_category), []).append(cid)

    def classify_intent(self, question: str) -> tuple[str, str]:
        prompt = CLASSIFY_USER.format(taxonomy=self.taxonomy_text, question=question)
        for attempt in range(2):
            try:
                resp = self.groq_client.chat.completions.create(
                    model=config.GROQ_CHAT_MODEL,
                    messages=[
                        {"role": "system", "content": CLASSIFY_SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                )
                data = json.loads(resp.choices[0].message.content)
                category = data["category"]
                sub_category = data["sub_category"]
                return validate_intent(self.taxonomy, category, sub_category)
            except Exception as e:
                logger.warning("Classification parse failed (attempt %d): %s", attempt + 1, e)

        first = self.taxonomy["categories"][0]
        return first["slug"], first["sub_intents"][0]["slug"]

    def _embed(self, text: str) -> np.ndarray:
        vec = self.embed_model.encode([text], convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(vec)
        return vec

    def retrieve(
        self, question: str, category: str, sub_category: str, k: int | None = None
    ) -> list[Chunk]:
        k = k or config.RETRIEVAL_TOP_K
        query_vec = self._embed(question)

        candidate_ids = self.by_sub.get((category, sub_category), [])
        if len(candidate_ids) < k:
            candidate_ids = self.by_category.get(category, list(self.chunks.keys()))

        max_id = self.index.ntotal
        candidate_ids = [i for i in candidate_ids if i < max_id]
        if not candidate_ids:
            candidate_ids = list(range(max_id))

        id_array = np.array(candidate_ids, dtype=np.int64)
        sub_vectors = np.vstack([
            self.index.reconstruct(int(i)) for i in candidate_ids
        ])
        scores = (query_vec @ sub_vectors.T).flatten()

        top_local = np.argsort(scores)[::-1][:k]
        results = []
        for idx in top_local:
            cid = int(id_array[idx])
            chunk = self.chunks[cid]
            results.append(Chunk(
                chunk_id=cid,
                text=chunk.text,
                category=chunk.category,
                sub_category=chunk.sub_category,
                source=chunk.source,
                score=float(scores[idx]),
            ))
        return results

    def rerank(self, question: str, candidates: list[Chunk]) -> list[Chunk]:
        if not candidates:
            return []
        if len(candidates) <= config.RERANK_TOP_N:
            return candidates[: config.RERANK_TOP_N]

        numbered = "\n\n".join(
            f"[{i}] {c.text[:600]}" for i, c in enumerate(candidates)
        )
        prompt = RERANK_USER.format(question=question, candidates=numbered)

        try:
            resp = self.groq_client.chat.completions.create(
                model=config.GROQ_CHAT_MODEL,
                messages=[
                    {"role": "system", "content": RERANK_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            data = json.loads(resp.choices[0].message.content)
            ranked_ids = data["ranked_ids"]
        except Exception as e:
            logger.warning("Rerank step failed or error: %s. Falling back to vector order.", e)
            ranked_ids = list(range(min(config.RERANK_TOP_N, len(candidates))))

        seen = set()
        reranked = []
        for idx in ranked_ids:
            if isinstance(idx, int) and 0 <= idx < len(candidates) and idx not in seen:
                reranked.append(candidates[idx])
                seen.add(idx)
            if len(reranked) >= config.RERANK_TOP_N:
                break

        for i, c in enumerate(candidates):
            if len(reranked) >= config.RERANK_TOP_N:
                break
            if i not in seen:
                reranked.append(c)
                seen.add(i)

        return reranked[: config.RERANK_TOP_N]

    def compose_answer(self, question: str, chunks: list[Chunk]) -> str:
        context = "\n\n".join(
            f"--- Source: {c.source} ---\n{c.text}" for c in chunks
        )
        prompt = COMPOSE_USER.format(question=question, context=context)
        resp = self.groq_client.chat.completions.create(
            model=config.GROQ_CHAT_MODEL,
            messages=[
                {"role": "system", "content": COMPOSE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        answer = resp.choices[0].message.content.strip()
        if len(answer) > 4000:
            answer = answer[:3997] + "..."
        return answer

    def pipeline(self, question: str) -> str:
        category, sub_category = self.classify_intent(question)
        if config.RAG_DEBUG:
            logger.info("Classified: %s / %s", category, sub_category)

        candidates = self.retrieve(question, category, sub_category)
        if config.RAG_DEBUG:
            for c in candidates[:3]:
                logger.info("Retrieved [%s/%s] score=%.3f: %s", c.category, c.sub_category, c.score, c.source)

        top_chunks = self.rerank(question, candidates)
        return self.compose_answer(question, top_chunks)


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    store = RAGStore()
    q = sys.argv[1] if len(sys.argv) > 1 else "I forgot my password"
    print(store.pipeline(q))
