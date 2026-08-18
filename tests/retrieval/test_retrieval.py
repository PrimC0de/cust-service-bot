import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import faiss
import numpy as np

from app.models import KnowledgeChunk
from app.rag.retrieval.dense import DenseRetrievalError, DenseRetriever


def chunk(identifier, text):
    return KnowledgeChunk(
        identifier,
        text,
        text,
        "category",
        "subcategory",
        f"source-{identifier}.txt",
        "Title",
        ("Section",),
    )


class FakeEmbeddings:
    def __init__(self, failures=0):
        self.failures = failures
        self.calls = 0

    async def create(self, **_kwargs):
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError("transient")
        return SimpleNamespace(data=[SimpleNamespace(embedding=[1.0, 0.0])])


class DenseTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.index_path = root / "index.faiss"
        self.manifest_path = root / "manifest.json"
        index = faiss.IndexFlatIP(2)
        index.add(np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
        faiss.write_index(index, str(self.index_path))
        self.manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "chunking_method": "structure-aware-v1",
                    "embedding_provider": "openai",
                    "embedding_model": "model",
                    "chunk_count": 2,
                    "dimensions": 2,
                    "dense_available": True,
                    "confidence_threshold": 0.7,
                }
            ),
            encoding="utf-8",
        )
        self.chunks = [chunk(0, "first"), chunk(1, "second")]

    def tearDown(self):
        self.temp.cleanup()

    async def test_dense_global_search_retries_embedding_once(self):
        embeddings = FakeEmbeddings(failures=1)
        dense = DenseRetriever(
            SimpleNamespace(embeddings=embeddings),
            "model",
            self.chunks,
            self.index_path,
            self.manifest_path,
        )
        with patch("app.rag.retrieval.dense.is_transient_error", return_value=True):
            results = await dense.search("query", 2, batch_id="batch")
        self.assertEqual(embeddings.calls, 2)
        self.assertEqual(results[0].chunk.chunk_id, 0)
        self.assertEqual(dense.confidence_threshold, 0.7)

    async def test_embedding_exhaustion_raises_unavailable(self):
        embeddings = FakeEmbeddings(failures=2)
        dense = DenseRetriever(
            SimpleNamespace(embeddings=embeddings),
            "model",
            self.chunks,
            self.index_path,
            self.manifest_path,
        )
        with patch("app.rag.retrieval.dense.is_transient_error", return_value=True):
            with self.assertRaises(DenseRetrievalError):
                await dense.search("query", 2, batch_id="batch")
        self.assertEqual(embeddings.calls, 2)

    async def test_manifest_without_calibration_is_unavailable(self):
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        del manifest["confidence_threshold"]
        self.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        dense = DenseRetriever(
            SimpleNamespace(embeddings=FakeEmbeddings()),
            "model",
            self.chunks,
            self.index_path,
            self.manifest_path,
        )
        self.assertFalse(dense.available)
        with self.assertRaises(DenseRetrievalError):
            await dense.search("query", batch_id="batch")
