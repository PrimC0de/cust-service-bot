import json
import tempfile
import unittest
from pathlib import Path

from app.rag.ingestion.chunker import CHUNKING_METHOD, chunk_documents, merge_small_sections
from app.rag.ingestion.indexer import SCHEMA_VERSION, build_indexes, manifest_compatible
from app.rag.ingestion.parser import parse_documents


class IngestionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.knowledge = self.root / "data" / "raw" / "knowledge"
        folder = self.knowledge / "account"
        folder.mkdir(parents=True)
        (folder / "login.txt").write_text(
            "Login Help\n\n## Password\nReset it here.\n\n## SMS\nUse the code.",
            encoding="utf-8",
        )
        taxonomy = {
            "categories": [{"slug": "account", "sub_intents": [{"slug": "login"}]}]
        }
        self.taxonomy = self.root / "data" / "taxonomy.json"
        self.taxonomy.parent.mkdir(exist_ok=True)
        self.taxonomy.write_text(json.dumps(taxonomy), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def test_heading_parsing_merging_splitting_and_metadata(self):
        documents = parse_documents(self.knowledge, self.taxonomy)
        self.assertEqual(documents[0].title, "Login Help")
        self.assertEqual(documents[0].sections[0].path, ("Password",))
        merged = merge_small_sections(documents[0].sections, 500)
        self.assertEqual(len(merged), 1)

        long_text = "Sentence. " * 100
        (self.knowledge / "account" / "login.txt").write_text(
            f"Login Help\n\n## Large\n{long_text}", encoding="utf-8"
        )
        chunks = chunk_documents(parse_documents(self.knowledge, self.taxonomy))
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk.text) <= 500 for chunk in chunks))
        self.assertTrue(chunks[0].embedding_text.startswith("Login Help\nLarge\n"))
        self.assertEqual(chunks[0].category, "account")

    async def test_bm25_only_manifest_is_compatible(self):
        documents = parse_documents(self.knowledge, self.taxonomy)
        chunks = chunk_documents(documents)
        manifest = await build_indexes(
            chunks,
            indexes_dir=self.root / "indexes",
            embedding_client=None,
            embedding_model="text-embedding-3-small",
        )
        self.assertEqual(manifest["schema_version"], SCHEMA_VERSION)
        self.assertEqual(manifest["chunking_method"], CHUNKING_METHOD)
        self.assertFalse(manifest["dense_available"])
        self.assertTrue(manifest_compatible(manifest, "text-embedding-3-small"))
        self.assertTrue((self.root / "indexes" / "chunks.json").exists())

