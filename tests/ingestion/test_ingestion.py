import json
import tempfile
import unittest
from pathlib import Path
import numpy as np

from app.rag.ingestion.chunker import (
    CHUNKING_METHOD,
    chunk_documents,
    chunk_intent_documents,
    load_atomic_intent_chunks,
    merge_small_sections,
)
from app.rag.ingestion.indexer import (
    SCHEMA_VERSION,
    build_indexes,
    calibrate_confidence,
    manifest_compatible,
)
from app.rag.ingestion.parser import parse_documents


class IngestionTests(unittest.TestCase):
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

    def test_dense_manifest_contains_safe_calibration(self):
        documents = parse_documents(self.knowledge, self.taxonomy)
        chunks = chunk_documents(documents)

        cases = [
            {"query": "How do I reset a password?", "expected_source": chunks[0].source},
            {"query": "What is the weather?", "expected_source": None},
        ]
        manifest = build_indexes(
            chunks,
            indexes_dir=self.root / "indexes",
            embedding_model="text-embedding-3-small",
            evaluation_cases=cases,
            vectors=np.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        )
        self.assertEqual(manifest["schema_version"], SCHEMA_VERSION)
        self.assertEqual(manifest["chunking_method"], CHUNKING_METHOD)
        self.assertEqual(manifest["embedding_provider"], "openai")
        self.assertTrue(manifest["dense_available"])
        self.assertEqual(manifest["confidence_threshold"], 1.0)
        self.assertTrue(manifest_compatible(manifest, "text-embedding-3-small"))
        self.assertTrue((self.root / "indexes" / "chunks.json").exists())
        self.assertTrue((self.root / "indexes" / "index.faiss").exists())

    def test_intent_playbook_is_one_retrieval_unit(self):
        intent_dir = self.root / "data" / "raw" / "intent-knowledge"
        intent_dir.mkdir(parents=True)
        (intent_dir / "top-up-deposit.txt").write_text(
            "Top Up Intent\n\n## Retrieval Profile\nClassification cues: topup; depo\n\n"
            "## Behavior Rules\nAsk for an amount.",
            encoding="utf-8",
        )
        chunks = chunk_intent_documents(parse_documents(intent_dir))
        self.assertEqual(len(chunks), 1)
        self.assertIn("topup\ndepo", chunks[0].embedding_text)
        self.assertNotIn("Ask for an amount", chunks[0].embedding_text)
        self.assertIn("Ask for an amount", chunks[0].text)

    def test_layered_intent_kb_loads_only_nonempty_atomic_examples(self):
        intent_dir = self.root / "data" / "raw" / "intent-knowledge"
        source_dir = intent_dir / "original-source"
        source_dir.mkdir(parents=True)
        (source_dir / "top-up-deposit.txt").write_text("original", encoding="utf-8")
        (intent_dir / "intent-catalog.txt").write_text(
            "Intent Catalog\n\n## INT_TOPUP\nintent_name: Top Up",
            encoding="utf-8",
        )
        (intent_dir / "behavior-rules.txt").write_text(
            "RULE_TOPUP_001 | INT_TOPUP | amount missing | ask amount | do not infer | amount | context | none | P0",
            encoding="utf-8",
        )
        (intent_dir / "ambiguity-map.txt").write_text(
            "AMB_001 | INT_TOPUP | INT_PAYMENT_DETAILS | rek mana | amount | history | unclear | note",
            encoding="utf-8",
        )
        (intent_dir / "atomic-utterance-examples.txt").write_text(
            "EX_TOPUP_001 | depo | INT_TOPUP | Top Up | id | slang | original-source/top-up-deposit.txt | false | direct",
            encoding="utf-8",
        )
        chunks = load_atomic_intent_chunks(intent_dir)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].embedding_text, "depo")
        self.assertIn("RULE_TOPUP_001", chunks[0].text)
        self.assertEqual(
            chunks[0].source,
            "data/raw/intent-knowledge/original-source/top-up-deposit.txt",
        )

    def test_overlapping_scores_fail_calibration(self):
        chunks = chunk_documents(parse_documents(self.knowledge, self.taxonomy))
        chunk_vectors = np.asarray([[1.0, 0.0]], dtype=np.float32)
        cases = [
            {"query": "supported", "expected_source": chunks[0].source},
            {"query": "unsupported", "expected_source": None},
        ]
        vectors = np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "scores overlap"):
            calibrate_confidence(chunk_vectors, chunks, vectors, cases)
