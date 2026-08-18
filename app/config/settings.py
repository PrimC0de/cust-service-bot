"""Environment-backed application settings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")


def _int_set(value: str) -> frozenset[int]:
    return frozenset(int(item.strip()) for item in value.split(",") if item.strip())


@dataclass(frozen=True)
class Settings:
    root_dir: Path = ROOT_DIR
    telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_admin_ids: frozenset[int] = _int_set(os.getenv("TELEGRAM_ADMIN_IDS", ""))

    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    initial_profile: str = os.getenv("ACTIVE_MODEL_PROFILE", "openai")

    embedding_model: str = "text-embedding-3-small"
    retrieval_top_k: int = 4
    embedding_batch_size: int = 100

    debounce_seconds: float = 3.0
    history_exchanges: int = 4
    idle_ttl_seconds: float = 86_400
    cleanup_interval_seconds: float = 3_600
    max_concurrent_pipelines: int = 20
    concurrent_updates: int = 32
    delivery_attempts: int = 3

    chunk_size: int = 500
    chunk_overlap: int = 50

    @property
    def knowledge_dir(self) -> Path:
        return self.root_dir / "data" / "raw" / "knowledge"

    @property
    def evaluation_dir(self) -> Path:
        return self.root_dir / "data" / "evaluation"

    @property
    def retrieval_evaluation_path(self) -> Path:
        return self.evaluation_dir / "retrieval_cases.json"

    @property
    def taxonomy_path(self) -> Path:
        return self.root_dir / "data" / "taxonomy.json"

    @property
    def indexes_dir(self) -> Path:
        return self.root_dir / "data" / "indexes"

    @property
    def index_path(self) -> Path:
        return self.indexes_dir / "index.faiss"

    @property
    def chunks_path(self) -> Path:
        return self.indexes_dir / "chunks.json"

    @property
    def manifest_path(self) -> Path:
        return self.indexes_dir / "manifest.json"
