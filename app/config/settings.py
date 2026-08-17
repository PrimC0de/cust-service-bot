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
    kimi_api_key: str = os.getenv("KIMI_API_KEY", "")
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")

    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    kimi_base_url: str = os.getenv("KIMI_BASE_URL", "https://api.moonshot.ai/v1")
    openrouter_base_url: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    initial_profile: str = os.getenv("ACTIVE_MODEL_PROFILE", "openrouter")

    embedding_model: str = "text-embedding-3-small"
    retrieval_top_k: int = 10
    rerank_top_n: int = 3
    embedding_batch_size: int = 100

    debounce_seconds: float = 3.0
    history_exchanges: int = 4
    idle_ttl_seconds: float = 86_400
    cleanup_interval_seconds: float = 3_600
    max_concurrent_pipelines: int = 20
    concurrent_updates: int = 32
    debounce_attempts: int = 3

    chunk_size: int = 500
    chunk_overlap: int = 50

    @property
    def knowledge_dir(self) -> Path:
        return self.root_dir / "data" / "raw" / "knowledge"

    @property
    def evaluation_dir(self) -> Path:
        return self.root_dir / "data" / "evaluation"

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

    def api_key(self, env_name: str) -> str:
        return {
            "OPENAI_API_KEY": self.openai_api_key,
            "KIMI_API_KEY": self.kimi_api_key,
            "OPENROUTER_API_KEY": self.openrouter_api_key,
        }[env_name]
