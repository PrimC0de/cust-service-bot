"""OpenAI-compatible client registry."""

from __future__ import annotations

from openai import AsyncOpenAI

from app.config.settings import Settings
from app.models import ModelProfileSnapshot


class ClientRegistry:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._client: AsyncOpenAI | None = None

    def for_profile(self, profile: ModelProfileSnapshot) -> AsyncOpenAI:
        return self._get(f"profile {profile.name}")

    def embeddings(self) -> AsyncOpenAI:
        return self._get("embeddings")

    def _get(self, purpose: str) -> AsyncOpenAI:
        if not self._settings.openrouter_api_key:
            raise RuntimeError(f"Missing OPENROUTER_API_KEY for {purpose}")
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self._settings.openrouter_api_key,
                base_url=self._settings.openrouter_base_url,
                max_retries=0,
            )
        return self._client
