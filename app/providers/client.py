"""OpenAI-compatible client registry."""

from __future__ import annotations

from openai import AsyncOpenAI

from app.config.settings import Settings
from app.models import ModelProfileSnapshot


class ClientRegistry:
    def __init__(self, settings: Settings, profiles: dict[str, ModelProfileSnapshot]):
        self._settings = settings
        self._profiles = profiles
        self._clients: dict[str, AsyncOpenAI] = {}

    def for_profile(self, profile: ModelProfileSnapshot) -> AsyncOpenAI:
        client = self._clients.get(profile.name)
        if client is not None:
            return client

        api_key = self._settings.api_key(profile.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing {profile.api_key_env} for profile {profile.name}")
        client = AsyncOpenAI(api_key=api_key, base_url=profile.base_url, max_retries=0)
        self._clients[profile.name] = client
        return client

    def embeddings(self) -> AsyncOpenAI:
        profile = self._profiles["reranking-reformulation"]
        return self.for_profile(profile)
