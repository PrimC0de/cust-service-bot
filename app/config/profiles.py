"""Immutable named model profiles and live profile selection."""

from __future__ import annotations

from app.config.settings import Settings
from app.models import ModelProfileSnapshot


def build_profiles(settings: Settings) -> dict[str, ModelProfileSnapshot]:
    return {
        "openrouter": ModelProfileSnapshot(
            name="openrouter",
            base_url=settings.openrouter_base_url,
            api_key_env="OPENROUTER_API_KEY",
            rerank_model="moonshotai/kimi-k2.6",
            reformulate_model="moonshotai/kimi-k2.6",
            compose_model="moonshotai/kimi-k3",
            attempts=2,
            backup_profile="reranking-reformulation",
        ),
        "kimi": ModelProfileSnapshot(
            name="kimi",
            base_url=settings.kimi_base_url,
            api_key_env="KIMI_API_KEY",
            rerank_model="kimi-k2.6",
            reformulate_model="kimi-k2.6",
            compose_model="kimi-k3",
            attempts=2,
            backup_profile="reranking-reformulation",
        ),
        "reranking-reformulation": ModelProfileSnapshot(
            name="reranking-reformulation",
            base_url=settings.openai_base_url,
            api_key_env="OPENAI_API_KEY",
            rerank_model="gpt-5.6-luna",
            reformulate_model="gpt-5.6-luna",
            compose_model="gpt-5.6-terra",
            attempts=3,
        ),
    }


class ProfileSelector:
    def __init__(self, profiles: dict[str, ModelProfileSnapshot], active: str):
        if active not in profiles:
            raise ValueError(f"Unknown model profile: {active}")
        self.profiles = profiles
        self._active = active

    @property
    def active_name(self) -> str:
        return self._active

    def snapshot(self) -> ModelProfileSnapshot:
        return self.profiles[self._active]

    def select(self, name: str) -> ModelProfileSnapshot:
        if name not in self.profiles:
            raise ValueError(f"Unknown model profile: {name}")
        self._active = name
        return self.profiles[name]
