"""Immutable named model profiles and live profile selection."""

from __future__ import annotations

from app.models import ModelProfileSnapshot


def build_profiles() -> dict[str, ModelProfileSnapshot]:
    return {
        "openrouter": ModelProfileSnapshot(
            name="openrouter",
            reformulate_model="openai/gpt-5-nano",
            compose_model="openai/gpt-5-nano",
            attempts=2,
            backup_profile="kimi",
        ),
        "kimi": ModelProfileSnapshot(
            name="kimi",
            reformulate_model="moonshotai/kimi-k2.6",
            compose_model="moonshotai/kimi-k3",
            attempts=2,
            backup_profile="openrouter",
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
