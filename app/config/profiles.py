"""Immutable named model profiles and live profile selection."""

from __future__ import annotations

from app.models import ModelProfileSnapshot


def build_profiles() -> dict[str, ModelProfileSnapshot]:
    return {
        "openai": ModelProfileSnapshot(
            name="openai",
            reformulate_model="gpt-5.6-luna",
            compose_model="gpt-5.6-luna",
            attempts=2,
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
