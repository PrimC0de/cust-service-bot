"""Bounded retries for OpenAI-compatible chat providers."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence

from openai import APIConnectionError, APIStatusError, APITimeoutError, InternalServerError, RateLimitError

from app.models import ModelProfileSnapshot
from app.providers.client import ClientRegistry

logger = logging.getLogger(__name__)


class ProviderFailure(RuntimeError):
    pass


def is_transient_error(exc: Exception) -> bool:
    if isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError)):
        return True
    return isinstance(exc, APIStatusError) and exc.status_code >= 500


class ProviderRouter:
    def __init__(self, clients: ClientRegistry):
        self.clients = clients

    async def text(
        self,
        profile: ModelProfileSnapshot,
        model: str,
        messages: Sequence[dict[str, str]],
        *,
        batch_id: str,
        stage: str,
        json_mode: bool = False,
    ) -> str:
        try:
            client = self.clients.for_profile(profile)
        except Exception as exc:
            logger.warning(
                "provider_call_failed batch=%s profile=%s stage=%s attempt=0 error=%s",
                batch_id,
                profile.name,
                stage,
                type(exc).__name__,
            )
            raise ProviderFailure(f"{profile.name}/{stage} unavailable") from exc
        last_error: Exception | None = None

        for attempt in range(1, profile.attempts + 1):
            try:
                kwargs = {
                    "model": model,
                    "messages": list(messages),
                }
                if json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                response = await client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content
                if not content or not content.strip():
                    raise ValueError("Provider returned empty content")
                return content.strip()
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "provider_call_failed batch=%s profile=%s stage=%s attempt=%d error=%s",
                    batch_id,
                    profile.name,
                    stage,
                    attempt,
                    type(exc).__name__,
                )
                if not is_transient_error(exc) or attempt == profile.attempts:
                    break

        raise ProviderFailure(f"{profile.name}/{stage} failed") from last_error

    async def json(
        self,
        profile: ModelProfileSnapshot,
        model: str,
        messages: Sequence[dict[str, str]],
        *,
        batch_id: str,
        stage: str,
    ) -> dict:
        content = await self.text(
            profile,
            model,
            messages,
            batch_id=batch_id,
            stage=stage,
            json_mode=True,
        )
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise ProviderFailure(f"{profile.name}/{stage} returned invalid JSON") from exc
