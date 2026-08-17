import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.models import ModelProfileSnapshot
from app.providers.router import ProviderFailure, ProviderRouter


PROFILE = ModelProfileSnapshot(
    "provider", "url", "OPENAI_API_KEY", "rerank", "rewrite", "compose", 3
)


class FakeCompletions:
    def __init__(self, failures):
        self.failures = failures
        self.calls = 0

    async def create(self, **_kwargs):
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError("provider failure")
        message = SimpleNamespace(content="answer")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class ProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_transient_errors_use_profile_attempt_count(self):
        completions = FakeCompletions(failures=2)
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        clients = SimpleNamespace(for_profile=lambda _profile: client)
        router = ProviderRouter(clients)
        with patch("app.providers.router.is_transient_error", return_value=True):
            result = await router.text(
                PROFILE,
                "model",
                [{"role": "user", "content": "question"}],
                batch_id="batch",
                stage="compose",
            )
        self.assertEqual(result, "answer")
        self.assertEqual(completions.calls, 3)

    async def test_non_transient_failure_is_not_retried(self):
        completions = FakeCompletions(failures=3)
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        clients = SimpleNamespace(for_profile=lambda _profile: client)
        router = ProviderRouter(clients)
        with patch("app.providers.router.is_transient_error", return_value=False):
            with self.assertRaises(ProviderFailure):
                await router.text(
                    PROFILE,
                    "model",
                    [{"role": "user", "content": "question"}],
                    batch_id="batch",
                    stage="compose",
                )
        self.assertEqual(completions.calls, 1)

    async def test_missing_primary_client_becomes_failover_eligible(self):
        def missing(_profile):
            raise RuntimeError("missing credential")

        router = ProviderRouter(SimpleNamespace(for_profile=missing))
        with self.assertRaises(ProviderFailure):
            await router.text(
                PROFILE,
                "model",
                [{"role": "user", "content": "question"}],
                batch_id="batch",
                stage="rerank",
            )
