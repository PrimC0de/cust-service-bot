import unittest

from app.config.profiles import build_profiles
from app.config.settings import Settings
from app.providers.client import ClientRegistry


class OpenRouterRoutingTests(unittest.TestCase):
    def test_every_hosted_profile_and_embedding_share_openrouter_client(self):
        profiles = build_profiles()
        settings = Settings(
            openrouter_api_key="test-key",
            openrouter_base_url="https://openrouter.example/v1",
        )
        clients = ClientRegistry(settings)
        profile_clients = [clients.for_profile(profile) for profile in profiles.values()]

        self.assertTrue(all(client is profile_clients[0] for client in profile_clients))
        self.assertIs(clients.embeddings(), profile_clients[0])
        self.assertTrue(
            all(
                model.startswith(("moonshotai/", "openai/", "google/"))
                for profile in profiles.values()
                for model in (profile.reformulate_model, profile.compose_model)
            )
        )
        self.assertEqual(settings.embedding_model, "openai/text-embedding-3-small")
