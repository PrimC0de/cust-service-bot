import unittest

from app.config.profiles import build_profiles
from app.config.settings import Settings
from app.providers.client import ClientRegistry


class OpenAIRoutingTests(unittest.TestCase):
    def test_luna_and_embeddings_share_direct_openai_client(self):
        profiles = build_profiles()
        settings = Settings(openai_api_key="test-key")
        clients = ClientRegistry(settings)
        client = clients.for_profile(profiles["openai"])

        self.assertEqual(set(profiles), {"openai"})
        self.assertEqual(profiles["openai"].reformulate_model, "gpt-5.6-luna")
        self.assertEqual(profiles["openai"].compose_model, "gpt-5.6-luna")
        self.assertIsNone(profiles["openai"].backup_profile)
        self.assertIs(clients.embeddings(), client)
        self.assertEqual(str(client.base_url), "https://api.openai.com/v1/")
        self.assertEqual(settings.embedding_model, "text-embedding-3-small")
