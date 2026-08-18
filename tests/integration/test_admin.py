import unittest
from types import SimpleNamespace

from app.bot.admin import ADMIN_IDS_KEY, SELECTOR_KEY, model_command
from app.config.profiles import ProfileSelector
from app.models import ModelProfileSnapshot


def profile(name):
    return ModelProfileSnapshot(name, "f", "c", 1)


class FakeMessage:
    def __init__(self):
        self.replies = []

    async def reply_text(self, text):
        self.replies.append(text)


class AdminTests(unittest.IsolatedAsyncioTestCase):
    async def test_allowlist_and_named_validation(self):
        selector = ProfileSelector({"openai": profile("openai")}, "openai")
        data = {ADMIN_IDS_KEY: frozenset({7}), SELECTOR_KEY: selector}

        denied = FakeMessage()
        update = SimpleNamespace(effective_user=SimpleNamespace(id=8), effective_message=denied)
        context = SimpleNamespace(application=SimpleNamespace(bot_data=data), args=["openai"])
        await model_command(update, context)
        self.assertEqual(selector.active_name, "openai")

        invalid = FakeMessage()
        update = SimpleNamespace(effective_user=SimpleNamespace(id=7), effective_message=invalid)
        context.args = ["arbitrary-model"]
        await model_command(update, context)
        self.assertEqual(selector.active_name, "openai")

        context.args = ["openai"]
        await model_command(update, context)
        self.assertEqual(selector.active_name, "openai")
