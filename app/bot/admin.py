"""Administrative Telegram commands."""

from __future__ import annotations

from telegram import Update
from telegram.ext import ContextTypes

from app.config.profiles import ProfileSelector


SELECTOR_KEY = "profile_selector"
ADMIN_IDS_KEY = "telegram_admin_ids"


async def model_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    message = update.effective_message
    if user is None or message is None:
        return

    admin_ids: frozenset[int] = context.application.bot_data[ADMIN_IDS_KEY]
    if user.id not in admin_ids:
        await message.reply_text("This command is restricted to administrators.")
        return

    selector: ProfileSelector = context.application.bot_data[SELECTOR_KEY]
    if not context.args:
        names = ", ".join(selector.profiles)
        await message.reply_text(f"Active profile: {selector.active_name}\nAvailable: {names}")
        return

    name = context.args[0]
    try:
        selector.select(name)
    except ValueError:
        names = ", ".join(selector.profiles)
        await message.reply_text(f"Unknown profile. Available: {names}")
        return
    await message.reply_text(f"Model profile changed to: {name}")

