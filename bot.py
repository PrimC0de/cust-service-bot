"""Telegram bot entrypoint — polling mode."""

import logging

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

import config
from rag import RAGStore

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

FALLBACK_REPLY = (
    "Sorry, something went wrong on our end. "
    "Please try again in a moment, or contact customer service during 09:00–02:00."
)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Hi! I'm here to help with deposits, withdrawals, account issues, "
        "promotions, and more. Just send me your question."
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return

    question = update.message.text.strip()
    if not question:
        return

    await update.message.chat.send_action(ChatAction.TYPING)

    store: RAGStore = context.bot_data["rag_store"]
    try:
        answer = store.pipeline(question)
        await update.message.reply_text(answer)
    except Exception:
        logger.exception("Pipeline error for question: %s", question[:100])
        await update.message.reply_text(FALLBACK_REPLY)


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.exception("Unhandled error", exc_info=context.error)
    if isinstance(update, Update) and update.message:
        try:
            await update.message.reply_text(FALLBACK_REPLY)
        except Exception:
            logger.exception("Failed to send fallback reply")


def main() -> None:
    rag_store = RAGStore()

    app = (
        Application.builder()
        .token(config.TELEGRAM_BOT_TOKEN)
        .build()
    )
    app.bot_data["rag_store"] = rag_store

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_error_handler(error_handler)

    logger.info("Bot starting (polling)...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
