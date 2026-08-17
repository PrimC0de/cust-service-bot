"""Telegram update handlers."""

from __future__ import annotations

import asyncio
import logging

from telegram import Update
from telegram.constants import ChatAction, ChatType
from telegram.ext import ContextTypes

from app.bot.conversation import ConversationStore
from app.models import PendingBatch
from app.rag.generation import service_unavailable_response
from app.rag.pipeline import RAGPipeline


logger = logging.getLogger(__name__)

STORE_KEY = "conversation_store"
PIPELINE_KEY = "rag_pipeline"
PIPELINE_SEMAPHORE_KEY = "pipeline_semaphore"


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_message:
        await update.effective_message.reply_text("Send me a question about the knowledge base.")


async def text_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    user = update.effective_user
    chat = update.effective_chat
    if message is None or user is None or chat is None or not message.text:
        return
    if chat.type != ChatType.PRIVATE:
        return

    store: ConversationStore = context.application.bot_data[STORE_KEY]
    pipeline: RAGPipeline = context.application.bot_data[PIPELINE_KEY]
    semaphore: asyncio.Semaphore = context.application.bot_data[PIPELINE_SEMAPHORE_KEY]

    await context.bot.send_chat_action(chat_id=chat.id, action=ChatAction.TYPING)

    async def process(batch: PendingBatch, history, unresolved):
        if batch.profile is None:
            raise RuntimeError("Batch has no model profile snapshot")
        async with semaphore:
            return await pipeline.run(
                batch.text,
                history,
                batch.profile,
                batch_id=batch.batch_id,
                unresolved=unresolved,
            )

    async def deliver(text: str) -> None:
        await context.bot.send_message(chat_id=chat.id, text=text)

    await store.submit(
        user_id=user.id,
        chat_id=chat.id,
        message_id=message.message_id,
        text=message.text,
        process=process,
        deliver=deliver,
        unavailable=service_unavailable_response,
    )


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("stage=telegram_update error_type=%s", type(context.error).__name__)
