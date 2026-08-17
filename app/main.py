"""Application bootstrap for Telegram polling."""

from __future__ import annotations

import asyncio
import logging

from telegram.ext import Application, CommandHandler, MessageHandler, filters

from app.bot.admin import ADMIN_IDS_KEY, SELECTOR_KEY, model_command
from app.bot.conversation import ConversationStore
from app.bot.telegram import (
    PIPELINE_KEY,
    PIPELINE_SEMAPHORE_KEY,
    STORE_KEY,
    error_handler,
    start_command,
    text_message,
)
from app.config.profiles import ProfileSelector, build_profiles
from app.config.settings import Settings
from app.providers.client import ClientRegistry
from app.providers.router import ProviderRouter
from app.rag.generation import GenerationService
from app.rag.pipeline import RAGPipeline
from app.rag.retrieval.bm25 import BM25Retriever
from app.rag.retrieval.dense import DenseRetriever, load_chunks
from app.rag.retrieval.hybrid import HybridRetriever


logger = logging.getLogger(__name__)
CLEANUP_TASK_KEY = "conversation_cleanup_task"


def build_application(settings: Settings | None = None) -> Application:
    settings = settings or Settings()
    if not settings.telegram_bot_token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is required")
    if not settings.chunks_path.exists():
        raise RuntimeError("No chunks index found. Run: python -m scripts.ingest --rebuild")

    chunks = load_chunks(settings.chunks_path)
    if not chunks:
        raise RuntimeError("The chunks index is empty; no valid retrieval source exists")

    profiles = build_profiles(settings)
    selector = ProfileSelector(profiles, settings.initial_profile)
    clients = ClientRegistry(settings, profiles)
    try:
        embedding_client = clients.embeddings()
    except RuntimeError:
        embedding_client = None
        logger.warning("OpenAI embeddings unavailable at startup; using BM25 only")

    dense = DenseRetriever(
        embedding_client,
        settings.embedding_model,
        chunks,
        settings.index_path,
        settings.manifest_path,
    )
    bm25 = BM25Retriever(chunks)
    if not dense.available and not bm25.available:
        raise RuntimeError("No valid dense or BM25 retrieval source exists")

    retriever = HybridRetriever(chunks, dense, bm25, top_k=settings.retrieval_top_k)
    generation = GenerationService(ProviderRouter(clients), top_n=settings.rerank_top_n)
    pipeline = RAGPipeline(retriever, generation, profiles)
    store = ConversationStore(
        debounce_seconds=settings.debounce_seconds,
        history_exchanges=settings.history_exchanges,
        idle_ttl_seconds=settings.idle_ttl_seconds,
        cleanup_interval_seconds=settings.cleanup_interval_seconds,
        dispatch_attempts=settings.debounce_attempts,
        profile_snapshot=selector.snapshot,
    )

    async def post_init(application: Application) -> None:
        application.bot_data[CLEANUP_TASK_KEY] = asyncio.create_task(
            store.cleanup_loop(), name="conversation-cleanup"
        )

    async def post_shutdown(application: Application) -> None:
        task = application.bot_data.get(CLEANUP_TASK_KEY)
        if task:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await store.close()

    application = (
        Application.builder()
        .token(settings.telegram_bot_token)
        .concurrent_updates(settings.concurrent_updates)
        .post_init(post_init)
        .post_shutdown(post_shutdown)
        .build()
    )
    application.bot_data.update(
        {
            STORE_KEY: store,
            PIPELINE_KEY: pipeline,
            PIPELINE_SEMAPHORE_KEY: asyncio.Semaphore(settings.max_concurrent_pipelines),
            SELECTOR_KEY: selector,
            ADMIN_IDS_KEY: settings.telegram_admin_ids,
        }
    )
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("model", model_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_message))
    application.add_error_handler(error_handler)
    return application


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    build_application().run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
