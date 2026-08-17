# Context-aware Telegram RAG bot

A single-process Telegram DM bot with in-memory conversational context and evidence-first hybrid retrieval. It combines a hosted OpenAI dense index with local multilingual BM25, always reranks retrieved evidence, and composes answers only from selected knowledge chunks.

The current deployment intentionally has no database, Redis, API server, health endpoint, or local ML model.

## Setup

Requires Python 3.11 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set the Telegram token, admin IDs, and provider credentials in `.env`. OpenAI is also used for hosted `text-embedding-3-small` embeddings. If it is unavailable at bot startup but valid chunks exist, retrieval starts in BM25-only mode.

## Model profiles

`ACTIVE_MODEL_PROFILE` accepts one of three preconfigured names:

- `openrouter`: development and provider experimentation through OpenRouter.
- `kimi`: direct Kimi reranking/reformulation and composition, with OpenAI failover.
- `reranking-reformulation`: direct OpenAI reranking/reformulation and composition.

Model names and routing behavior are fixed in code. Telegram users listed in `TELEGRAM_ADMIN_IDS` may inspect or switch the live profile with `/model` and `/model <profile>`. A batch keeps the profile selected when its debounce completes.

## Build retrieval artifacts

Place approved `.txt` or `.md` documents under `data/raw/knowledge/<category>/`. Every category/document filename pair must exist in `data/taxonomy.json`.

```bash
python -m scripts.ingest --rebuild
```

Ingestion parses document structure, merges small related sections, splits large sections at 500 characters with 50-character overlap, and writes ordered metadata plus a manifest under `data/indexes/`. With `OPENAI_API_KEY`, it also creates a normalized FAISS index. Rebuild whenever the chunking schema or embedding model changes.

## Run

```bash
python -m app.main
```

The bot uses Telegram polling and accepts private chats only. Messages arriving within three seconds of the latest bubble are combined into one user turn. Conversation state contains the latest four user/assistant exchanges, expires after 24 hours of inactivity, and is erased on restart.

## Retrieval behavior

The current batch and immediately preceding exchange form the initial global retrieval query. Dense and BM25 candidates are merged and reranked by the active provider:

- Strong KB evidence is composed into a grounded answer.
- Ambiguous intent or conflicting strong evidence produces a clarification question.
- Weak evidence gets one history-aware reformulation/translation and a second hybrid retrieval.
- Evidence that remains weak produces a fixed insufficient-information response.
- Provider exhaustion produces a fixed service-unavailable response.

The model is never allowed to silently replace missing KB evidence with its own knowledge.

## Tests

```bash
python -m unittest discover -s tests -v
```

Evaluation fixtures remain under `data/evaluation/`. Generated retrieval artifacts and `.env` are ignored by Git.
