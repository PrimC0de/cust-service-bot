# Context-aware Telegram RAG bot

A single-process Telegram DM bot with in-memory conversational context and calibrated dense retrieval. Hosted embedding and language-model calls use the direct OpenAI-compatible API. Answers may use facts explicitly provided by the user in recent conversation, while company and platform claims must come from retrieved knowledge chunks.

The current deployment intentionally has no database, Redis, API server, health endpoint, or local ML model.

## Setup

Requires Python 3.11 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set the Telegram token, admin IDs, and `OPENAI_API_KEY` in `.env`. OpenAI serves `text-embedding-3-small` embeddings. A compatible calibrated FAISS index is required at startup.

## Model profiles

`ACTIVE_MODEL_PROFILE` accepts one preconfigured name:

- `openai`: GPT-5.6 Luna for reformulation and composition through the direct OpenAI-compatible API.

Model names and routing behavior are fixed in code. Telegram users listed in `TELEGRAM_ADMIN_IDS` may inspect the active profile with `/model`. A batch keeps the selected profile snapshot when its debounce completes.

## Build retrieval artifacts

Place intent playbooks under `data/raw/intent-knowledge/`. The previous files under `data/raw/knowledge/` remain available but are not indexed by this command.

```bash
python -m scripts.ingest --rebuild
```

Ingestion embeds each row in `atomic-utterance-examples.txt` independently. Intent catalog, behavior rules, and ambiguity records are attached as grounded context but are not embedded. Files under `original-source/` are retained only for traceability. Ingestion writes the normalized FAISS index under `data/indexes/`, calibrates the confidence cutoff from `data/evaluation/retrieval_cases.json`, and installs artifacts only after calibration succeeds.

Rebuild whenever the chunking schema, embedding model, KB, or retrieval evaluation changes. There is no lexical-only fallback.

## Run

```bash
python -m app.main
```

The bot uses Telegram polling and accepts private chats only. Messages arriving within three seconds of the latest bubble are combined into one user turn. Conversation state contains the latest four user/assistant exchanges, expires after 24 hours of inactivity, and is erased on restart.

## Retrieval behavior

The current batch alone is embedded for the first global search:

- A top score at or above the calibrated cutoff goes directly to grounded composition.
- Composition receives the current message, four recent exchanges, and retrieved chunks. It may use explicit user statements for conversational facts and relevant KB chunks for company claims.
- A weak score gets one history-aware resolution step. It may answer directly from explicit user statements or produce a standalone reformulated query for one more dense search.
- A second weak result produces one focused clarification question.
- A clarification reply is searched alone first; when weak, the unresolved query and reply are resolved together for one final search.
- Remaining weak evidence produces fixed insufficient-information text.
- Missing retrieval or exhausted OpenAI retries produce fixed service-unavailable text.
- Provider retries repeat only the failed API call. Telegram delivery retries reuse the completed reply.

Prior assistant replies and the model's own knowledge are never treated as evidence.

## Request flow

```mermaid
flowchart TD
    A["Telegram DM bubbles"] --> B["3-second per-user debounce"]
    B --> C["Current-message dense search (top 4)"]
    C --> D{"Score meets calibrated cutoff?"}
    D -- Yes --> E["Compose from user context and relevant KB evidence"]
    E --> F{"Answer or conflicting evidence?"}
    F -- Answer --> G["Deliver cached reply"]
    F -- Conflict --> H["Ask one clarification"]
    D -- No --> J["Resolve once with recent history"]
    J --> I{"Explicit user context answers it?"}
    I -- Yes --> G
    I -- No --> K["Search the standalone resolved query"]
    K --> L{"Score meets cutoff?"}
    L -- Yes --> E
    L -- No, first attempt --> H
    L -- No, clarification already used --> P["Fixed insufficient-information reply"]
    C -. Retrieval unavailable .-> Q["Fixed service-unavailable reply"]
    J -. Providers exhausted .-> Q
    E -. Providers exhausted .-> Q
```

## Tests

```bash
python -m unittest discover -s tests -v
```

Retrieval calibration and other evaluation fixtures remain under `data/evaluation/`. Generated retrieval artifacts and `.env` are ignored by Git.
