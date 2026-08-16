Telegram AI RAG Chatbot (POC)
=============================

A Telegram customer-support bot with intent-scoped RAG:
classify -> scoped FAISS retrieve -> LLM rerank -> compose answer.

Stack: Python, python-telegram-bot (polling), OpenAI, FAISS, local files only.
No Docker, no database.


Setup
-----

1. Create a virtual environment and install dependencies:

   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt

2. Copy .env.example to .env and fill in your keys:

   cp .env.example .env
   # Edit .env:
   #   OPENAI_API_KEY=sk-...
   #   TELEGRAM_BOT_TOKEN=...

   Get a Telegram bot token from @BotFather on Telegram.


Build the knowledge index
-------------------------

Run once (or whenever knowledge/ files change):

   python ingest.py --rebuild

This reads knowledge/, chunks documents, embeds them with OpenAI,
and writes index.faiss + chunks.json.


Run the bot
-----------

   python bot.py

The bot uses long polling — no public URL or HTTPS required.
Send any text message to your bot on Telegram.


Project layout
--------------

knowledge/          Answer docs per category/sub-intent (placeholder for POC)
sampling-knowledge/ Sample customer messages used as classification reference
intents.json        Intent taxonomy + classification hints
ingest.py           Offline index builder
rag.py              RAG pipeline (classify, retrieve, rerank, compose)
bot.py              Telegram polling bot
index.faiss         Generated vector index (run ingest.py)
chunks.json         Generated chunk metadata (run ingest.py)


Test locally without Telegram
-----------------------------

   python rag.py "I forgot my password"
   python rag.py "USDT入金教學"


Manual spot-check questions
---------------------------

Question                                              Expected category
"I forgot my password"                                account-login-password-line
"My USDT deposit hasn't arrived"                      deposit-top-up
"Why is my withdrawal still pending?"                 withdrawal-payout
"How do I claim birthday bonus?"                        promotion-bonus-rewards-vip
"Are withdrawals and store deposits working?"         mixed-transaction-status
"I wagered but daily task still incomplete"           task-wagering-progress
"I don't want this payment card"                      payment-card-application
"What are your customer service hours?"               common-questions-how-to
"How much turnover do I need?"                        common-questions-how-to


Replacing placeholder knowledge
-------------------------------

When real docs are ready, replace files in knowledge/ and rerun:

   python ingest.py --rebuild

No code changes needed.


Debug logging
-------------

Set RAG_DEBUG=1 in .env to log classification and retrieval details.
