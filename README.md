# IDX Property AI Agent

A production-style multi-agent AI assistant for real estate, built on the
[OpenClaw](https://github.com/openclaw/openclaw) runtime. The assistant performs
natural-language MLS property search, market analytics, semantic recommendations,
RAG-based knowledge retrieval, and WhatsApp + email communication over the
curated California MLS working tables (~140K records).

> **AI Agentic Engineer Internship — IDX Exchange · Summer 2026 · 12 Weeks**

---

## Overview

This project wires a set of custom OpenClaw **skills** and **agents** over two real
MLS datasets, coordinated by an orchestrator that routes each incoming query to the
right specialized agent and returns a unified response through WhatsApp.

```
User → WhatsApp → OpenClaw Runtime → Orchestrator → [specialized agents] → MySQL → response → User
```

**Tech stack:** OpenClaw · TypeScript · Python · MySQL · OpenAI (embeddings + chat) · WhatsApp · Nodemailer

---

## Data

Two MySQL tables in a local schema (`idx_exchange`):

Three MySQL tables imported into a local schema (`idx_exchange`). Counts below are
the actual curated working tables from the internship FTP `sql/` folder:

| Table | Rows | Role |
|---|---|---|
| `rets_property` | 53,122 active listings (CA), 130+ cryptic `L_*` fields | Live search & discovery |
| `california_sold` | 87,157 sold transactions (CA), 46 fields | Historical comps & analytics |
| `rets_openhouse` | 4,282 open-house events | Open-house lookups (NLP track) |

**Join pattern:** `CAST(rets_property.L_ListingID AS UNSIGNED) = california_sold.ListingKey`,
or match on city + postal code for market-level analysis.

> The MLS data is confidential and is **not** committed to this repository
> (see `.gitignore`). Dumps are downloaded from the internship FTP (see Slack) and
> imported into a local MySQL instance only. A larger raw monthly export
> (`CRMLSListing*` / `CRMLSSold*`, the source behind the marketed "667K+") also
> exists on the FTP but the project uses the curated tables above.

---

## Project Structure

```
idx-property-ai-agent/
├── scripts/       # Data-layer tooling
│   ├── import.sh      # Idempotent import of the three SQL tables
│   └── check_env.py   # Validates .env + DB connectivity + OpenAI key
├── schema/        # SQL DDL (committed; this is code, not data)
│   └── indexes.sql    # High-frequency filter indexes
├── Makefile       # make import | indexes | check | rebuild | db-up | db-down
├── skills/        # Custom OpenClaw skills (one folder per capability) — Week 1+
├── src/           # Query layers, embedding pipelines, agent logic (TS/Python) — Week 1+
├── docs/          # Architecture diagram, schema annotations, design notes — Week 1+
├── .env.example   # Template for required environment variables (no real keys)
└── README.md
```

OpenClaw discovers any skill whose `SKILL.md` lives under a configured workspace
root, so this repo is used as the agent **workspace** — the OpenClaw runtime itself
is installed separately and is not part of this repo.

---

## Setup

### Prerequisites
- Node.js (v20+) and npm
- Python 3.10+
- MySQL (running locally)
- OpenClaw installed (`npm install -g openclaw`, then `openclaw onboard`)
- An OpenAI API key with available billing credit

### 1. Clone
```bash
git clone https://github.com/lwwdsb/idx-property-ai-agent.git
cd idx-property-ai-agent
```

### 2. Configure environment
Copy `.env.example` to `.env` and fill in your values. **Never commit `.env`.**
```bash
cp .env.example .env   # then set DB_PASSWORD, and OPENAI_API_KEY before Week 6
```

### 3. Get the data & import
Download `rets_property.sql`, `california_sold.sql`, `rets_openhouse.sql` from the
internship FTP (credentials are in the Slack channel) into the repo root, then:
```bash
make db-up      # start MySQL (Homebrew)
make import     # idempotent import of all three tables into idx_exchange
make indexes    # add high-frequency filter indexes
make check      # validate env + DB row counts + OpenAI key
```
> The MariaDB dumps use zero-date defaults that MySQL 8/9 rejects by default;
> `import.sh` runs each load with `SET sql_mode=''` to handle this.

### 4. Connect WhatsApp (Week 1, with the OpenClaw skeleton)
```bash
openclaw channels login --channel whatsapp
# Scan the QR via WhatsApp → Settings → Linked Devices
```

---

## Environment Variables

See `.env.example`. Keys:

```
DB_HOST=127.0.0.1
DB_PORT=3306
DB_USER=root
DB_PASSWORD=
DB_NAME=idx_exchange
OPENAI_API_KEY=            # not needed until Week 6 (embeddings)
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_CHAT_MODEL=gpt-4o-mini
WHATSAPP_SESSION_NAME=idx-assistant
```

---

## 12-Week Roadmap

| Week | Module | Status |
|------|--------|--------|
| 0 | Environment, MySQL import, indexes, env validation | ✅ |
| 1 | Skeleton + field dictionary + shared modules (db/config/logger) | ✅ |
| 2 | NL property search (normalize → regex fast-path → LLM slot) | ✅ |
| 3 | MySQL query layer (parameterized, ≤50 cap, FULLTEXT, DTO) | ✅ |
| 4 | Conversational agent (multi-turn, patch-merge, pluggable store) | ✅ |
| 5 | Market analytics over `california_sold` (true median, trend) | ✅ |
| 6 | Hybrid retrieval — dense + BM25 + RRF in Qdrant | ✅ |
| 7 | Recommendation + comp price validation + verifier gate | ✅ |
| 8 | RAG Q&A (grounded + cited, extractive fallback) | ✅ |
| 9 | Orchestrator — skill registry + deterministic router + recipes | ✅ |
| 10 | Warm retrieval service + embedding intent + WhatsApp handler | ✅ |
| 11 | Email agent — draft-then-approve, outbound locked at tool layer | ✅ |
| 12 | Wrap-up: 5th skill wired, full test suite, docs | ✅ |

**Reserved (need a credential/key, degrade gracefully without one):** LLM parsing/
reasoning/generation (empty `LLM_API_KEY` → rules/templates/extractive); real email
send (empty `EMAIL_*` → dry-run). **Not wired:** WhatsApp *inbound* auto-reply (send +
handler ready; needs a gateway WebSocket `message_received` subscription).

---

## Running

```bash
# core (MySQL only) — search / market / multi-turn / email drafting
make db-up                                   # start MySQL
npm run search -- "在 Irvine 找 3 居室带泳池 250万以下"
npm run search -- --market "Irvine"
npm run drafts -- report Irvine client@x.com  # draft a report; then: approve <id>

# semantic / RAG (Python) — hybrid vector+BM25 retrieval
docker start idx-qdrant                       # Qdrant (vectors already ingested)
source .venv/bin/activate
uvicorn --app-dir retrieval service:app --port 8099   # warm service (model preloaded)
python retrieval/search.py "craftsman with a big backyard" --city Irvine --max-price 2500000
python retrieval/rag.py "what is DOM?"

# tests
npm run test:all                              # all TS suites + typecheck + eval
python retrieval/test_recommend.py && python retrieval/test_rag.py

---

## Safety & Guardrails

This project follows the program's non-negotiable safety rules:

- **No autonomous outbound actions.** Emails are drafted, previewed, and require
  explicit confirmation before sending.
- **No secrets in logs or version control.** Credentials live only in `.env`.
- **No bulk data export.** Query result sets are capped (≤50 rows per query); full
  MLS dumps are never committed or exported.
- **Human oversight** on every destructive or outbound operation.

---

## License

Internship coursework — not licensed for redistribution. MLS data is confidential
and property of IDX Exchange.
