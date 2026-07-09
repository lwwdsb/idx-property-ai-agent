# IDX Property AI Agent

A production-style AI assistant for real estate, built on the
[OpenClaw](https://github.com/openclaw/openclaw) runtime. It answers natural-language
questions over ~145K real California MLS records — property search, market analytics,
semantic recommendations, RAG knowledge Q&A — and communicates over WhatsApp and email,
with human approval on every outbound action.

> **AI Agentic Engineer Internship — IDX Exchange · Summer 2026 · 12 Weeks**

The system is a **registry of specialized skills coordinated by a deterministic
orchestrator** (with a reserved LLM reasoning layer) — not an autonomous agent loop.
That choice is deliberate: robustness and predictability over free-form autonomy.

---

## Architecture

```
WhatsApp  ⇄  OpenClaw gateway  ⇄  agent (DeepSeek — thin front door)
                                     │  calls one tool: ask_idx_assistant
                                     ▼
                         Orchestrate service (HTTP :8100)
                                     │  deterministic router (intent + parse)
        ┌────────────┬──────────────┼──────────────┬──────────────┐
      search       market        recommend        RAG           email
   MySQL / Qdrant  median/trend  structured+       grounded      draft → human
   (hard filter +  over sold     semantic blend    + cited       approve → send
    dense+BM25)    comps         + comp verifier                 (locked at tool layer)
        │            │             │                │              │
      MySQL  ◄───────┴─────────────┴──── Qdrant (vectors) ─────────┘
                                   ▲
                    Warm Python retrieval service (:8099)
                    fastembed bge-small (local) · no embedding key
```

**Design principle — the LLM lives at the edges.** Routing, SQL, price validation, and
the send gate are deterministic code. The LLM (DeepSeek) is used only where it is
irreplaceable: parsing fuzzy phrasing, phrasing grounded answers, and nuanced reasoning.
Facts are never authored by the model — prices and addresses come from database rows.

**Robustness order — 丙 > 甲 > 乙** (a machine-executable tie-breaker): never send a
wrong outbound message > never be confidently wrong (hallucinate) > always stay
responsive (degrade gracefully).

**Tech stack:** OpenClaw · TypeScript · Python · MySQL · Qdrant · fastembed (local
`bge-small`) · DeepSeek (chat) · WhatsApp · Nodemailer

---

## Data

Three curated MySQL tables in a local schema (`idx_exchange`) — the working tables from
the internship FTP `sql/` folder:

| Table | Rows | Role |
|---|---|---|
| `rets_property` | 53,122 active listings (CA), 130+ cryptic `L_*` fields | Live search & discovery |
| `california_sold` | 87,157 sold transactions (CA), 46 fields | Historical comps & analytics |
| `rets_openhouse` | 4,282 open-house events | Open-house lookups |

**Join pattern:** `CAST(rets_property.L_ListingID AS UNSIGNED) = california_sold.ListingKey`,
or match on city + postal code for market-level analysis. A **field dictionary**
(`schema/columns.ts`) maps semantic names to the cryptic physical columns
(e.g. `beds → L_Keyword2`) as a single source of truth.

> The MLS data is confidential and is **not** committed to this repository
> (see `.gitignore`). Dumps are imported into a local MySQL instance only.

---

## Capabilities

- **NL property search** — `normalize → regex fast-path → DeepSeek fallback → clarify`.
  Bilingual (EN/中文). Pure structured queries hit MySQL; queries with soft/semantic
  content (e.g. *"ocean-view craftsman with a big backyard"*) route to the Qdrant hybrid.
- **Semantic search** — dense (`bge-small`) + BM25, fused with Reciprocal Rank Fusion,
  with hard constraints applied as **payload filters inside Qdrant** (filter-first, then
  rank). Falls back to MySQL if Qdrant is down.
- **Market analytics** — true **median** price, $/sqft, days-on-market, sold-to-list, and
  a 12-month trend over `california_sold`.
- **Recommendations** — blend of structured similarity (60%) + semantic similarity (40%),
  with a **comp price check** and a deterministic verifier gate (withholds a verdict when
  fewer than 3 comparable sales exist — no confident guess on thin data).
- **RAG knowledge Q&A** — grounded, cited answers over a curated corpus; extractive
  fallback with no key. The model never invents facts beyond retrieved context.
- **WhatsApp inbound auto-reply** — a real message routes through the orchestrator and
  replies automatically.
- **Email** — draft a report, preview it, then **`approve <id>`** (from WhatsApp or CLI)
  to send. Single/bulk share one path (multi-recipient uses BCC); sending is **locked at
  the tool layer** so the LLM structurally cannot send on its own.

---

## Project Structure

```
idx-property-ai-agent/
├── schema/            # Field dictionary (columns.ts) + SQL DDL/indexes (code, not data)
├── src/               # TypeScript live path
│   ├── search/          # NL parse (regex+normalize+LLM) → filters → parameterized SQL
│   ├── market/          # median-based market stats
│   ├── agent/           # multi-turn session + conversation
│   ├── orchestrator/    # router, skill registry, bridge, draft commands
│   ├── email/           # draft-then-approve, persistent draft store, templates
│   ├── whatsapp/        # inbound handler (idempotency, rate limit, degrade)
│   ├── server/          # orchestrate HTTP service (:8100)
│   └── llm/             # provider-agnostic chat client
├── retrieval/         # Python: fastembed + Qdrant hybrid search, recommend, RAG, warm service
├── openclaw-plugin/   # OpenClaw plugin: registers ask_idx_assistant (thin front door)
├── knowledge/         # RAG corpus (field glossary + hand-written terms)
├── scripts/           # import.sh, check_env.py, start-local.sh (make up), stop-local.sh
├── Makefile           # import | indexes | check | up | down | test:all
└── .env.example
```

---

## Setup

### Prerequisites
- Node.js (v20+) and npm · Python 3.10+ · MySQL · Docker (for Qdrant)
- OpenClaw installed (`npm install -g openclaw`, then `openclaw onboard`)
- **No OpenAI billing needed** — embeddings run locally (fastembed); a chat key
  (DeepSeek recommended) is optional and the system degrades gracefully without one.

### 1. Clone & configure
```bash
git clone https://github.com/lwwdsb/idx-property-ai-agent.git
cd idx-property-ai-agent
cp .env.example .env          # set DB_PASSWORD; optionally LLM_API_KEY, EMAIL_*
npm install
python3 -m venv .venv && source .venv/bin/activate && pip install -r retrieval/requirements.txt
```

### 2. Data & indexes
Download the SQL tables from the internship FTP into the repo root, then:
```bash
make import     # idempotent import into idx_exchange (SET sql_mode='' for zero-dates)
make indexes    # high-frequency filter indexes
make check      # validate env + DB row counts
```

### 3. Vectors (for semantic search / recommend)
```bash
docker start idx-qdrant                      # or: docker run -p 6333:6333 qdrant/qdrant
source .venv/bin/activate
python retrieval/ingest.py --resume          # embed + upsert listings into Qdrant
```

### 4. Connect WhatsApp
```bash
openclaw channels login --channel whatsapp   # scan the QR via Linked Devices
```

---

## Running

```bash
# bring up the whole local stack (MySQL, Qdrant, retrieval :8099, orchestrate :8100)
make up
make down                                     # stop the app services

# ask the orchestrator directly (what the WhatsApp tool calls)
curl -s localhost:8100/orchestrate -H 'Content-Type: application/json' \
  -d '{"userId":"me","message":"在 Irvine 找有山景的 3 居室 300万以下"}'

# CLI utilities
npm run search  -- "在 Irvine 找 3 居室带泳池 250万以下"
npm run drafts  -- report Irvine client@x.com     # draft; then "approve <id>" to send
python retrieval/search.py "craftsman with a big backyard" --city Irvine --max-price 2500000
python retrieval/rag.py "what is DOM?"
```

Once the stack is up and WhatsApp is linked, messaging the linked number returns an
automatic reply through the full pipeline.

---

## 12-Week Roadmap

| Week | Module | Status |
|------|--------|--------|
| 0 | Environment, MySQL import, indexes, env validation | ✅ |
| 1 | Skeleton + field dictionary + shared modules | ✅ |
| 2 | NL property search (normalize → regex → LLM fallback) | ✅ |
| 3 | MySQL query layer (parameterized, ≤50 cap, FULLTEXT, DTO) | ✅ |
| 4 | Conversational agent (multi-turn, patch-merge, pluggable store) | ✅ |
| 5 | Market analytics over `california_sold` (true median, trend) | ✅ |
| 6 | Hybrid retrieval — dense + BM25 + RRF in Qdrant | ✅ |
| 7 | Recommendation + comp price validation + verifier gate | ✅ |
| 8 | RAG Q&A (grounded + cited, extractive fallback) | ✅ |
| 9 | Orchestrator — skill registry + deterministic router + recipes | ✅ |
| 10 | Warm retrieval service (~160× faster) + embedding intent + WhatsApp handler | ✅ |
| 11 | Email agent — draft-then-approve, outbound locked at tool layer | ✅ |
| 12 | Wrap-up: full skill set wired, test suite, docs | ✅ |

**Post-12 (live, verified end-to-end):**
- WhatsApp **inbound auto-reply** via an OpenClaw plugin (thin front door → our tool).
- **DeepSeek** wired for all reserved LLM slots (parse / RAG / verifier / email) and the
  front-door agent.
- **Email send live** (Gmail SMTP) with approval from WhatsApp (`approve <id>` / `cancel <id>`).
- **Semantic listing search wired into the orchestrator** (soft query → Qdrant hybrid).

*Reserved / optional (degrade gracefully):* empty `LLM_API_KEY` → rules/templates/extractive;
empty `EMAIL_*` → dry-run. *Enhancement not yet done:* cross-encoder rerank after RRF.

---

## Safety & Guardrails

- **No autonomous outbound.** Email is drafted and previewed; a human `approve <id>` is
  the only send path, and the send capability is not exposed to the LLM (丙).
- **Operator allowlist** on draft/approve; **batch cap** (≤25 recipients, BCC for bulk).
- **Grounded answers only.** Facts come from DB rows; the verifier withholds on thin data.
- **No secrets or bulk data in git.** Query result sets capped (≤50 rows); `.env`, SQL
  dumps, and vectors are gitignored.

---

## Tests

```bash
npm run test:all                                          # all TS suites + typecheck + eval
python retrieval/test_recommend.py && python retrieval/test_rag.py
```
All green: TS `unit / eval / agent / market / orchestrator / whatsapp / email`;
Python `recommend 20 · rag 10`.

---

## License

Internship coursework — not licensed for redistribution. MLS data is confidential
and property of IDX Exchange.
