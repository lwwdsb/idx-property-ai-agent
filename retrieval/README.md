# Hybrid Retrieval (Week 6)

Semantic + keyword multi-path recall over the active listings, in Qdrant.

## Why this design

- **Two paths, fused.** A dense vector path (semantic: "craftsman with character")
  and a BM25 sparse path (keyword: "solar panels", "ADU", proper nouns). They find
  *different* things — dense misses exact terms, BM25 misses paraphrase — so we run
  both and fuse with **Reciprocal Rank Fusion (RRF)**. `benchmark.py` quantifies the
  low overlap between the two paths (the justification for fusing).
- **Filtered ANN.** Hard constraints (city / price / beds / type / pool) live in the
  Qdrant payload and are applied *inside* the vector search, so "先筛后排" happens in
  one query instead of SQL-then-rank.
- **Local + free.** Dense (`BAAI/bge-small-en-v1.5`, 384-d) and BM25 (`Qdrant/bm25`,
  IDF computed server-side by Qdrant) both run locally via **fastembed** — no
  embedding API key, ~$0, and MLS text never leaves the machine.

## Scale note (deliberate)

At ~53K listings this does **not** need a managed vector DB or ANN tuning — exact
search would already be sub-millisecond after filtering. Qdrant + HNSW is used to
practice the production pattern (filtered hybrid search) and to benchmark it against
the crossover point where ANN actually starts to matter (millions of vectors).

## Layout

- `common.py`  — Qdrant client, models, collection schema, doc-text + payload builders
- `ingest.py`  — MySQL active listings → dense + BM25 vectors → Qdrant (incremental-friendly by id)
- `search.py`  — hybrid / dense / bm25 query with payload filters
- `benchmark.py` — latency + path-complementarity report
- `requirements.txt`

## Usage

```bash
# 0. Qdrant must be running:  docker run -d --name idx-qdrant -p 6333:6333 -v idx_qdrant_storage:/qdrant/storage qdrant/qdrant
python3 -m venv .venv && source .venv/bin/activate && pip install -r retrieval/requirements.txt

python retrieval/ingest.py --recreate                         # build the collection (all active listings)
python retrieval/search.py "craftsman with a big backyard" --city Irvine --max-price 2500000 --min-beds 3
python retrieval/search.py "solar panels" --mode bm25         # compare a single path
python retrieval/benchmark.py                                 # dense vs bm25 vs hybrid
```

Data (vectors) lives in the Qdrant docker volume `idx_qdrant_storage`, never in git.
