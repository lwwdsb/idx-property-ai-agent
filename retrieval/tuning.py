"""Tunable algorithm parameters — Python side of the single source in config/tuning.json.

Mirrors src/tuning.ts: same file, same env override, same blast-radius grouping.
Loaded once at import (services are warm, and a sweep gets a fresh process anyway).

  IDX_TUNING=/abs/path/candidate.json uvicorn service:app --port 8099
"""
import json
import os

# Resolved from THIS file's location, not cwd — uvicorn starts from retrieval/.
TUNING_PATH = os.environ.get("IDX_TUNING", "").strip() or os.path.join(
    os.path.dirname(__file__), "..", "config", "tuning.json")

# No try/except on purpose: a missing/malformed file must fail loudly at startup rather
# than silently fall back to hardcoded values (which would make a sweep compare two runs
# that secretly used the same parameters).
with open(TUNING_PATH, encoding="utf-8") as fh:
    TUNING = json.load(fh)

_r = TUNING["shared"]["retrieval"]
_g = TUNING["shared"]["rag"]

RERANK_COARSE = _r["rerankCoarse"]     # coarse pool handed to the cross-encoder
PREFETCH = _r["prefetch"]              # per-path (dense/bm25) depth before RRF fusion
RERANK_ENABLED = _r["rerankEnabled"]   # default for SearchReq.rerank
SEARCH_TOP_K = _r["topK"]              # default for SearchReq.k
RAG_CHUNK_SIZE = _g["chunkSize"]
RAG_OVERLAP = _g["chunkOverlap"]
RAG_TOP_K = _g["topK"]
