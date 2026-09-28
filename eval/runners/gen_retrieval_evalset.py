"""Build a LARGER, stratified retrieval eval set with pooled + LLM-judged relevance.

SCHEMA v2 (2026-09-28) — the label now records EVERY judged id, including the zeros.

v1 stored only the positives plus a `pool_size` count, which made "judged and found
irrelevant" indistinguishable from "never judged". Scoring treats an unjudged document as
irrelevant (the standard convention), so the distinction is not cosmetic: after the index was
rebuilt, 50% of dense's top-10 and ~30% of bm25/hybrid's were documents nobody had ever judged,
and dense's nDCG@10 "fell" from ~0.75 to 0.445 on a change the objective known-item set found
indistinguishable (21/39 -> 20/39). The number moved because the ruler went blind, and nothing
in the data could say so.

Two changes make that detectable instead of silent:
  label.graded   every pooled id -> 0/1/2. `relevant` is still written (derived, positives only)
                 so existing readers keep working, but `graded` is the source of truth.
  label.pool     which modes were pooled, how deep, and WHICH COLLECTION — the pool belongs to an
                 index, and a pool from another index is not a pool.

And one thing that had to be fixed before any of it could be re-run: see the cache key below.

Pipeline (mirrors how the original 24-set was built, but bigger + category-tagged):
  1. read stratified query seeds (eval/datasets/retrieval_queries.txt)
  2. POOL candidates per query = union of dense / bm25 / hybrid top-N  (unbiased across modes)
  3. fetch listing text (remarks + structured) from MySQL for the pool
  4. LLM-judge relevance 0/1/2 in ONE call per query (tightened rubric)
  5. write eval/datasets/retrieval_large.jsonl  (schema matches retrieval.jsonl + meta.category)

Judgments are cached (eval/history/retrieval_large_judge_cache.json) so reruns are cheap.
Needs Qdrant up (pooling) + MySQL (text) + LLM_API_KEY (judge).

  python eval/runners/gen_retrieval_evalset.py
"""
import json
import os
import re
import sys

import pymysql

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "retrieval"))
from common import load_env, COLLECTION  # noqa: E402
from search import hybrid_search  # noqa: E402
from llm import chat, llm_available  # noqa: E402

SEEDS = os.path.join(ROOT, "eval", "datasets", "retrieval_queries.txt")
OUT = os.path.join(ROOT, "eval", "datasets", "retrieval_large.jsonl")
HIST = os.path.join(ROOT, "eval", "history")
CACHE = os.path.join(HIST, "retrieval_large_judge_cache.json")
POOL_EACH = 12          # top-N per mode -> union pool
REMARK_CHARS = 350      # truncate remarks fed to the judge
SCHEMA = 2
# The reranker's candidate depth, read from the same config production reads.
RERANK_COARSE = json.load(open(os.path.join(ROOT, "config", "tuning.json")))["shared"]["retrieval"]["rerankCoarse"]


def pool_key(q, ids):
    """Cache key = query AND the pool it was judged against.

    It used to be the query alone, and that is the bug that would have silently defeated this
    whole exercise: after an index change the pool changes, but a query-only key reports 70/70
    cache hits and zero LLM cost, handing back grades that describe documents the new index no
    longer retrieves. The tool meant to fix the blind ruler was itself blind in the same way.
    Including the pool makes a changed pool a cache MISS, which is what it is."""
    import hashlib
    h = hashlib.sha256(",".join(sorted(str(i) for i in ids)).encode()).hexdigest()[:12]
    return f"{q}||{h}"


def read_queries_from(path):
    """Re-pool an EXISTING set: take its queries, pool against the CURRENT index, re-judge.

    This is the operation an index change forces, so it is a tool rather than a one-off script.
    The 24-case set has no seed file — its queries live in the jsonl — and it is the set the
    report and the gate actually read, so it has to be re-poolable too."""
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        out.append((r["input"], (r.get("meta") or {}).get("category", "unknown"), r["id"]))
    return out


def read_seeds():
    out = []
    for line in open(SEEDS, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        q, _, cat = line.partition("|")
        out.append((q.strip(), (cat.strip() or "uncategorized")))
    return out


def pool_ids(query):
    """Union of every system's COARSE candidate set, not just its top-k.

    The pool has to cover what each evaluated system could surface, and production is
    "hybrid coarse top-RERANK_COARSE -> cross-encoder -> top-k". Pooling hybrid at 12 while the
    reranker chooses out of 20 leaves the reranker able to promote documents nobody judged — the
    same blindness one level up, and it would show as the reranker "winning" or "losing" on
    coverage rather than on quality. Pooling hybrid to the full coarse depth closes it without
    needing the service running, because every document the reranker can reach is then judged."""
    ids = []
    for mode, depth in (("dense", POOL_EACH), ("bm25", POOL_EACH), ("hybrid", max(POOL_EACH, RERANK_COARSE))):
        ids += [p.id for p in hybrid_search(query, None, k=depth, mode=mode)]
    seen, uniq = set(), []
    for i in ids:
        if i not in seen:
            seen.add(i); uniq.append(i)
    return uniq


def fetch_texts(ids):
    if not ids:
        return {}
    env = load_env()
    conn = pymysql.connect(
        host=env.get("DB_HOST", "127.0.0.1"), port=int(env.get("DB_PORT", 3306)),
        user=env.get("DB_USER", "root"), password=env.get("DB_PASSWORD", ""),
        database=env.get("DB_NAME", "idx_exchange"), cursorclass=pymysql.cursors.DictCursor)
    fmt = ",".join(["%s"] * len(ids))
    sql = (f"SELECT id, L_City, L_Type_, L_Keyword2, LM_Dec_3, L_Remarks "
           f"FROM rets_property WHERE id IN ({fmt})")
    with conn.cursor() as cur:
        cur.execute(sql, ids)
        rows = cur.fetchall()
    conn.close()
    out = {}
    for r in rows:
        head = " ".join(str(x) for x in [r.get("L_Type_") or "", f"in {r.get('L_City')}" if r.get("L_City") else "",
                        f"{r.get('L_Keyword2')}bd" if r.get("L_Keyword2") else "",
                        f"{r.get('LM_Dec_3')}ba" if r.get("LM_Dec_3") else ""] if x)
        rem = (r.get("L_Remarks") or "").replace("\n", " ").strip()[:REMARK_CHARS]
        out[int(r["id"])] = f"{head}. {rem}"
    return out


JUDGE_SYS = ("You are a strict real-estate search relevance judge. Grade how well each listing "
             "matches the search intent. Be conservative: only give 2 when the listing CLEARLY has "
             "the distinctive feature(s) the query asks for.")
JUDGE_RUBRIC = ("Grades: 2 = clearly and specifically matches the key feature(s); "
                "1 = partial / plausible but weak or generic match; 0 = does not match. "
                'Return ONLY a JSON object mapping listing id (string) to grade, e.g. {"123":2,"456":0}.')


def judge(query, texts):
    listings = "\n".join(f"[{i}] {t}" for i, t in texts.items())
    out = chat(f"{JUDGE_RUBRIC}\n\nQuery: {query}\n\nListings:\n{listings}", system=JUDGE_SYS) or ""
    m = re.search(r"\{[\s\S]*\}", out)
    try:
        raw = json.loads(m.group(0)) if m else {}
    except Exception:
        raw = {}
    grades = {}
    for k, v in raw.items():
        try:
            g = int(v)
            if str(k).isdigit() and g in (0, 1, 2):
                grades[str(k)] = g
        except (TypeError, ValueError):
            continue
    return grades


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", metavar="JSONL",
                    help="re-pool an existing set's queries against the CURRENT index "
                         "(keeps the ids, re-judges); default is to build from the seed file")
    ap.add_argument("--out", dest="out", help="output path (default depends on mode)")
    a = ap.parse_args()

    if not llm_available():
        print("LLM_API_KEY not set — the judge needs the LLM. Aborting."); sys.exit(1)

    keep_ids = None
    if a.src:
        src = a.src if os.path.isabs(a.src) else os.path.join(ROOT, a.src)
        rows = read_queries_from(src)
        seeds = [(q, cat) for q, cat, _ in rows]
        keep_ids = [rid for _, _, rid in rows]
        out_path = a.out or src
        print(f"RE-POOLING {os.path.basename(src)}: {len(seeds)} queries against collection "
              f"'{COLLECTION}' (dense/bm25 top-{POOL_EACH}, hybrid top-{max(POOL_EACH, RERANK_COARSE)})\n")
    else:
        seeds = read_seeds()
        out_path = a.out or OUT
        print(f"generating retrieval eval set: {len(seeds)} queries (pool top-{POOL_EACH}/mode)\n")
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}

    records, n_judged, empty = [], 0, []
    for idx, (q, cat) in enumerate(seeds, 1):
        ids = pool_ids(q)
        texts = fetch_texts(ids)
        ck = pool_key(q, ids)
        if ck in cache:
            grades = cache[ck]
        else:
            grades = judge(q, texts)
            cache[ck] = grades
            n_judged += 1
            os.makedirs(HIST, exist_ok=True)
            json.dump(cache, open(CACHE, "w"), ensure_ascii=False)
        # EVERY pooled id gets an entry, zeros included. A pooled id the judge did not mention is
        # recorded as 0 explicitly — "the judge saw it and said no" — which is exactly the state v1
        # could not express.
        graded = {str(i): int(grades.get(str(i), grades.get(i, 0)) or 0) for i in ids}
        relevant = {i: g for i, g in graded.items() if g > 0}
        if not relevant:
            empty.append(q)
        rid = keep_ids[idx - 1] if keep_ids else f"ret-l-{idx:03d}"
        records.append({"id": rid, "input": q,
                        "label": {"schema": SCHEMA, "graded": graded, "relevant": relevant,
                                  "pool_size": len(ids),
                                  "pool": {"modes": list(MODES) if "MODES" in globals() else ["dense", "bm25", "hybrid"],
                                           "top_n_each": POOL_EACH, "collection": COLLECTION}},
                        "meta": {"source": "llm-gen", "verified": False, "category": cat}})
        print(f"  [{idx:>2}/{len(seeds)}] {cat:9} pool={len(ids):>2} rel={len(relevant):>2}  {q[:44]}")

    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    cats = Counter(r["meta"]["category"] for r in records)
    rels = [len(r["label"]["relevant"]) for r in records]
    print(f"\n{len(records)} queries -> {out_path}")
    print(f"  categories: {dict(cats)}")
    print(f"  relevant/query: min {min(rels)} / avg {round(sum(rels)/len(rels),1)} / max {max(rels)}")
    print(f"  newly judged this run: {n_judged}")
    if empty:
        print(f"  ! {len(empty)} queries with NO relevant items (judge too strict or no matches): {empty[:5]}")


if __name__ == "__main__":
    main()
