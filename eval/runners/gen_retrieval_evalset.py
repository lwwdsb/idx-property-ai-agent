"""Build a LARGER, stratified retrieval eval set with pooled + LLM-judged relevance.

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
from common import load_env  # noqa: E402
from search import hybrid_search  # noqa: E402
from llm import chat, llm_available  # noqa: E402

SEEDS = os.path.join(ROOT, "eval", "datasets", "retrieval_queries.txt")
OUT = os.path.join(ROOT, "eval", "datasets", "retrieval_large.jsonl")
HIST = os.path.join(ROOT, "eval", "history")
CACHE = os.path.join(HIST, "retrieval_large_judge_cache.json")
POOL_EACH = 12          # top-N per mode -> union pool
REMARK_CHARS = 350      # truncate remarks fed to the judge


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
    ids = []
    for mode in ("dense", "bm25", "hybrid"):
        ids += [p.id for p in hybrid_search(query, None, k=POOL_EACH, mode=mode)]
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
    if not llm_available():
        print("LLM_API_KEY not set — the judge needs the LLM. Aborting."); sys.exit(1)

    seeds = read_seeds()
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    print(f"generating retrieval eval set: {len(seeds)} queries (pool top-{POOL_EACH}/mode)\n")

    records, n_judged, empty = [], 0, []
    for idx, (q, cat) in enumerate(seeds, 1):
        ids = pool_ids(q)
        texts = fetch_texts(ids)
        if q in cache:
            grades = cache[q]
        else:
            grades = judge(q, texts)
            cache[q] = grades
            n_judged += 1
            os.makedirs(HIST, exist_ok=True)
            json.dump(cache, open(CACHE, "w"), ensure_ascii=False)
        relevant = {i: g for i, g in grades.items() if g > 0}
        if not relevant:
            empty.append(q)
        records.append({"id": f"ret-l-{idx:03d}", "input": q,
                        "label": {"relevant": relevant, "pool_size": len(ids)},
                        "meta": {"source": "llm-gen", "verified": False, "category": cat}})
        print(f"  [{idx:>2}/{len(seeds)}] {cat:9} pool={len(ids):>2} rel={len(relevant):>2}  {q[:44]}")

    with open(OUT, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    cats = Counter(r["meta"]["category"] for r in records)
    rels = [len(r["label"]["relevant"]) for r in records]
    print(f"\n{len(records)} queries -> {OUT}")
    print(f"  categories: {dict(cats)}")
    print(f"  relevant/query: min {min(rels)} / avg {round(sum(rels)/len(rels),1)} / max {max(rels)}")
    print(f"  newly judged this run: {n_judged}")
    if empty:
        print(f"  ! {len(empty)} queries with NO relevant items (judge too strict or no matches): {empty[:5]}")


if __name__ == "__main__":
    main()
