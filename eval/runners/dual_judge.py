"""Dual-judge calibration: score pooled candidates with a cross-encoder (different bias
profile than the LLM judge), compare the two judges, and select the most INFORMATIVE
(query, listing) pairs for a tiny human review — where the two judges DISAGREE most.

Why two judges: the LLM judge reads remarks and rewards lexical overlap -> tilts toward
BM25. A cross-encoder (ms-marco-MiniLM, trained on relevance) has a different bias -> where
they AGREE we trust the label; where they DISAGREE we ask the human. That's the cheap way
to get a fair-ish gold without mass annotation.

Outputs eval/history/human_review_queue.json: a small stratified set of pairs with a Chinese
display (query + listing summary) for the user to grade 相关/沾边/不相关.

  python eval/runners/dual_judge.py
Needs: Qdrant (pools cached already), MySQL (listing text), LLM (Chinese display), fastembed CE.
"""
import json
import os
import sys

import pymysql

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "retrieval"))
from common import load_env  # noqa: E402
from search import hybrid_search  # noqa: E402
from llm import chat  # noqa: E402
from fastembed.rerank.cross_encoder import TextCrossEncoder  # noqa: E402

DATA = os.path.join(ROOT, "eval", "datasets", "retrieval_large.jsonl")
JUDGE_CACHE = os.path.join(ROOT, "eval", "history", "retrieval_large_judge_cache.json")
OUT = os.path.join(ROOT, "eval", "history", "human_review_queue.json")
POOL_EACH = 12
# representative queries to review (one per category; concrete + discriminative)
REVIEW_QUERIES = [
    "cozy craftsman bungalow with lots of character",  # semantic
    "Sub-Zero and Wolf appliances",                    # lexical
    "single story ranch with a pool in a gated community",  # mixed
    "55 plus senior community",                        # concept
]
PER_QUERY = 3   # informative listings per query


def pool_ids(query):
    ids, seen = [], set()
    for mode in ("dense", "bm25", "hybrid"):
        for p in hybrid_search(query, None, k=POOL_EACH, mode=mode):
            if p.id not in seen:
                seen.add(p.id); ids.append(p.id)
    return ids


def fetch_texts(ids):
    env = load_env()
    conn = pymysql.connect(host=env.get("DB_HOST"), port=int(env.get("DB_PORT", 3306)),
                           user=env.get("DB_USER"), password=env.get("DB_PASSWORD"),
                           database=env.get("DB_NAME"), cursorclass=pymysql.cursors.DictCursor)
    fmt = ",".join(["%s"] * len(ids))
    with conn.cursor() as cur:
        cur.execute(f"SELECT id,L_City,L_Type_,L_Keyword2,LM_Dec_3,L_SystemPrice,L_Remarks "
                    f"FROM rets_property WHERE id IN ({fmt})", ids)
        rows = cur.fetchall()
    conn.close()
    out = {}
    for r in rows:
        out[int(r["id"])] = {
            "city": r.get("L_City"), "type": r.get("L_Type_"),
            "beds": r.get("L_Keyword2"), "baths": r.get("LM_Dec_3"),
            "price": r.get("L_SystemPrice"),
            "remarks": (r.get("L_Remarks") or "").replace("\n", " ").strip(),
        }
    return out


def ce_scores(ce, query, ids, texts):
    docs = [f"{texts[i]['type']} in {texts[i]['city']}. {texts[i]['remarks'][:400]}" for i in ids]
    scores = list(ce.rerank(query, docs))
    order = sorted(range(len(ids)), key=lambda j: -scores[j])
    rank = {ids[j]: r for r, j in enumerate(order)}   # 0 = best by cross-encoder
    return {ids[j]: float(scores[j]) for j in range(len(ids))}, rank


def zh_query(q):
    return chat(f"把这个英文房产搜索需求翻译成简洁自然的中文(只回译文):\n{q}") or q


def zh_listing(t):
    prompt = (f"用中文两句话概括这套房源的卖点/特征,供人工判断相关性(只回概括,不要评价):\n"
              f"类型 {t['type']}, 城市 {t['city']}, {t['beds']}室 {t['baths']}卫, 价格 {t['price']}\n"
              f"描述: {t['remarks'][:500]}")
    return chat(prompt) or t["remarks"][:120]


def main():
    grades_cache = json.load(open(JUDGE_CACHE))
    print("loading cross-encoder (ms-marco-MiniLM-L-6-v2)...")
    ce = TextCrossEncoder("Xenova/ms-marco-MiniLM-L-6-v2")

    items = []
    for q in REVIEW_QUERIES:
        ids = pool_ids(q)
        texts = fetch_texts(ids)
        llm = {int(k): v for k, v in grades_cache.get(q, {}).items()}   # id -> 0/1/2
        cs, crank = ce_scores(ce, q, ids, texts)
        n = len(ids)
        # informativeness: LLM says relevant(2) but CE ranks it low  OR  CE ranks top but LLM=0
        cand = []
        for i in ids:
            lg = llm.get(i, 0); cr = crank[i]
            llm_only = lg >= 2 and cr >= n * 0.5          # LLM loves, CE doesn't
            ce_only = cr <= 2 and lg == 0                  # CE loves, LLM doesn't
            agree = lg >= 2 and cr <= 2                    # both love (control)
            tag = "LLM偏爱" if llm_only else "CE偏爱" if ce_only else "两裁判一致" if agree else None
            score = (2 if llm_only or ce_only else 1 if agree else 0)
            cand.append((score, tag, i, lg, cr))
        cand = [c for c in cand if c[1]]
        cand.sort(key=lambda c: -c[0])
        picked, seen_tags = [], set()
        for sc, tag, i, lg, cr in cand:
            if tag in seen_tags and len(picked) < PER_QUERY:
                continue
            picked.append((tag, i, lg, cr)); seen_tags.add(tag)
            if len(picked) >= PER_QUERY:
                break
        qz = zh_query(q)
        for tag, i, lg, cr in picked:
            items.append({
                "query": q, "query_zh": qz, "listing_id": i,
                "llm_grade": lg, "ce_rank": cr, "ce_score": round(cs[i], 3),
                "disagreement": tag,
                "listing_zh": zh_listing(texts[i]),
                "attrs": {k: texts[i][k] for k in ("type", "city", "beds", "baths", "price")},
            })
        print(f"  {q[:40]:40} pooled {len(ids)}  picked {len(picked)}")

    json.dump(items, open(OUT, "w"), ensure_ascii=False, indent=2, default=str)
    print(f"\n{len(items)} review items -> {OUT}")


if __name__ == "__main__":
    main()
