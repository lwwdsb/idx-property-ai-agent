"""Cross-encoder reranking experiment (#3): retrieve wide, rerank the candidates with a
cross-encoder, take top-k. Does it lift known-item recall over plain hybrid?

Builds on #2 (uses the English-translated query, since translation is a proven win). Compares:
  - baseline : hybrid top-k (translated query, auto filter)
  - reranked : hybrid top-50 -> cross-encoder(query, remark) rerank -> top-k
Objective known-item recall (no LLM judge). Needs Qdrant + MySQL + LLM(translate) + fastembed CE.
  python eval/runners/eval_rerank.py
"""
import json
import os
import sys
from collections import defaultdict

import pymysql

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "retrieval"))
from common import load_env  # noqa: E402
from search import hybrid_search, build_filter  # noqa: E402
from llm import chat, llm_available  # noqa: E402
from fastembed.rerank.cross_encoder import TextCrossEncoder  # noqa: E402

DATA = os.path.join(ROOT, "eval", "datasets", "mode_retrieval.jsonl")
PREDS = os.path.join(ROOT, "eval", "history", "mode_retrieval.preds.jsonl")
CACHE = os.path.join(ROOT, "eval", "history", "xlingual_cache.json")
TYPE_DB = {"condo": "Condominium", "townhouse": "Townhouse", "single-family": "SingleFamilyResidence"}
WIDE = 50


def load(p): return [json.loads(l) for l in open(p) if l.strip()]


def flt_of(f):
    return build_filter(city=f.get("city"), max_price=f.get("maxPrice"), min_price=f.get("minPrice"),
                        min_beds=f.get("beds"), pool=f.get("pool"),
                        ptype=TYPE_DB.get(f.get("propertyType")) if f.get("propertyType") else None)


def remarks_for(ids, conn):
    if not ids:
        return {}
    fmt = ",".join(["%s"] * len(ids))
    with conn.cursor() as cur:
        cur.execute(f"SELECT id,L_City,L_Type_,L_Remarks FROM rets_property WHERE id IN ({fmt})", ids)
        rows = cur.fetchall()
    return {int(r["id"]): f"{r.get('L_Type_') or ''} in {r.get('L_City') or ''}. "
            f"{(r.get('L_Remarks') or '').replace(chr(10),' ')[:400]}" for r in rows}


def main():
    if not llm_available():
        print("need LLM"); sys.exit(1)
    gold = {c["id"]: c for c in load(DATA)}
    preds = {p["id"]: p for p in load(PREDS)}
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    ce = TextCrossEncoder("Xenova/ms-marco-MiniLM-L-6-v2")
    env = load_env()
    conn = pymysql.connect(host=env.get("DB_HOST"), port=int(env.get("DB_PORT", 3306)),
                           user=env.get("DB_USER"), password=env.get("DB_PASSWORD"),
                           database=env.get("DB_NAME"), cursorclass=pymysql.cursors.DictCursor)

    rec = {v: defaultdict(lambda: [0, 0, 0, 0]) for v in ("baseline", "reranked")}
    lifts = []
    for cid, c in gold.items():
        ki = c["gold"].get("known_item")
        if not ki:
            continue
        q = cache.get(c["input"], c["input"]) if c["lang"] != "en" else c["input"]  # translated (from #2)
        flt = flt_of(preds[cid]["auto_filter"])
        wide = [p.id for p in hybrid_search(q, flt, k=WIDE, mode="hybrid")]
        base_ids = wide[:20]
        texts = remarks_for(wide, conn)
        docs = [texts.get(i, "") for i in wide]
        scores = list(ce.rerank(q, docs))
        rer_ids = [wide[j] for j in sorted(range(len(wide)), key=lambda j: -scores[j])][:20]
        for variant, ids in (("baseline", base_ids), ("reranked", rer_ids)):
            h5, h10, h20 = int(ki in ids[:5]), int(ki in ids[:10]), int(ki in ids[:20])
            for bucket in (c["style"], "ALL"):
                a = rec[variant][bucket]; a[0] += h5; a[1] += h10; a[2] += h20; a[3] += 1
        b10, r10 = int(ki in base_ids[:10]), int(ki in rer_ids[:10])
        if b10 != r10:
            lifts.append((cid, c["style"], b10, r10))
    conn.close()

    print(f"cross-encoder rerank (wide={WIDE}) vs plain hybrid — both on translated query + auto filter\n")
    print(f"{'stratum':13} {'variant':9} {'R@5':>6} {'R@10':>6} {'R@20':>6} {'n':>3}")
    for bucket in ("clean", "messy", "multi-intent", "ALL"):
        for v in ("baseline", "reranked"):
            a = rec[v][bucket]
            if a[3] == 0:
                continue
            print(f"{bucket:13} {v:9} {a[0]/a[3]:>6.2f} {a[1]/a[3]:>6.2f} {a[2]/a[3]:>6.2f} {a[3]:>3}")
        print()
    print("per-case recall@10 changes (baseline -> reranked):")
    for cid, st, b, r in lifts:
        print(f"  {cid} [{st}] {b} -> {r}  {'LIFT' if r > b else 'DROP'}")


if __name__ == "__main__":
    main()
