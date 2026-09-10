"""Sweep the cross-encoder pool size against KNOWN-ITEM recall (objective labels).

The judge-graded sweep (sweep_rerank_pool.py) asks "how many of the top-10 are relevant".
This asks a different question: "is the ONE listing a human designated still in the top-k".
Objective gold, so no unjudged-item bias — but a target need not be the most 'relevant'-
looking listing, so rerank can help one metric and hurt the other. Both are worth watching.

Uses each mode's real production inputs: its filter AND the semantic text it would send
(auto's tool arg / the deterministic path's extractSemanticText), per skills.ts:139-141.

Needs Qdrant + MySQL.  Run: python eval/runners/sweep_rerank_knownitem.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "retrieval"))
from search import hybrid_search, build_filter  # noqa: E402
from common import load_env  # noqa: E402
from fastembed.rerank.cross_encoder import TextCrossEncoder  # noqa: E402
import pymysql  # noqa: E402

DATA = os.path.join(ROOT, "eval", "datasets", "mode_retrieval.jsonl")
PREDS = os.path.join(ROOT, "eval", "history", "mode_retrieval.preds.jsonl")
TYPE_DB = {"house": "Single Family Residence", "condo": "Condominium", "townhouse": "Townhouse"}
NS = [10, 15, 20, 30, 50]
POOL = 50


def main():
    gold = {json.loads(l)["id"]: json.loads(l) for l in open(DATA) if l.strip()}
    preds = {json.loads(l)["id"]: json.loads(l) for l in open(PREDS) if l.strip()}
    ce = TextCrossEncoder("Xenova/ms-marco-MiniLM-L-6-v2")
    env = load_env()
    conn = pymysql.connect(host=env.get("DB_HOST"), port=int(env.get("DB_PORT", 3306)),
                           user=env.get("DB_USER"), password=env.get("DB_PASSWORD"),
                           database=env.get("DB_NAME"), cursorclass=pymysql.cursors.DictCursor)

    def remarks(ids):
        if not ids:
            return {}
        fmt = ",".join(["%s"] * len(ids))
        with conn.cursor() as cur:
            cur.execute(f"SELECT id,L_City,L_Type_,L_Remarks FROM rets_property WHERE id IN ({fmt})", ids)
            return {int(r["id"]): f"{r.get('L_Type_') or ''} in {r.get('L_City') or ''}. {r.get('L_Remarks') or ''}"
                    for r in cur.fetchall()}

    modes = {"regex": ("regex_filter", "regex_semantic"), "auto": ("auto_filter", "auto_semantic")}
    agg = {(m, cfg): {5: [0, 0], 10: [0, 0]} for m in modes for cfg in ["hybrid"] + [f"CE top-{n}" for n in NS]}

    for cid, g in gold.items():
        tgt = g["gold"].get("known_item")
        if not tgt or cid not in preds:
            continue
        p = preds[cid]
        for m, (fkey, skey) in modes.items():
            f = p[fkey]
            text = (p.get(skey) or "").strip() or g["input"]
            flt = build_filter(city=f.get("city"), max_price=f.get("maxPrice"), min_price=f.get("minPrice"),
                               min_beds=f.get("beds"), pool=f.get("pool"),
                               ptype=TYPE_DB.get(f.get("propertyType")) if f.get("propertyType") else None)
            pool = [x.id for x in hybrid_search(text, flt, k=POOL, mode="hybrid")]
            variants = {"hybrid": pool[:10]}
            for n in NS:
                cand = pool[:n]
                if cand:
                    txt = remarks(cand)
                    scores = list(ce.rerank(text, [txt.get(i, "") for i in cand]))
                    variants[f"CE top-{n}"] = [i for i, _ in sorted(zip(cand, scores), key=lambda x: -x[1])][:10]
                else:
                    variants[f"CE top-{n}"] = []
            for cfg, ids in variants.items():
                for k in (5, 10):
                    agg[(m, cfg)][k][0] += int(tgt in ids[:k])
                    agg[(m, cfg)][k][1] += 1
    conn.close()

    print(f"known-item recall · {sum(1 for g in gold.values() if g['gold'].get('known_item'))} 条带 target\n")
    print(f"  {'模式':<7}{'配置':<14}{'R@5':>8}{'R@10':>8}{'Δ@10':>9}")
    for m in modes:
        base = agg[(m, "hybrid")][10]
        b10 = base[0] / base[1] if base[1] else 0
        for cfg in ["hybrid"] + [f"CE top-{n}" for n in NS]:
            a = agg[(m, cfg)]
            r5, r10 = a[5][0] / a[5][1], a[10][0] / a[10][1]
            d = "—" if cfg == "hybrid" else f"{r10 - b10:+.3f}"
            print(f"  {m:<7}{cfg:<14}{r5:>8.3f}{r10:>8.3f}{d:>9}")
        print()


if __name__ == "__main__":
    main()
