"""Rerank precision eval (#3, done right): does cross-encoder rerank improve RANKING QUALITY
over plain hybrid? Uses the GRADED set (retrieval_large.jsonl, 0/1/2 relevance) + nDCG — the
only metric that can judge ordering (known-item recall can't). Trusts the graded labels.

Two configs (to separate the pooling-depth artifact):
  - rerank@10 : CE reorders hybrid's top-10 (all in the judged pool -> artifact-free ordering test)
  - rerank@30 : CE reranks hybrid's top-30 -> top-10 (realistic, but ranks 11-30 may be unjudged
                -> promoted-but-unjudged items score 0 -> BIASED AGAINST rerank; read with caveat)
Paired per-query + significance (small set -> mean alone lies). Needs Qdrant + MySQL + fastembed CE.
  python eval/runners/eval_rerank_graded.py
"""
import json
import os
import sys
import statistics as st

import pymysql

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "retrieval"))
sys.path.insert(0, os.path.join(ROOT, "eval", "metrics"))
from common import load_env  # noqa: E402
from search import hybrid_search  # noqa: E402
from ir import ndcg_at_k, precision_at_k, mrr, mean  # noqa: E402
from fastembed.rerank.cross_encoder import TextCrossEncoder  # noqa: E402

DATA = os.path.join(ROOT, "eval", "datasets", "retrieval_large.jsonl")


def load():
    return [json.loads(l) for l in open(DATA) if l.strip() and json.loads(l)["label"]["relevant"]]


def remarks_for(ids, conn):
    if not ids:
        return {}
    fmt = ",".join(["%s"] * len(ids))
    with conn.cursor() as cur:
        cur.execute(f"SELECT id,L_City,L_Type_,L_Remarks FROM rets_property WHERE id IN ({fmt})", ids)
        rows = cur.fetchall()
    return {int(r["id"]): f"{r.get('L_Type_') or ''} in {r.get('L_City') or ''}. "
            f"{(r.get('L_Remarks') or '').replace(chr(10),' ')[:400]}" for r in rows}


def ce_reorder(ce, query, ids, conn):
    texts = remarks_for(ids, conn)
    docs = [texts.get(i, "") for i in ids]
    scores = list(ce.rerank(query, docs))
    return [ids[j] for j in sorted(range(len(ids)), key=lambda j: -scores[j])]


def paired(deltas, name):
    n = len(deltas); w = sum(1 for d in deltas if d > 1e-9); l = sum(1 for d in deltas if d < -1e-9)
    md = mean(deltas); sd = st.pstdev(deltas) * (n / (n - 1)) ** .5 if n > 1 else 0
    se = sd / n ** .5 if n else 0; t = md / se if se else 0
    print(f"  {name}: win {w}/tie {n-w-l}/loss {l}  meanΔ {md:+.4f}  t≈{t:.2f}  "
          f"{'SIG' if abs(t) >= 2 else 'ns'}")


def main():
    cases = load()
    model = os.environ.get("RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")
    print(f"[reranker model: {model}]")
    ce = TextCrossEncoder(model)
    env = load_env()
    conn = pymysql.connect(host=env.get("DB_HOST"), port=int(env.get("DB_PORT", 3306)),
                           user=env.get("DB_USER"), password=env.get("DB_PASSWORD"),
                           database=env.get("DB_NAME"), cursorclass=pymysql.cursors.DictCursor)

    agg = {k: {"ndcg": [], "p5": [], "mrr": []} for k in ("baseline", "rerank@10", "rerank@30")}
    d10, d30 = [], []
    for c in cases:
        q, rel = c["input"], c["label"]["relevant"]
        top10 = [p.id for p in hybrid_search(q, None, k=10, mode="hybrid")]
        top30 = [p.id for p in hybrid_search(q, None, k=30, mode="hybrid")]
        r10 = ce_reorder(ce, q, top10, conn)[:10]              # reorder top-10 (artifact-free)
        r30 = ce_reorder(ce, q, top30, conn)[:10]              # rerank top-30 -> top-10 (realistic)
        for name, ids in (("baseline", top10), ("rerank@10", r10), ("rerank@30", r30)):
            agg[name]["ndcg"].append(ndcg_at_k(ids, rel, 10))
            agg[name]["p5"].append(precision_at_k(ids, rel, 5))
            agg[name]["mrr"].append(mrr(ids, rel))
        b = ndcg_at_k(top10, rel, 10)
        d10.append(ndcg_at_k(r10, rel, 10) - b)
        d30.append(ndcg_at_k(r30, rel, 10) - b)
    conn.close()

    print(f"rerank precision on graded set: {len(cases)} queries (cross-encoder ms-marco-MiniLM)\n")
    print(f"  {'config':11} {'nDCG@10':>9} {'P@5':>7} {'MRR':>7}")
    for name in ("baseline", "rerank@10", "rerank@30"):
        a = agg[name]
        print(f"  {name:11} {mean(a['ndcg']):>9.4f} {mean(a['p5']):>7.4f} {mean(a['mrr']):>7.4f}")
    print("\npaired vs baseline (nDCG@10):")
    paired(d10, "rerank@10")
    paired(d30, "rerank@30")
    print("\n(rerank@10 = pure reorder of judged top-10, artifact-free;")
    print(" rerank@30 = realistic but ranks 11-30 may be unjudged -> biased AGAINST rerank)")


if __name__ == "__main__":
    main()
