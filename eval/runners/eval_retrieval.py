"""Retrieval eval — runs dense / BM25 / hybrid over the labeled queries and computes
IR metrics against the graded relevance judgments.

The headline result: does HYBRID beat single-path on nDCG@10? This upgrades the old
"0.03 top-10 overlap" (complementarity) into a QUALITY number (hybrid ranks the
relevant items higher). Also sweeps RRF prefetch for the tuning phase.

  python eval/runners/eval_retrieval.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "retrieval"))
sys.path.insert(0, os.path.join(ROOT, "eval", "metrics"))
from search import hybrid_search  # noqa: E402
from ir import ndcg_at_k, recall_at_k, mrr, precision_at_k, mean  # noqa: E402

DATA = os.path.join(ROOT, "eval", "datasets", "retrieval.jsonl")
HIST = os.path.join(ROOT, "eval", "history")
MODES = ["dense", "bm25", "hybrid"]


def load():
    with open(DATA) as f:
        return [json.loads(l) for l in f if l.strip()]


def ranked_ids(query, mode, k=10):
    return [p.id for p in hybrid_search(query, None, k=k, mode=mode)]


def eval_mode(cases, mode):
    ndcg, rec, rr, prec = [], [], [], []
    for c in cases:
        rel = c["label"]["relevant"]
        if not rel:
            continue
        ranked = ranked_ids(c["input"], mode, k=10)
        ndcg.append(ndcg_at_k(ranked, rel, 10))
        rec.append(recall_at_k(ranked, rel, 10))
        rr.append(mrr(ranked, rel))
        prec.append(precision_at_k(ranked, rel, 5))
    return {
        "nDCG@10": round(mean(ndcg), 4),
        "recall@10": round(mean(rec), 4),
        "MRR": round(mean(rr), 4),
        "precision@5": round(mean(prec), 4),
        "n": len(ndcg),
    }


def main():
    cases = load()
    verified = sum(1 for c in cases if c["meta"].get("verified"))
    print(f"retrieval eval: {len(cases)} queries "
          f"(human-verified: {verified}/{len(cases)})\n")

    results = {}
    print(f"  {'mode':8} {'nDCG@10':>9} {'recall@10':>10} {'MRR':>7} {'P@5':>7}")
    for mode in MODES:
        m = eval_mode(cases, mode)
        results[mode] = m
        print(f"  {mode:8} {m['nDCG@10']:>9} {m['recall@10']:>10} {m['MRR']:>7} {m['precision@5']:>7}")

    best = max(MODES, key=lambda mo: results[mo]["nDCG@10"])
    print(f"\n  best nDCG@10: {best} ({results[best]['nDCG@10']})")
    if best == "hybrid":
        d = results["hybrid"]["nDCG@10"] - max(results["dense"]["nDCG@10"], results["bm25"]["nDCG@10"])
        print(f"  hybrid beats best single-path by +{round(d, 4)} nDCG@10 → fusion adds quality, not just coverage.")

    out = os.path.join(HIST, "retrieval.metrics.json")
    os.makedirs(HIST, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"per_mode": results, "n_queries": len(cases), "verified": verified}, f, indent=2)
    print(f"\nmetrics written to {out}")


if __name__ == "__main__":
    main()
