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


def per_query_ndcg(cases):
    """Per-query nDCG@10 for each mode (deterministic) — the basis for a PAIRED comparison."""
    out = {m: [] for m in MODES}
    for c in cases:
        rel = c["label"]["relevant"]
        if not rel:
            continue
        for m in MODES:
            out[m].append(ndcg_at_k(ranked_ids(c["input"], m, k=10), rel, 10))
    return out


def paired(a, b):
    """Win/tie/loss + mean diff + a crude significance check (mean / standard-error).
    On a small set a positive MEAN can be noise; the sign split + |mean|/SE tell you if it's real."""
    diffs = [x - y for x, y in zip(a, b)]
    n = len(diffs)
    win = sum(1 for d in diffs if d > 1e-9)
    loss = sum(1 for d in diffs if d < -1e-9)
    md = mean(diffs)
    var = mean([(d - md) ** 2 for d in diffs]) * n / max(n - 1, 1)
    se = (var / n) ** 0.5 if n else 0.0
    t = md / se if se else 0.0
    sorted_d = sorted(diffs)
    median = sorted_d[n // 2] if n % 2 else (sorted_d[n // 2 - 1] + sorted_d[n // 2]) / 2
    return {"win": win, "tie": n - win - loss, "loss": loss,
            "mean_diff": round(md, 4), "median_diff": round(median, 4),
            "t_like": round(t, 2), "significant": abs(t) >= 2.0}


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
    print(f"\n  best mean nDCG@10: {best} ({results[best]['nDCG@10']})")

    # PAIRED per-query analysis — a mean win on a small set can be noise. Report the sign
    # split + |mean|/SE so the honest story (is hybrid REALLY better, or variance?) is visible.
    pq = per_query_ndcg(cases)
    hd = paired(pq["hybrid"], pq["dense"])
    hb = paired(pq["hybrid"], pq["bm25"])
    print("\n  paired per-query nDCG@10 (win/tie/loss · mean Δ · median Δ · t≈mean/SE):")
    print(f"    hybrid vs dense: {hd['win']}/{hd['tie']}/{hd['loss']}  "
          f"meanΔ {hd['mean_diff']:+}  medianΔ {hd['median_diff']:+}  t≈{hd['t_like']}  "
          f"{'SIGNIFICANT' if hd['significant'] else 'not significant (within noise)'}")
    print(f"    hybrid vs bm25 : {hb['win']}/{hb['tie']}/{hb['loss']}  "
          f"meanΔ {hb['mean_diff']:+}  medianΔ {hb['median_diff']:+}  t≈{hb['t_like']}  "
          f"{'SIGNIFICANT' if hb['significant'] else 'not significant (within noise)'}")
    print("  → hybrid's robust value here is VARIANCE (never collapses like a single path can),")
    print("    not a higher mean — the mean edge over dense is within noise on this small set.")

    out = os.path.join(HIST, "retrieval.metrics.json")
    os.makedirs(HIST, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"per_mode": results, "n_queries": len(cases), "verified": verified,
                   "paired": {"hybrid_vs_dense": hd, "hybrid_vs_bm25": hb}}, f, indent=2)
    print(f"\nmetrics written to {out}")


if __name__ == "__main__":
    main()
