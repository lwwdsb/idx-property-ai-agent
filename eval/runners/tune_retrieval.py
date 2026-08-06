"""Data-driven tuning of the retrieval hyperparameters, using the labeled set.

Turns "pitched" values into "tuned + justified":
1. prefetch depth — a REAL knob in the production path (hybrid_search prefetch=). Swept
   by nDCG@10; pick the best actionable value.
2. RRF k — production uses Qdrant's native RRF (k fixed internally, not API-exposed).
   Here we reimplement RRF over the dense/bm25 rankings and sweep k to SHOW nDCG is
   nearly flat across k — i.e. RRF is k-insensitive, so the library default is fine.
   This is the data-backed answer to "how did you pick k / did you tune it".

  python eval/runners/tune_retrieval.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "retrieval"))
sys.path.insert(0, os.path.join(ROOT, "eval", "metrics"))
from search import hybrid_search  # noqa: E402
from ir import ndcg_at_k, mean  # noqa: E402

DATA = os.path.join(ROOT, "eval", "datasets", "retrieval.jsonl")
HIST = os.path.join(ROOT, "eval", "history")
PREFETCH_GRID = [10, 20, 30, 50, 100]
K_GRID = [1, 10, 30, 60, 100, 200, 500]
POOL_N = 30   # depth of each single-path ranking used for the manual RRF k-sweep


def load():
    with open(DATA) as f:
        cases = [json.loads(l) for l in f if l.strip()]
    return [c for c in cases if c["label"]["relevant"]]


def rrf_fuse(rankings, k):
    """Reciprocal Rank Fusion: score(id) = sum over lists of 1/(k+rank)."""
    scores = {}
    for ranking in rankings:
        for rank, i in enumerate(ranking, start=1):
            scores[i] = scores.get(i, 0.0) + 1.0 / (k + rank)
    return [i for i, _ in sorted(scores.items(), key=lambda kv: -kv[1])]


def sweep_prefetch(cases):
    out = {}
    for pf in PREFETCH_GRID:
        nd = [ndcg_at_k([p.id for p in hybrid_search(c["input"], None, k=10, mode="hybrid", prefetch=pf)],
                        c["label"]["relevant"], 10) for c in cases]
        out[pf] = round(mean(nd), 4)
    return out


def sweep_k(cases):
    # precompute single-path rankings once per query, then fuse at each k
    paths = []
    for c in cases:
        dense = [p.id for p in hybrid_search(c["input"], None, k=POOL_N, mode="dense")]
        bm25 = [p.id for p in hybrid_search(c["input"], None, k=POOL_N, mode="bm25")]
        paths.append((dense, bm25, c["label"]["relevant"]))
    out = {}
    for k in K_GRID:
        nd = [ndcg_at_k(rrf_fuse([d, b], k)[:10], rel, 10) for d, b, rel in paths]
        out[k] = round(mean(nd), 4)
    return out


def main():
    cases = load()
    print(f"tuning on {len(cases)} labeled queries\n")

    pf = sweep_prefetch(cases)
    best_pf = max(pf, key=pf.get)
    print("prefetch sweep (nDCG@10):")
    for p, v in pf.items():
        print(f"  prefetch={p:<4} {v}{'   <- best' if p == best_pf else ''}")

    ks = sweep_k(cases)
    spread = round(max(ks.values()) - min(ks.values()), 4)
    print("\nRRF k sweep (manual RRF, nDCG@10):")
    for k, v in ks.items():
        print(f"  k={k:<4} {v}")
    print(f"  spread across k = {spread}  → RRF is {'k-insensitive (default fine)' if spread < 0.02 else 'k-sensitive (tune it)'}")

    result = {"prefetch": pf, "best_prefetch": best_pf, "rrf_k": ks, "k_spread": spread,
              "n_queries": len(cases)}
    os.makedirs(HIST, exist_ok=True)
    with open(os.path.join(HIST, "tuning.metrics.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nwritten to {os.path.join(HIST, 'tuning.metrics.json')}")


if __name__ == "__main__":
    main()
