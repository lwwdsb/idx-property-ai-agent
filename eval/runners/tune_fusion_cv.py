"""Cross-validated fusion tuning — pick RRF k + dense/bm25 weights that are ROBUST,
not overfit to one small set.

Two robustness reports on eval/datasets/retrieval_large.jsonl:
  A. per-config: mean nDCG@10 over all queries + std ACROSS 5 folds (stability).
  B. NESTED 5-fold CV: on each fold pick argmax config on the TRAIN split, score it on the
     held-out TEST split. The averaged test score is an honest generalization estimate
     (params were never tuned on the data they're graded on) + shows if the winner is stable.

Fusion is client-side weighted RRF so k and weights are sweepable (Qdrant native RRF is not).
Per-query dense/bm25 rank lists are cached (needs Qdrant once).

  python eval/runners/tune_fusion_cv.py
"""
import json
import os
import sys
import random
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "retrieval"))
sys.path.insert(0, os.path.join(ROOT, "eval", "metrics"))
from ir import ndcg_at_k, recall_at_k, mean  # noqa: E402

DATA = os.path.join(ROOT, "eval", "datasets", "retrieval_large.jsonl")
HIST = os.path.join(ROOT, "eval", "history")
PATHS = os.path.join(HIST, "retrieval_large_paths.json")
DEPTH = 30
K_GRID = [1, 2, 4, 10]
W_GRID = [(1, 1), (1.5, 1), (2, 1), (3, 1), (1, 1.5), (1, 2)]
FOLDS = 5
SEED = 13


def load():
    return [json.loads(l) for l in open(DATA) if l.strip() and json.loads(l)["label"]["relevant"]]


def get_paths(cases):
    """Per-query dense/bm25 top-DEPTH id lists (cached; Qdrant needed only on first run)."""
    if os.path.exists(PATHS):
        cache = json.load(open(PATHS))
        if all(c["input"] in cache for c in cases):
            return cache
    else:
        cache = {}
    sys.path.insert(0, os.path.join(ROOT, "retrieval"))
    from qdrant_client import models
    from common import COLLECTION, get_qdrant, get_dense, get_sparse
    client, dense, sparse = get_qdrant(), get_dense(), get_sparse()
    for c in cases:
        q = c["input"]
        if q in cache:
            continue
        dv = list(dense.embed([q]))[0].tolist()
        s = list(sparse.embed([q]))[0]
        sv = models.SparseVector(indices=s.indices.tolist(), values=s.values.tolist())
        d = [p.id for p in client.query_points(COLLECTION, query=dv, using="dense", limit=DEPTH, with_payload=False).points]
        b = [p.id for p in client.query_points(COLLECTION, query=sv, using="bm25", limit=DEPTH, with_payload=False).points]
        cache[q] = {"dense": d, "bm25": b}
    os.makedirs(HIST, exist_ok=True)
    json.dump(cache, open(PATHS, "w"))
    return cache


def wrrf(dl, bl, wd, wb, k, topk=10):
    sc = {}
    for r, pid in enumerate(dl):
        sc[pid] = sc.get(pid, 0.0) + wd / (k + r + 1)
    for r, pid in enumerate(bl):
        sc[pid] = sc.get(pid, 0.0) + wb / (k + r + 1)
    return [p for p, _ in sorted(sc.items(), key=lambda x: -x[1])][:topk]


def config_ndcg(cases, paths, cfg):
    """Per-query nDCG@10 for a fusion config ('dense','bm25', or (k,wd,wb))."""
    out = []
    for c in cases:
        p = paths[c["input"]]
        rel = c["label"]["relevant"]
        if cfg == "dense":
            ids = p["dense"][:10]
        elif cfg == "bm25":
            ids = p["bm25"][:10]
        else:
            k, wd, wb = cfg
            ids = wrrf(p["dense"], p["bm25"], wd, wb, k)
        out.append(ndcg_at_k(ids, rel, 10))
    return out


def main():
    cases = load()
    paths = get_paths(cases)
    n = len(cases)
    print(f"CV fusion tuning: {n} queries, {FOLDS}-fold, seed {SEED}\n")

    configs = {"dense": "dense", "bm25": "bm25"}
    for k in K_GRID:
        for wd, wb in W_GRID:
            configs[f"k{k} {wd}:{wb}"] = (k, wd, wb)
    # per-query nDCG for every config (deterministic)
    pq = {name: config_ndcg(cases, paths, cfg) for name, cfg in configs.items()}

    idx = list(range(n))
    random.Random(SEED).shuffle(idx)
    folds = [idx[i::FOLDS] for i in range(FOLDS)]

    # A. per-config: overall mean + std across folds (fold mean = mean over that fold's queries)
    print("A. per-config nDCG@10  (overall mean · std across folds)")
    rows = []
    for name in configs:
        scores = pq[name]
        overall = mean(scores)
        fold_means = [mean([scores[i] for i in f]) for f in folds]
        rows.append((name, overall, st.pstdev(fold_means)))
    for name, overall, sd in sorted(rows, key=lambda r: -r[1]):
        mark = "  <-- best mean" if name == max(rows, key=lambda r: r[1])[0] else ""
        print(f"   {name:12} {overall:.4f}  ±{sd:.4f}{mark}")

    # B. nested CV: pick best FUSION config on train, score on held-out test
    fusion_names = [nm for nm, cfg in configs.items() if cfg not in ("dense", "bm25")]
    test_scores, chosen = [], []
    for t, test in enumerate(folds):
        train = [i for i in idx if i not in set(test)]
        best = max(fusion_names, key=lambda nm: mean([pq[nm][i] for i in train]))
        chosen.append(best)
        test_scores.append(mean([pq[best][i] for i in test]))
    from collections import Counter
    print("\nB. nested 5-fold CV (tune on train, score on held-out test):")
    print(f"   generalization nDCG@10 = {mean(test_scores):.4f}  (±{st.pstdev(test_scores):.4f} across folds)")
    print(f"   config picked per fold : {dict(Counter(chosen))}")
    dense_mean = mean(pq['dense'])
    print(f"   dense baseline         = {dense_mean:.4f}  -> nested-CV lift {mean(test_scores)-dense_mean:+.4f}")

    # robust recommendation = fusion config with best overall mean (report its recall too)
    best_fusion = max(fusion_names, key=lambda nm: mean(pq[nm]))
    k, wd, wb = configs[best_fusion]
    rec = mean([recall_at_k(wrrf(paths[c['input']]['dense'], paths[c['input']]['bm25'], wd, wb, k), c['label']['relevant'], 10) for c in cases])
    print(f"\nrobust pick: {best_fusion}  nDCG@10 {mean(pq[best_fusion]):.4f}  recall@10 {rec:.4f}")

    json.dump({"n": n, "folds": FOLDS,
               "per_config": {nm: round(mean(pq[nm]), 4) for nm in configs},
               "nested_cv_ndcg": round(mean(test_scores), 4),
               "nested_cv_std": round(st.pstdev(test_scores), 4),
               "chosen_per_fold": chosen, "robust_pick": best_fusion,
               "dense_baseline": round(dense_mean, 4)},
              open(os.path.join(HIST, "fusion_cv.metrics.json"), "w"), indent=2)
    print(f"\nmetrics -> {os.path.join(HIST, 'fusion_cv.metrics.json')}")


if __name__ == "__main__":
    main()
