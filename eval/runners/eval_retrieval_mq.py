"""Multi-query retrieval eval — does query rewrite + RRF fusion lift recall/nDCG?

Mirrors src/search/multiQuery.ts (the shipped logic) in Python so the eval measures the
SAME thing the search skill does:
  - expandQuery: LLM rewrites the query into n-1 same-meaning variants (+ the original).
  - per-variant hybrid_search at the same depth as baseline (fair budget).
  - rrfFuse: identical formula  score += 1/(k + rank + 1), k=60.

Compares, on eval/datasets/retrieval.jsonl:
  - baseline : single hybrid_search (what deterministic mode ships)
  - multiquery: expand -> retrieve each -> RRF-fuse -> top-10 (what auto mode does when on)

Variants are cached (eval/history/retrieval_mq_cache.json) so reruns don't re-call the LLM.
Needs Qdrant up (listings indexed) + LLM_API_KEY (expansion). Not part of `make eval`.

  python eval/runners/eval_retrieval_mq.py
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "retrieval"))
sys.path.insert(0, os.path.join(ROOT, "eval", "metrics"))
from search import hybrid_search  # noqa: E402
from llm import chat, llm_available  # noqa: E402
from ir import ndcg_at_k, recall_at_k, mrr, precision_at_k, mean  # noqa: E402

DATA = os.path.join(ROOT, "eval", "datasets", "retrieval.jsonl")
HIST = os.path.join(ROOT, "eval", "history")
CACHE = os.path.join(HIST, "retrieval_mq_cache.json")
K = 10          # metric cutoff (matches eval_retrieval.py)
N_VARIANTS = 3  # original + 2 rewrites (matches multiQuery.ts default n=3)
RRF_K = 60      # matches multiQuery.ts rrfFuse default


def load():
    with open(DATA) as f:
        return [json.loads(l) for l in f if l.strip()]


def load_cache():
    if os.path.exists(CACHE):
        with open(CACHE) as f:
            return json.load(f)
    return {}


def expand_query(query, n=N_VARIANTS):
    """Mirror of multiQuery.ts expandQuery: LLM -> JSON array of n-1 same-meaning variants."""
    original = query.strip()
    out = chat(
        f"Rewrite the real-estate search phrase into {n - 1} alternative phrasings that mean "
        f"the SAME thing but use different words/angles (synonyms, related styles/features). "
        f"Return ONLY a JSON array of strings.\n\nPhrase: {original}",
        system=None,
    ) or ""
    m = re.search(r"\[[\s\S]*\]", out)
    try:
        arr = json.loads(m.group(0)) if m else []
    except Exception:
        arr = []
    variants = [s.strip() for s in arr if isinstance(s, str) and s.strip()][: n - 1]
    seen, dedup = set(), []
    for q in [original, *variants]:
        if q not in seen:
            seen.add(q)
            dedup.append(q)
    return dedup


def rrf_fuse(lists, k=RRF_K, weights=None):
    """Mirror of multiQuery.ts rrfFuse: fuse by rank, score += w/(k+rank+1).
    weights (optional) lets us up-weight the original query vs paraphrases."""
    score = {}
    for i, lst in enumerate(lists):
        w = weights[i] if weights else 1.0
        for rank, pid in enumerate(lst):
            score[pid] = score.get(pid, 0.0) + w / (k + rank + 1)
    return [pid for pid, _ in sorted(score.items(), key=lambda x: -x[1])]


def ranked_baseline(query):
    return [p.id for p in hybrid_search(query, None, k=K, mode="hybrid")]


def ranked_multiquery(variants, depth=K, weights=None):
    # retrieve `depth` per variant (deeper pool -> fusion has more material), fuse, cut to K
    lists = [[p.id for p in hybrid_search(v, None, k=depth, mode="hybrid")] for v in variants]
    return rrf_fuse(lists, weights=weights)[:K]


def score(cases, ranked_of):
    ndcg, rec, rr, prec = [], [], [], []
    for c in cases:
        rel = c["label"]["relevant"]
        if not rel:
            continue
        ranked = ranked_of(c)
        ndcg.append(ndcg_at_k(ranked, rel, K))
        rec.append(recall_at_k(ranked, rel, K))
        rr.append(mrr(ranked, rel))
        prec.append(precision_at_k(ranked, rel, 5))
    return {"nDCG@10": round(mean(ndcg), 4), "recall@10": round(mean(rec), 4),
            "MRR": round(mean(rr), 4), "precision@5": round(mean(prec), 4), "n": len(ndcg)}


def main():
    if not llm_available():
        print("LLM_API_KEY not set — multi-query needs the LLM to rewrite. Aborting.")
        sys.exit(1)

    cases = [c for c in load() if c["label"]["relevant"]]
    cache = load_cache()

    # expand + cache variants for every query
    n_gen = 0
    for c in cases:
        q = c["input"]
        if q not in cache:
            cache[q] = expand_query(q)
            n_gen += 1
    os.makedirs(HIST, exist_ok=True)
    with open(CACHE, "w") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

    avg_variants = round(mean([len(cache[c["input"]]) for c in cases]), 2)
    print(f"multi-query retrieval eval: {len(cases)} queries, avg {avg_variants} variants/query "
          f"({n_gen} newly expanded), k={K}\n")

    def orig_weights(c):
        # original query weighted 2x vs paraphrases (it's the most on-target for the gold set)
        v = cache[c["input"]]
        return [2.0] + [1.0] * (len(v) - 1)

    configs = [
        ("baseline",        lambda c: ranked_baseline(c["input"])),
        ("mq depth10",      lambda c: ranked_multiquery(cache[c["input"]], depth=10)),
        ("mq depth30",      lambda c: ranked_multiquery(cache[c["input"]], depth=30)),
        ("mq d30 orig2x",   lambda c: ranked_multiquery(cache[c["input"]], depth=30, weights=orig_weights(c))),
    ]
    scored = {name: score(cases, fn) for name, fn in configs}
    base = scored["baseline"]

    hdr = f"  {'config':14} {'nDCG@10':>9} {'recall@10':>10} {'MRR':>7} {'P@5':>7}   {'ΔnDCG':>7} {'Δrecall':>8}"
    print(hdr)
    for name, m in scored.items():
        dn = "" if name == "baseline" else f"{m['nDCG@10']-base['nDCG@10']:>+7.4f}"
        dr = "" if name == "baseline" else f"{m['recall@10']-base['recall@10']:>+8.4f}"
        print(f"  {name:14} {m['nDCG@10']:>9} {m['recall@10']:>10} {m['MRR']:>7} {m['precision@5']:>7}   {dn:>7} {dr:>8}")

    best_mq = max((n for n in scored if n != "baseline"), key=lambda n: scored[n]["nDCG@10"])
    mq = scored[best_mq]
    print(f"\n  best multi-query config: {best_mq}")
    d_ndcg = round(mq["nDCG@10"] - base["nDCG@10"], 4)
    d_rec = round(mq["recall@10"] - base["recall@10"], 4)
    print(f"\n  Δ nDCG@10: {d_ndcg:+}   Δ recall@10: {d_rec:+}")
    if d_rec > 0 or d_ndcg > 0:
        print("  → multi-query helps (variants surface relevant items a single phrasing misses).")
    else:
        print("  → multi-query does NOT help here → keep it OFF (single phrasing already covers "
              "these queries; N× cost unjustified).")

    with open(os.path.join(HIST, "retrieval_mq.metrics.json"), "w") as f:
        json.dump({"k": K, "n_queries": len(cases), "avg_variants": avg_variants,
                   "configs": scored, "best_multiquery": best_mq,
                   "delta_best": {"nDCG@10": d_ndcg, "recall@10": d_rec}}, f, indent=2)
    print(f"\nmetrics written to {os.path.join(HIST, 'retrieval_mq.metrics.json')}")


if __name__ == "__main__":
    main()
