"""Retrieval benchmark: dense vs BM25 vs hybrid.

Measures per-mode latency and the *complementarity* of the two paths — how little
their top-K overlap (low Jaccard) — which is the quantitative justification for
multi-path recall + RRF fusion: each path finds things the other misses.

  python retrieval/benchmark.py
"""
import time

from common import COLLECTION, get_qdrant
from search import hybrid_search

QUERIES = [
    "cozy craftsman bungalow with character",
    "modern home with open floor plan and natural light",
    "quiet cul-de-sac family home with a big backyard",
    "fixer upper with lots of potential",
    "solar panels and energy efficient",
    "granite countertops and stainless steel appliances",
    "ADU guest house or in-law suite",
    "RV parking and a three car garage",
    "ocean view and walk to the beach",
    "remodeled kitchen and updated bathrooms",
]

K = 10


def ids(points):
    return {p.id for p in points}


def jaccard(a, b):
    return len(a & b) / len(a | b) if (a | b) else 0.0


def timed(fn):
    t = time.perf_counter()
    out = fn()
    return out, (time.perf_counter() - t) * 1000


def main():
    total = get_qdrant().count(COLLECTION).count
    print(f"collection '{COLLECTION}': {total} listings\n")
    lat = {"dense": [], "bm25": [], "hybrid": []}
    overlaps = []

    for q in QUERIES:
        d, td = timed(lambda: hybrid_search(q, k=K, mode="dense"))
        b, tb = timed(lambda: hybrid_search(q, k=K, mode="bm25"))
        h, th = timed(lambda: hybrid_search(q, k=K, mode="hybrid"))
        lat["dense"].append(td); lat["bm25"].append(tb); lat["hybrid"].append(th)
        j = jaccard(ids(d), ids(b))
        overlaps.append(j)
        # how many hybrid results were NOT in dense-topK (i.e. contributed by BM25)
        from_bm25_only = len(ids(h) - ids(d))
        print(f'"{q[:44]:44}"  dense∩bm25 overlap={j:.2f}  hybrid picked {from_bm25_only}/{K} that dense missed')

    def stats(xs):
        xs = sorted(xs); n = len(xs)
        return f"avg {sum(xs)/n:.1f}ms  p50 {xs[n//2]:.1f}ms  p95 {xs[min(n-1,int(n*0.95))]:.1f}ms"
    print("\nLatency:")
    for m in ("dense", "bm25", "hybrid"):
        print(f"  {m:7} {stats(lat[m])}")
    print(f"\nMean dense/BM25 top-{K} overlap: {sum(overlaps)/len(overlaps):.2f} "
          f"(low overlap => the two paths are complementary => fusion helps)")


if __name__ == "__main__":
    main()
