"""Hybrid retrieval: dense (semantic) + BM25 (keyword) multi-path recall, fused
with Reciprocal Rank Fusion (RRF), with hard filters applied inside Qdrant.

  python retrieval/search.py "craftsman home with a big backyard" --city Irvine --max-price 2500000 --min-beds 3
  python retrieval/search.py "solar panels and ocean view" --mode dense   # compare single-path
"""
import argparse

from qdrant_client import models

from common import COLLECTION, get_dense, get_qdrant, get_sparse


def build_filter(city=None, max_price=None, min_price=None, min_beds=None, pool=None, ptype=None):
    must = []
    if city:
        must.append(models.FieldCondition(key="city", match=models.MatchValue(value=city)))
    if max_price is not None or min_price is not None:
        must.append(models.FieldCondition(key="price", range=models.Range(gte=min_price, lte=max_price)))
    if min_beds is not None:
        must.append(models.FieldCondition(key="beds", range=models.Range(gte=min_beds)))
    if pool is not None:  # tri-state: True=has pool, False=no pool, None=don't care
        must.append(models.FieldCondition(key="pool", match=models.MatchValue(value=bool(pool))))
    if ptype:
        must.append(models.FieldCondition(key="type", match=models.MatchValue(value=ptype)))
    return models.Filter(must=must) if must else None


def hybrid_search(text, flt=None, k=5, mode="hybrid", prefetch=30):
    client, dense, sparse = get_qdrant(), get_dense(), get_sparse()
    dvec = list(dense.embed([text]))[0].tolist()
    s = list(sparse.embed([text]))[0]
    svec = models.SparseVector(indices=s.indices.tolist(), values=s.values.tolist())

    if mode == "dense":
        res = client.query_points(COLLECTION, query=dvec, using="dense", query_filter=flt, limit=k, with_payload=True)
    elif mode == "bm25":
        res = client.query_points(COLLECTION, query=svec, using="bm25", query_filter=flt, limit=k, with_payload=True)
    else:  # hybrid: two paths + RRF fusion
        res = client.query_points(
            COLLECTION,
            prefetch=[
                models.Prefetch(query=dvec, using="dense", limit=prefetch, filter=flt),
                models.Prefetch(query=svec, using="bm25", limit=prefetch, filter=flt),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=k, with_payload=True,
        )
    return res.points


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text")
    ap.add_argument("--city")
    ap.add_argument("--max-price", type=float)
    ap.add_argument("--min-price", type=float)
    ap.add_argument("--min-beds", type=float)
    ap.add_argument("--pool", action="store_const", const=True, default=None)
    ap.add_argument("--no-pool", dest="pool", action="store_const", const=False)
    ap.add_argument("--type", dest="ptype")
    ap.add_argument("--mode", default="hybrid", choices=["hybrid", "dense", "bm25"])
    ap.add_argument("-k", type=int, default=5)
    args = ap.parse_args()

    flt = build_filter(args.city, args.max_price, args.min_price, args.min_beds, args.pool, args.ptype)
    pts = hybrid_search(args.text, flt, k=args.k, mode=args.mode)
    print(f'\n"{args.text}"  [mode={args.mode}]  {len(pts)} results\n')
    for p in pts:
        pl = p.payload
        price = f"${int(pl['price']):,}" if pl.get("price") else "N/A"
        print(f"  {p.score:.4f}  {pl.get('address')}, {pl.get('city')} — {price} "
              f"· {pl.get('beds')}bd/{pl.get('baths')}ba · {pl.get('type')}"
              f"{' · pool' if pl.get('pool') else ''}")


if __name__ == "__main__":
    main()
