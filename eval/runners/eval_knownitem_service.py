"""Known-item recall through the PRODUCTION path (/search), not the library call.

compare_sources.py calls hybrid_search directly, which is right for comparing retrieval modes
in isolation — but it bypasses everything the service adds: the cross-encoder rerank and the
result-level dedupe. A change to either is invisible to it. This runner asks the same objective
question of the endpoint a user's query actually reaches.

Switch index by restarting the service with QDRANT_COLLECTION=<name>; this script reports
whatever /health says it is serving, so an A/B cannot silently compare a collection to itself.

  python eval/runners/eval_knownitem_service.py [--k 10] [--tag before-reindex]
"""
import argparse
import json
import os
import sys
import urllib.request
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "eval", "datasets")
HIST = os.path.join(ROOT, "eval", "history")
URL = os.environ.get("RETRIEVAL_URL", "http://localhost:8099")
TYPE_DB = {"condo": "Condominium", "townhouse": "Townhouse", "single-family": "SingleFamilyResidence"}


def post(path, body, timeout=120):
    req = urllib.request.Request(f"{URL}{path}", method="POST",
                                data=json.dumps(body).encode(),
                                headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def load(name):
    p = os.path.join(DATA, name)
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if os.path.exists(p) else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    with urllib.request.urlopen(f"{URL}/health", timeout=10) as r:
        health = json.loads(r.read())
    print(f"service: {URL}  {health}")

    sets = {"human": load("mode_retrieval.jsonl"), "generated": load("mode_retrieval_gen.jsonl")}
    out = {"tag": args.tag, "k": args.k, "health": health, "subsets": {}}
    for label, cases in sets.items():
        cases = [c for c in cases if (c.get("gold") or {}).get("known_item")]
        if not cases:
            continue
        hits, n, empty, dupes = 0, 0, 0, 0
        per_lang = defaultdict(lambda: [0, 0])
        for i, c in enumerate(cases, 1):
            f = c["gold"].get("filter") or {}
            body = {"text": c["input"], "k": args.k,
                    "city": f.get("city"), "max_price": f.get("maxPrice"),
                    "min_price": f.get("minPrice"), "min_beds": f.get("beds"),
                    "pool": f.get("pool"),
                    "ptype": TYPE_DB.get(f.get("propertyType")) if f.get("propertyType") else None}
            res = post("/search", body).get("results", [])
            ids = [r.get("listing_id") for r in res]
            hit = int(c["gold"]["known_item"] in ids)
            hits += hit
            n += 1
            per_lang[c.get("lang", "?")][0] += hit
            per_lang[c.get("lang", "?")][1] += 1
            # quality signals a recall number cannot show
            if len(set((str(r.get("address")).lower(), r.get("price")) for r in res)) < len(res):
                dupes += 1
            if len(res) < args.k:
                empty += 1
            if i % 25 == 0:
                print(f"  {label}: {i}/{len(cases)}", flush=True)
        out["subsets"][label] = {
            "recall_at_k": round(hits / n, 4), "hits": hits, "n": n,
            "queries_with_duplicate_results": dupes,
            "queries_returning_fewer_than_k": empty,
            "by_lang": {k: round(v[0] / v[1], 4) for k, v in per_lang.items()}}
        s = out["subsets"][label]
        print(f"  {label:10} recall@{args.k} {s['recall_at_k']} ({hits}/{n})"
              f"  dup-in-results {dupes}  short-results {empty}  by-lang {s['by_lang']}")

    name = f"knownitem_service{('_' + args.tag) if args.tag else ''}.metrics.json"
    with open(os.path.join(HIST, name), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(f"\nwrote eval/history/{name}")


if __name__ == "__main__":
    main()
