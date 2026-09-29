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
import re
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


def strip_city(text, city):
    """Production does NOT send the raw query to /search.

    The TS search skill calls extractSemanticText(message, filter), whose first act is to
    remove the matched city from the text (skills.ts) — the city travels as a hard FILTER, and
    what goes to the semantic side is the residue. Sending the raw query here made the A/B
    measure a path that does not exist: with the city removed from the document text but still
    present in the query, the query vector points partly at a direction no document has, and
    known-item recall fell by ~0.18 on the human subset. That was an artifact of the harness,
    not of the index. Only the city is reproduced here, because the city is the variable under
    test; the rest of extractSemanticText's stripping is identical across both collections and
    so cannot bias the comparison.
    """
    if not city:
        return text
    out = re.sub(re.escape(city), " ", text, flags=re.I)
    return re.sub(r"\s+", " ", out).strip() or text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--tag", default="")
    ap.add_argument("--raw", action="store_true",
                    help="send the raw query instead of the residue (NOT production)")
    ap.add_argument("--semantic", choices=["translated", "untranslated", "strip"], default="translated",
                    help="哪一条语义通道。translated=生产现状(extractSemanticText + 中文翻译);"
                         "untranslated=同上但跳过翻译(用来隔离翻译的效果);"
                         "strip=这个 runner 旧的近似(原始 query 减去城市)")
    args = ap.parse_args()

    with urllib.request.urlopen(f"{URL}/health", timeout=10) as r:
        health = json.loads(r.read())
    print(f"service: {URL}  {health}")

    sets = {"human": load("mode_retrieval.jsonl"), "generated": load("mode_retrieval_gen.jsonl")}

    # 预计算的【生产】语义文本,按 id 取。这个 runner 过去自己做 city 剥离,而生产跑的是
    # extractSemanticText(先归一化、再剥城市/数字单位/房型/泳池词/填充词)外加中文残余翻译 ——
    # 自己近似一遍就是又测了一条生产不走的路,和"把原始 query 直送 /search"是同一个错误。
    SEM = {"human": os.path.join(HIST, "mode_retrieval.semantic.jsonl"),
           "generated": os.path.join(HIST, "mode_retrieval_gen.preds.jsonl")}
    sem = {}
    for label, path in SEM.items():
        if os.path.exists(path):
            sem[label] = {r["id"]: r for r in (json.loads(l) for l in open(path, encoding="utf-8") if l.strip())}
        else:
            sem[label] = {}
            if args.semantic != "strip":
                print(f"⚠️  缺 {os.path.relpath(path, ROOT)} —— {label} 会退回 strip 近似。"
                      f"先跑 npx tsx eval/runners/genSemanticPreds.ts")
    out = {"tag": args.tag, "k": args.k, "health": health, "subsets": {}}
    for label, cases in sets.items():
        cases = [c for c in cases if (c.get("gold") or {}).get("known_item")]
        if not cases:
            continue
        hits, n, empty, dupes = 0, 0, 0, 0
        per_case = {}   # id -> hit/miss, so two runs can be compared PAIRED rather than as
                        # two point estimates — the whole discipline of the loop's gate
        per_lang = defaultdict(lambda: [0, 0])
        for i, c in enumerate(cases, 1):
            f = c["gold"].get("filter") or {}
            pre = sem.get(label, {}).get(c["id"])
            if args.raw:
                text = c["input"]
            elif args.semantic == "strip" or not pre:
                text = strip_city(c["input"], f.get("city"))
            else:
                key = "semantic" if args.semantic == "translated" else "raw"
                # 空残余在生产里根本不进 Qdrant(走 MySQL 结构化)。这里仍然要发一次请求才能
                # 算召回,所以退回城市剥离,并在输出里单独计数 —— 否则这些用例会被静默当成
                # "语义检索的成绩",而生产从没让语义检索碰过它们。
                text = pre.get(key) or strip_city(c["input"], f.get("city"))
            body = {"text": text, "k": args.k,
                    "city": f.get("city"), "max_price": f.get("maxPrice"),
                    "min_price": f.get("minPrice"), "min_beds": f.get("beds"),
                    "pool": f.get("pool"),
                    "ptype": TYPE_DB.get(f.get("propertyType")) if f.get("propertyType") else None}
            res = post("/search", body).get("results", [])
            ids = [r.get("listing_id") for r in res]
            hit = int(c["gold"]["known_item"] in ids)
            hits += hit
            n += 1
            per_case[c["id"]] = hit
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
            "by_lang": {k: round(v[0] / v[1], 4) for k, v in per_lang.items()},
            "per_case": per_case}
        s = out["subsets"][label]
        print(f"  {label:10} recall@{args.k} {s['recall_at_k']} ({hits}/{n})"
              f"  dup-in-results {dupes}  short-results {empty}  by-lang {s['by_lang']}")

    name = f"knownitem_service{('_' + args.tag) if args.tag else ''}.metrics.json"
    with open(os.path.join(HIST, name), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(f"\nwrote eval/history/{name}")


if __name__ == "__main__":
    main()
