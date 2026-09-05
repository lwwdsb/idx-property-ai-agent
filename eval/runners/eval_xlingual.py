"""Cross-lingual retrieval experiment (#2): does translating the query to English before
embedding lift known-item recall — especially on Chinese queries?

Holds extraction fixed (uses auto's filter from the predictions) and varies ONLY the dense
TEXT fed to hybrid_search: raw query vs LLM-translated-to-English. English queries are the
control (translation ~= identity). Objective known-item recall, no LLM judge.

Translations cached (eval/history/xlingual_cache.json). Needs Qdrant + LLM. Run:
  python eval/runners/eval_xlingual.py
"""
import json
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "retrieval"))
from search import hybrid_search, build_filter  # noqa: E402
from llm import chat, llm_available  # noqa: E402

DATA = os.path.join(ROOT, "eval", "datasets", "mode_retrieval.jsonl")
PREDS = os.path.join(ROOT, "eval", "history", "mode_retrieval.preds.jsonl")
CACHE = os.path.join(ROOT, "eval", "history", "xlingual_cache.json")
TYPE_DB = {"condo": "Condominium", "townhouse": "Townhouse", "single-family": "SingleFamilyResidence"}


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def translate(q, cache):
    if q in cache:
        return cache[q]
    out = chat("Translate this real-estate search query to natural English. Return ONLY the "
               f"translation, nothing else.\n\n{q}") or q
    cache[q] = out.strip()
    return cache[q]


def ids_for(text, flt_fields, k=20):
    flt = build_filter(city=flt_fields.get("city"), max_price=flt_fields.get("maxPrice"),
                       min_price=flt_fields.get("minPrice"), min_beds=flt_fields.get("beds"),
                       pool=flt_fields.get("pool"),
                       ptype=TYPE_DB.get(flt_fields.get("propertyType")) if flt_fields.get("propertyType") else None)
    return [p.id for p in hybrid_search(text, flt, k=k, mode="hybrid")]


def main():
    if not llm_available():
        print("need LLM for translation"); sys.exit(1)
    gold = {c["id"]: c for c in load(DATA)}
    preds = {p["id"]: p for p in load(PREDS)}
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}

    rec = {v: defaultdict(lambda: [0, 0, 0, 0]) for v in ("raw", "translated")}   # lang -> [h5,h10,h20,n]
    lifts = []
    for cid, c in gold.items():
        ki = c["gold"].get("known_item")
        if not ki:
            continue
        flt = preds[cid]["auto_filter"]
        raw_text = c["input"]
        tr_text = translate(raw_text, cache) if c["lang"] != "en" else raw_text
        for variant, text in (("raw", raw_text), ("translated", tr_text)):
            ids = ids_for(text, flt, k=20)
            h5, h10, h20 = int(ki in ids[:5]), int(ki in ids[:10]), int(ki in ids[:20])
            for bucket in (c["lang"], "ALL"):
                a = rec[variant][bucket]; a[0] += h5; a[1] += h10; a[2] += h20; a[3] += 1
        # per-case lift on recall@10
        r10 = int(ki in ids_for(raw_text, flt, 10))
        t10 = int(ki in ids_for(tr_text, flt, 10))
        if r10 != t10:
            lifts.append((cid, c["lang"], r10, t10))

    json.dump(cache, open(CACHE, "w"), ensure_ascii=False, indent=1)

    print("cross-lingual: raw query vs translated-to-EN (extraction held fixed = auto filter)\n")
    print(f"{'lang':6} {'variant':11} {'R@5':>6} {'R@10':>6} {'R@20':>6} {'n':>3}")
    for bucket in ("zh", "en", "ALL"):
        for v in ("raw", "translated"):
            a = rec[v][bucket]
            if a[3] == 0:
                continue
            print(f"{bucket:6} {v:11} {a[0]/a[3]:>6.2f} {a[1]/a[3]:>6.2f} {a[2]/a[3]:>6.2f} {a[3]:>3}")
        print()
    print("per-case recall@10 changes (raw -> translated):")
    for cid, lang, r, t in lifts:
        print(f"  {cid} [{lang}] {r} -> {t}  {'LIFT' if t > r else 'DROP'}")


if __name__ == "__main__":
    main()
