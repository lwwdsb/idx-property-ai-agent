"""Mode-retrieval scorer (Python side): deterministic(regex) vs auto(LLM) extraction, then
their effect on retrieval — using KNOWN-ITEM recall (objective gold, no LLM judge).

Reads mode_retrieval.jsonl (gold) + mode_retrieval.preds.jsonl (each mode's extracted filter,
from evalModeRetrieval.ts). Computes:
  1. param-extraction P/R/F1 on structured slots (regex vs auto vs gold) + over-extraction rate;
  2. known-item recall@5/10/20: run hybrid_search with each mode's filter, is the gold target in top-k;
  3. light intent-recall: auto's chosen tools vs gold intents.
All stratified by style. Needs Qdrant up. Run: python eval/runners/eval_mode_retrieval.py
"""
import json
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "retrieval"))
from search import hybrid_search, build_filter  # noqa: E402

DATA = os.path.join(ROOT, "eval", "datasets", "mode_retrieval.jsonl")
PREDS = os.path.join(ROOT, "eval", "history", "mode_retrieval.preds.jsonl")
OUT = os.path.join(ROOT, "eval", "history", "mode_retrieval.metrics.json")
SLOTS = ["city", "beds", "baths", "maxPrice", "minPrice", "propertyType", "pool", "minSqft"]
TYPE_DB = {"condo": "Condominium", "townhouse": "Townhouse", "single-family": "SingleFamilyResidence"}


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def prf(pred: dict, gold: dict):
    """TP/FP/FN over structured slots (exact value match). over = predicted field absent in gold."""
    tp = fp = fn = over = 0
    for k in SLOTS:
        pv, gv = pred.get(k), gold.get(k)
        if gv is not None and pv is not None and pv == gv:
            tp += 1
        elif pv is not None and gv is None:
            fp += 1; over += 1                 # over-extraction: invented a field
        elif pv is not None and gv is not None and pv != gv:
            fp += 1                            # wrong value
        elif gv is not None and pv is None:
            fn += 1                            # missed
    return tp, fp, fn, over


def retrieve_ids(text, flt_fields, k=20):
    flt = build_filter(
        city=flt_fields.get("city"), max_price=flt_fields.get("maxPrice"),
        min_price=flt_fields.get("minPrice"), min_beds=flt_fields.get("beds"),
        pool=flt_fields.get("pool"),
        ptype=TYPE_DB.get(flt_fields.get("propertyType")) if flt_fields.get("propertyType") else None)
    return [p.id for p in hybrid_search(text, flt, k=k, mode="hybrid")]


def main():
    gold = {c["id"]: c for c in load(DATA)}
    preds = {p["id"]: p for p in load(PREDS)}
    modes = {"regex": "regex_filter", "auto": "auto_filter"}

    # 1. param-extraction F1 (overall + by style)
    agg = {m: defaultdict(lambda: [0, 0, 0, 0]) for m in modes}   # style -> [tp,fp,fn,over]
    for cid, c in gold.items():
        gf = c["gold"]["filter"]
        for m, key in modes.items():
            tp, fp, fn, over = prf(preds[cid][key], gf)
            for bucket in (c["style"], "ALL"):
                a = agg[m][bucket]; a[0] += tp; a[1] += fp; a[2] += fn; a[3] += over

    def f1(tp, fp, fn):
        p = tp / (tp + fp) if tp + fp else 1.0
        r = tp / (tp + fn) if tp + fn else 1.0
        return p, r, (2 * p * r / (p + r) if p + r else 0.0)

    print("=== 1. PARAM EXTRACTION (structured slots, exact match) ===")
    print(f"{'stratum':14} {'mode':6} {'P':>5} {'R':>5} {'F1':>5} {'over':>5}")
    for bucket in ["clean", "messy", "multi-intent", "ood-probe", "ALL"]:
        for m in modes:
            tp, fp, fn, over = agg[m][bucket]
            if tp + fp + fn == 0:
                continue
            p, r, f = f1(tp, fp, fn)
            print(f"{bucket:14} {m:6} {p:>5.2f} {r:>5.2f} {f:>5.2f} {over:>5}")
        print()

    # 2. known-item retrieval recall (objective; only cases with a target)
    print("=== 2. KNOWN-ITEM RECALL (hybrid_search with each mode's filter) ===")
    rec = {m: defaultdict(lambda: [0, 0, 0, 0]) for m in modes}   # style -> [hit@5,hit@10,hit@20,n]
    diverge = []
    for cid, c in gold.items():
        ki = c["gold"].get("known_item")
        if not ki:
            continue
        hits = {}
        for m, key in modes.items():
            ids = retrieve_ids(c["input"], preds[cid][key], k=20)
            h5, h10, h20 = int(ki in ids[:5]), int(ki in ids[:10]), int(ki in ids[:20])
            hits[m] = h10
            for bucket in (c["style"], "ALL"):
                a = rec[m][bucket]; a[0] += h5; a[1] += h10; a[2] += h20; a[3] += 1
        if hits["regex"] != hits["auto"]:
            diverge.append((cid, c["style"], hits["regex"], hits["auto"]))

    print(f"{'stratum':14} {'mode':6} {'R@5':>6} {'R@10':>6} {'R@20':>6} {'n':>3}")
    for bucket in ["clean", "messy", "multi-intent", "ALL"]:
        for m in modes:
            a = rec[m][bucket]
            if a[3] == 0:
                continue
            print(f"{bucket:14} {m:6} {a[0]/a[3]:>6.2f} {a[1]/a[3]:>6.2f} {a[2]/a[3]:>6.2f} {a[3]:>3}")
        print()
    print("per-case divergence (recall@10, regex vs auto):")
    for cid, st, rgx, at in diverge:
        who = "auto wins" if at > rgx else "regex wins"
        print(f"  {cid} [{st}]: regex={rgx} auto={at}  <- {who}")

    # 3. light intent-recall: auto tools vs gold intents (search/market/email; unknown=no tool)
    print("\n=== 3. INTENT/TOOL SELECTION (auto only, single-turn) ===")
    ok = 0; total = 0
    for cid, c in gold.items():
        gi = set(c["gold"]["intents"]); at = set(preds[cid].get("auto_tools", []))
        if "unknown" in gi:
            hit = len(at) == 0
        else:
            hit = gi.issubset(at)          # recall: did auto call all required tools
        ok += int(hit); total += 1
        mark = "" if hit else "  <- MISS"
        print(f"  {cid} gold={sorted(gi)} auto_tools={sorted(at)}{mark}")
    print(f"intent/tool recall (all required tools called / no tool on OOD): {ok}/{total}")

    json.dump({"note": "known-item recall + param F1, det(regex) vs auto(LLM)"}, open(OUT, "w"))
    print(f"\nmetrics stub -> {OUT}")


if __name__ == "__main__":
    main()
