"""A1 ACCEPTANCE TEST — does the generated known-item set support the same conclusions as
the human-written one?

Row count is not the acceptance criterion. An eval set exists to rank configurations, so the
question is whether HUMAN rows and GENERATED rows put the retrieval modes in the SAME ORDER.
If they disagree on direction, the generator is a confound and the generated rows must not be
used to pick parameters — exactly the failure already found on the judge side, where swapping
only the labels flipped dense against bm25.

Both subsets are scored with the GOLD filter and the raw query text, so the parser is held
fixed and only retrieval varies.

  python eval/runners/compare_sources.py            (needs Qdrant + the retrieval service)
"""
import json
import math
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "retrieval"))
from search import hybrid_search, build_filter      # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "eval", "datasets")
HIST = os.path.join(ROOT, "eval", "history")
MODES = ("dense", "bm25", "hybrid")
TYPE_DB = {"condo": "Condominium", "townhouse": "Townhouse", "single-family": "SingleFamilyResidence"}


def load(name):
    path = os.path.join(DATA, name)
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def recall_at(case, mode, k=10):
    g = case["gold"]
    f = g.get("filter") or {}
    flt = build_filter(f.get("city"), f.get("maxPrice"), f.get("minPrice"),
                       f.get("beds"), f.get("pool"),
                       TYPE_DB.get(f.get("propertyType")) if f.get("propertyType") else None)
    ids = [p.id for p in hybrid_search(case["input"], flt, k=k, mode=mode)]
    return int(g["known_item"] in ids)


def wilson(hits, n):
    """95% CI on a proportion. A difference between two modes means nothing next to the width
    of these intervals, which is the point of printing them."""
    if not n:
        return (0.0, 0.0)
    p, z = hits / n, 1.96
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - m) / d), min(1.0, (c + m) / d))


def main():
    human = [c for c in load("mode_retrieval.jsonl") if (c["gold"] or {}).get("known_item")]
    gen = [c for c in load("mode_retrieval_gen.jsonl") if (c["gold"] or {}).get("known_item")]
    if not gen:
        sys.exit("no generated rows — run gen_knownitem_set.py first")
    print(f"human {len(human)} rows | generated {len(gen)} rows\n")

    res = {}
    for label, cases in (("human", human), ("generated", gen)):
        per = defaultdict(int)
        for i, c in enumerate(cases, 1):
            for m in MODES:
                per[m] += recall_at(c, m)
            if i % 25 == 0:
                print(f"  {label}: {i}/{len(cases)}", flush=True)
        res[label] = {m: {"hits": per[m], "n": len(cases), "recall": round(per[m] / len(cases), 4),
                          "ci95": [round(x, 4) for x in wilson(per[m], len(cases))]}
                      for m in MODES}

    print(f"\n{'subset':11} {'mode':8} {'recall@10':>10} {'95% CI':>18}  order")
    orders = {}
    for label in ("human", "generated"):
        rank = sorted(MODES, key=lambda m: -res[label][m]["recall"])
        orders[label] = rank
        for m in MODES:
            r = res[label][m]
            mark = "  <- best" if m == rank[0] else ""
            print(f"{label:11} {m:8} {r['recall']:>10.4f} "
                  f"[{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]{mark}")
    agree_best = orders["human"][0] == orders["generated"][0]
    agree_full = orders["human"] == orders["generated"]
    print(f"\nhuman order     : {' > '.join(orders['human'])}")
    print(f"generated order : {' > '.join(orders['generated'])}")
    print(f"\nACCEPTANCE: best mode agrees = {agree_best} | full order agrees = {agree_full}")
    if not agree_best:
        print("  -> DO NOT use the generated rows to pick retrieval parameters: they rank the\n"
              "     modes differently than human-written queries, i.e. the generator is a\n"
              "     confound in the same way the LLM judge was.")
    elif not agree_full:
        print("  -> Best mode agrees but the tail order differs; usable for the primary\n"
              "     comparison, not for fine distinctions between the weaker modes.")
    else:
        print("  -> Generated rows support the same conclusion as human rows; usable.")

    out = {"human": res["human"], "generated": res["generated"],
           "orders": orders, "agree_best": agree_best, "agree_full": agree_full}
    with open(os.path.join(HIST, "source_agreement.metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("\nwrote eval/history/source_agreement.metrics.json")


if __name__ == "__main__":
    main()
