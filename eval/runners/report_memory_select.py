"""Memory selection metrics — 注入准确率(precision) / 召回准确率(recall).

Gold is human-labelled and the metric is set overlap, so nothing here depends on an LLM judge.

BOTH EMPTY SIDES NEED AN EXPLICIT RULE, because each makes one of the two metrics 0/0.

EMPTY GOLD (a greeting, an out-of-domain request): the correct answer is to select NOTHING, so
precision is 1 when nothing was selected and 0 otherwise, and recall is undefined (there is nothing
to recall) and excluded from the recall average. Stating this matters — scoring those cases as "no
relevant items, so trivially perfect" would hide exactly the failure they were written to catch.

EMPTY SELECTION against a non-empty gold: precision is UNDEFINED (0/0) and excluded from the
precision average; the failure is fully carried by recall = 0 and missed = len(gold). It used to be
scored 0.0, which double-counted one error in two metrics and made precision uninterpretable — a
fallen precision could mean "selected wrong things" OR "selected nothing", which are opposite
defects with opposite fixes. The symptom was a reading of precision 0.923 alongside
over_selected_total 0, which cannot both be true under the ordinary definition. Empty selections
are counted in their own field instead, so nothing is hidden.

  python eval/runners/report_memory_select.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics.ir import mean            # noqa: E402
from stats import binom_two_sided       # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HIST = os.path.join(ROOT, "eval", "history")
ARMS = ("llm", "fallback")


def score(selected, gold):
    s, g = set(selected), set(gold)
    if not g:                                  # empty-gold probe: selecting anything is wrong
        return {"precision": 1.0 if not s else 0.0, "recall": None,
                "exact": 1.0 if not s else 0.0, "extra": len(s), "missed": 0}
    hit = len(s & g)
    return {"precision": (hit / len(s)) if s else None,   # 0/0 -> undefined, not 0
            "recall": hit / len(g),
            "exact": 1.0 if s == g else 0.0,
            "extra": len(s - g), "missed": len(g - s),
            "empty_selection": not s}


def main():
    path = os.path.join(HIST, "memory_select.preds.jsonl")
    if not os.path.exists(path):
        sys.exit("no predictions — run evalMemorySelect.ts first")
    preds = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    meta = json.load(open(os.path.join(HIST, "memory_select.meta.json"), encoding="utf-8"))
    probes = [p for p in preds if not p["gold"]]
    normal = [p for p in preds if p["gold"]]

    out = {"n": len(preds), "n_probes": len(probes), "llm_live": meta.get("llm_live"), "arms": {}}
    print(f"memory selection — {len(preds)} cases ({len(probes)} empty-gold probes) "
          f"[llm={'live' if meta.get('llm_live') else 'off'}]\n")
    print(f"  {'arm':10} {'precision':>10} {'recall':>8} {'exact':>7} {'probe P':>9} {'over-select':>12} {'missed':>7}")
    for arm in ARMS:
        sc = [score(p[arm], p["gold"]) for p in preds]
        scn = [score(p[arm], p["gold"]) for p in normal]
        scp = [score(p[arm], p["gold"]) for p in probes]
        # mean() over the DEFINED values only. `recall` is already filtered by using `normal`
        # (empty-gold probes have no recall); `precision` now needs the same treatment for the
        # other 0/0 case — an empty selection against a non-empty gold. Both exclusions are
        # counted and printed rather than silently dropped.
        prec = [x["precision"] for x in sc if x["precision"] is not None]
        empties = sum(1 for x in sc if x.get("empty_selection") and x["precision"] is None)
        row = {
            "precision": round(mean(prec), 4) if prec else None,
            "precision_n": len(prec),
            "recall": round(mean([x["recall"] for x in scn]), 4),
            "exact_match": round(mean([x["exact"] for x in sc]), 4),
            "probe_precision": round(mean([x["precision"] for x in scp]), 4) if scp else None,
            "over_selected_total": sum(x["extra"] for x in sc),
            "missed_total": sum(x["missed"] for x in sc),
            # An empty selection is a real failure; it is excluded from precision (undefined) but
            # never from view. recall and missed already carry its cost.
            "empty_selections": empties,
        }
        out["arms"][arm] = row
        print(f"  {arm:10} {row['precision']:>10.3f} {row['recall']:>8.3f} {row['exact_match']:>7.3f}"
              f" {row['probe_precision']:>9.3f} {row['over_selected_total']:>12} {row['missed_total']:>7}"
              + (f"   (precision over {row['precision_n']}/{len(sc)} cases;"
                 f" {row['empty_selections']} empty selection(s) undefined)" if row["empty_selections"] else ""))

    # Paired: does the LLM arm beat the free fallback, case by case? An unpaired mean difference
    # on 10 cases would be meaningless.
    wins = sum(1 for p in preds if score(p["llm"], p["gold"])["exact"] > score(p["fallback"], p["gold"])["exact"])
    losses = sum(1 for p in preds if score(p["llm"], p["gold"])["exact"] < score(p["fallback"], p["gold"])["exact"])
    pv = binom_two_sided(min(wins, losses), wins + losses)
    out["paired_exact"] = {"llm_wins": wins, "fallback_wins": losses, "p": round(pv, 6)}
    print(f"\n  paired on exact match: llm {wins} / fallback {losses} / tie {len(preds) - wins - losses}"
          f"  (sign test p={pv:.4f})")
    print("  > the fallback ignores the task entirely (salience x recency x frequency), so it is\n"
          "    the floor the extra LLM call has to clear.")

    worst = sorted(preds, key=lambda p: (score(p["llm"], p["gold"])["exact"],
                                         -score(p["llm"], p["gold"])["extra"]))[:5]
    print("\n  worst llm cases:")
    for p in worst:
        s = score(p["llm"], p["gold"])
        tag = f" [{p['note']}]" if p.get("note") else ""
        print(f"    {p['id']}{tag}  gold={p['gold']} llm={p['llm']}"
              f"  (+{s['extra']} 多选 / -{s['missed']} 漏选)")

    with open(os.path.join(HIST, "memory_select.metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print("\nwrote eval/history/memory_select.metrics.json")


if __name__ == "__main__":
    main()
