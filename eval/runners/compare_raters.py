"""Three-way rater comparison on the blinded pairwise task.

Raters live in separate files ON PURPOSE, because they are not interchangeable evidence:
  pairwise_task.json  "verdict"  HUMAN            — the only ground truth
  pairwise_judge.json            DeepSeek         — the cheap judge under evaluation
  pairwise_claude.json           Claude           — a SECOND MODEL, not a human stand-in

The distinction is the whole point. Two language models agreeing does NOT calibrate either
one: both read the same remark text and can share the same bias toward lexical overlap — the
exact confound the graded sets were already suspected of. Agreement between models measures
reproducibility across models; only the human column measures correctness. Writing model
verdicts into the human field would have destroyed that distinction silently, so it is
structurally impossible here.

  python eval/runners/compare_raters.py
"""
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats import binom_two_sided, sign_test          # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HIST = os.path.join(ROOT, "eval", "history")


def load(name):
    p = os.path.join(HIST, name)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}


def main():
    task = load("pairwise_task.json")
    if not task:
        sys.exit("no pairwise task")
    key = {k["id"]: k for k in load("pairwise_key.json")["key"]}
    judge, claude = load("pairwise_judge.json"), load("pairwise_claude.json")
    a, b = task["a"], task["b"]

    def unblind(tid, v):
        return "tie" if v == "tie" else (key[tid]["side1_is"] if v == "1" else key[tid]["side2_is"])

    raters = {"human": {}, "deepseek": {}, "claude": {}}
    for t in task["tasks"]:
        tid = t["id"]
        if t.get("verdict"):
            raters["human"][tid] = unblind(tid, t["verdict"])
        if tid in judge:
            raters["deepseek"][tid] = unblind(tid, judge[tid]["verdict"])
        if tid in claude:
            raters["claude"][tid] = unblind(tid, claude[tid]["verdict"])

    print(f"{a} vs {b}\n")
    print(f"  {'rater':10} {a:>7} {b:>7} {'tie':>5} {'n':>4} {'sign p':>9}  preference")
    prefs = {}
    for name, v in raters.items():
        w = collections.Counter(v.values())
        p = sign_test(w[a], w[b])
        est = "none established" if p > 0.05 else f"{a if w[a] > w[b] else b}"
        prefs[name] = {"wins_a": w[a], "wins_b": w[b], "tie": w["tie"], "n": len(v),
                       "sign_p": round(p, 6), "preference": est,
                       "tie_rate": round(w["tie"] / len(v), 4) if v else None}
        print(f"  {name:10} {w[a]:>7} {w[b]:>7} {w['tie']:>5} {len(v):>4} {p:>9.4f}  {est}")

    print("\n  pairwise agreement (only rows BOTH rated):")
    pairs = {}
    for x, y in (("claude", "deepseek"), ("claude", "human"), ("deepseek", "human")):
        both = [(raters[x][t], raters[y][t]) for t in raters[x] if t in raters[y]]
        if not both:
            continue
        exact = sum(1 for p, q in both if p == q)
        dec = [(p, q) for p, q in both if p != "tie" and q != "tie"]
        same = sum(1 for p, q in dec if p == q)
        opp = len(dec) - same
        pv = binom_two_sided(min(same, opp), len(dec)) if dec else None
        pairs[f"{x}_vs_{y}"] = {"n": len(both), "exact": exact,
                                "exact_rate": round(exact / len(both), 4),
                                "both_decisive": len(dec), "same_direction": same,
                                "opposite": opp, "p": round(pv, 6) if pv is not None else None}
        print(f"    {x:9} vs {y:9} exact {exact}/{len(both)} = {exact / len(both):.2f}"
              f" | decisive {len(dec)}: same {same}, opposite {opp}"
              + (f", p={pv:.4f}" if pv is not None else ""))

    print("\n  READ THIS BEFORE USING THE NUMBERS:")
    print("    - Only `human` is ground truth. claude is a second MODEL: model-model agreement")
    print("      measures reproducibility, not correctness, and can be two shared biases.")
    print("    - A rater whose own sign test is not significant has NOT established a")
    print("      preference; quoting its win count as a result would be reading noise.")

    out = {"a": a, "b": b, "raters": prefs, "agreement": pairs,
           "caveat": "human is the only ground truth; claude is a second model and "
                     "model-model agreement does not calibrate the judge"}
    with open(os.path.join(HIST, "rater_comparison.metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print("\nwrote eval/history/rater_comparison.metrics.json")


if __name__ == "__main__":
    main()
