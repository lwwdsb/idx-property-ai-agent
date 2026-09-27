"""A3 — un-blind the pairwise verdicts and test them.

Reports a sign test over the non-tied judgements. Ties are dropped rather than split, because
"I cannot separate these" is evidence about the SIZE of the difference and forcing it into a
direction would manufacture a signal.

  python eval/runners/score_pairwise.py
"""
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats import sign_test          # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HIST = os.path.join(ROOT, "eval", "history")


def main():
    task = json.load(open(os.path.join(HIST, "pairwise_task.json"), encoding="utf-8"))
    key = {k["id"]: k for k in json.load(open(os.path.join(HIST, "pairwise_key.json"),
                                              encoding="utf-8"))["key"]}
    a, b = task["a"], task["b"]
    wins = defaultdict(int)
    by_cat = defaultdict(lambda: defaultdict(int))
    judged = 0
    for t in task["tasks"]:
        v = t["verdict"]
        if not v:
            continue
        judged += 1
        k = key[t["id"]]
        winner = "tie" if v == "tie" else (k["side1_is"] if v == "1" else k["side2_is"])
        wins[winner] += 1
        by_cat[t.get("category") or "-"][winner] += 1

    n_total = len(task["tasks"])
    if not judged:
        sys.exit("nothing judged yet — run annotate_pairwise.py")
    p = sign_test(wins[a], wins[b])
    print(f"{a} vs {b}   judged {judged}/{n_total}\n")
    print(f"  {a:12} wins : {wins[a]}")
    print(f"  {b:12} wins : {wins[b]}")
    print(f"  {'tie':12}      : {wins['tie']}")
    print(f"\n  sign test on {wins[a] + wins[b]} non-tied judgements: p = {p:.4f}")
    if wins[a] + wins[b] == 0:
        verdict = "every judgement was a tie — the two configurations are indistinguishable to a human here"
    elif p > 0.05:
        verdict = (f"NOT SIGNIFICANT — a human preference between {a} and {b} is not "
                   f"established at this sample size; do not pick a winner from this")
    else:
        lead = a if wins[a] > wins[b] else b
        verdict = f"SIGNIFICANT — {lead} is preferred (p={p:.4f})"
    print(f"  => {verdict}")

    if len(by_cat) > 1:
        print(f"\n  by category ({a}/{b}/tie):")
        for cat, w in sorted(by_cat.items()):
            print(f"    {cat:10} {w[a]}/{w[b]}/{w['tie']}")

    out = {"a": a, "b": b, "judged": judged, "n_total": n_total,
           "wins": {a: wins[a], b: wins[b], "tie": wins["tie"]},
           "sign_test_p": round(p, 6), "verdict": verdict,
           "by_category": {c: dict(w) for c, w in by_cat.items()}}
    with open(os.path.join(HIST, "pairwise.metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print("\nwrote eval/history/pairwise.metrics.json")


if __name__ == "__main__":
    main()
