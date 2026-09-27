"""A2 — calibrate the cheap automatic judge against human judgement.

The small human set is a CALIBRATION set, not an optimisation target. Thirty rows cannot tune
anything without overfitting, but they can answer the one question that decides whether the
cheap metric may be trusted at all: does the LLM judge point the SAME WAY as a person?

It reuses A3's pairwise task instead of asking for fresh 0/1/2 grades, for two reasons: the
human work is already done, and the judge is asked the EXACT question it would be used for —
"which of these two result sets better answers the query" — rather than a proxy. Both see the
same blinded side-1/side-2 presentation, so neither gets a positional hint the other lacks.

  python eval/runners/calibrate_judge.py --judge     # run the judge over the task (LLM calls)
  python eval/runners/calibrate_judge.py             # agreement on rows BOTH have judged

Reads/writes only under eval/history/ — the cards contain confidential remark text.
"""
import argparse
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "retrieval"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm import chat, llm_available          # noqa: E402
from stats import binom_two_sided            # noqa: E402

HIST = os.path.join(ROOT, "eval", "history")
TASK = os.path.join(HIST, "pairwise_task.json")
JUDGE = os.path.join(HIST, "pairwise_judge.json")
KEY = os.path.join(HIST, "pairwise_key.json")

SYSTEM = """You compare two sets of property listings against a buyer's query.

Answer with the side whose listings better satisfy what the buyer asked for. Judge the FIT to
the query, not which listings are nicer. Answer "tie" only when you genuinely cannot separate
them.

Return STRICT JSON: {"verdict": "1" | "2" | "tie", "why": "<one short clause>"}"""


def render(cards):
    return "\n".join(f"- #{c['id']} {c['head']}\n  {c['remark']}" for c in cards)


def run_judge(task):
    out = {}
    if os.path.exists(JUDGE):
        out = json.load(open(JUDGE, encoding="utf-8"))
    todo = [t for t in task["tasks"] if t["id"] not in out]
    print(f"judging {len(todo)} of {len(task['tasks'])} (already done: {len(out)})")
    for i, t in enumerate(todo, 1):
        prompt = (f"QUERY: {t['query']}\n\nSIDE 1:\n{render(t['side1'])}\n\n"
                  f"SIDE 2:\n{render(t['side2'])}")
        raw = chat(prompt, system=SYSTEM, temperature=0)
        v, why = "", ""
        if raw:
            m = re.search(r"\{.*\}", raw, re.S)
            if m:
                try:
                    d = json.loads(m.group(0))
                    v = str(d.get("verdict", "")).strip()
                    why = str(d.get("why", ""))[:120]
                except json.JSONDecodeError:
                    pass
        if v not in ("1", "2", "tie"):
            print(f"  [{i}] {t['id']}: unparseable verdict, skipped")
            continue
        out[t["id"]] = {"verdict": v, "why": why}
        print(f"  [{i}] {t['id']}: {v}  {why}")
        with open(JUDGE, "w", encoding="utf-8") as fh:      # resumable
            json.dump(out, fh, indent=2, ensure_ascii=False)
    print(f"\nwrote {len(out)} judge verdicts -> eval/history/pairwise_judge.json")


def kappa(pairs, cats):
    """Cohen's kappa. Raw agreement flatters a rater that always answers the majority class,
    which matters here because the judge may almost never say tie."""
    n = len(pairs)
    if not n:
        return 0.0
    obs = sum(1 for a, b in pairs if a == b) / n
    exp = sum((sum(1 for a, _ in pairs if a == c) / n) * (sum(1 for _, b in pairs if b == c) / n)
              for c in cats)
    return 0.0 if exp >= 1 else round((obs - exp) / (1 - exp), 4)


def agreement(task, judge):
    key = {k["id"]: k for k in json.load(open(KEY, encoding="utf-8"))["key"]}
    a, b = task["a"], task["b"]

    def unblind(tid, v):
        if v == "tie":
            return "tie"
        k = key[tid]
        return k["side1_is"] if v == "1" else k["side2_is"]

    pairs, rows = [], []
    for t in task["tasks"]:
        hv, jv = t.get("verdict"), (judge.get(t["id"]) or {}).get("verdict")
        if not hv or not jv:
            continue
        h, j = unblind(t["id"], hv), unblind(t["id"], jv)
        pairs.append((h, j))
        rows.append((t["id"], t.get("category"), h, j))
    if not pairs:
        sys.exit("no row has BOTH a human and a judge verdict yet — "
                 "annotate with annotate_pairwise.py, then run --judge")

    cats = [a, b, "tie"]
    n = len(pairs)
    exact = sum(1 for h, j in pairs if h == j)
    # Direction-only agreement: ignore rows where either side said tie, then ask whether the
    # two agree on WHICH config wins. This is the question that matters for tuning — a judge
    # that is merely more decisive than a human is still usable if it never points the wrong way.
    dirp = [(h, j) for h, j in pairs if h != "tie" and j != "tie"]
    same_dir = sum(1 for h, j in dirp if h == j)
    opposite = len(dirp) - same_dir
    p = binom_two_sided(min(same_dir, opposite), len(dirp))

    print(f"{a} vs {b}   rows judged by BOTH: {n}\n")
    print(f"  exact agreement (incl. ties) : {exact}/{n} = {exact / n:.2f}")
    print(f"  Cohen's kappa                : {kappa(pairs, cats)}")
    print(f"  human ties / judge ties      : {sum(1 for h, _ in pairs if h == 'tie')}"
          f" / {sum(1 for _, j in pairs if j == 'tie')}")
    print(f"  both decisive                : {len(dirp)} rows -> same direction {same_dir}, "
          f"opposite {opposite}")
    if len(dirp):
        print(f"  sign test on those           : p = {p:.4f}")

    if not dirp:
        verdict = "no row where both were decisive — cannot calibrate direction yet"
    elif opposite == 0:
        verdict = (f"USABLE: the judge never pointed against the human on {len(dirp)} decisive "
                   f"rows. It may stand in for a human on this comparison.")
    elif same_dir == 0:
        verdict = (f"INVERTED: the judge pointed AGAINST the human on all {len(dirp)} decisive "
                   f"rows. Do not use it to pick parameters — this is the judge confound, "
                   f"measured directly.")
    elif p <= 0.05 and same_dir > opposite:
        verdict = f"USABLE WITH CARE: agrees on direction {same_dir}/{len(dirp)} (p={p:.4f})."
    else:
        verdict = (f"NOT CALIBRATED: {same_dir} agree vs {opposite} opposite on {len(dirp)} "
                   f"rows (p={p:.4f}) — the judge is not distinguishable from a coin on the "
                   f"direction, so it must not gate parameter choices.")
    print(f"\n  => {verdict}")

    disagree = [r for r in rows if r[2] != r[3]]
    if disagree:
        print(f"\n  disagreements ({len(disagree)}):")
        for tid, cat, h, j in disagree:
            print(f"    {tid:12} {str(cat):10} human={h:8} judge={j}")

    out = {"a": a, "b": b, "n_both": n, "exact_agreement": round(exact / n, 4),
           "kappa": kappa(pairs, cats), "human_ties": sum(1 for h, _ in pairs if h == "tie"),
           "judge_ties": sum(1 for _, j in pairs if j == "tie"),
           "both_decisive": len(dirp), "same_direction": same_dir, "opposite": opposite,
           "sign_test_p": round(p, 6), "verdict": verdict}
    with open(os.path.join(HIST, "judge_calibration.metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print("\nwrote eval/history/judge_calibration.metrics.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", action="store_true", help="run the LLM judge over the task")
    args = ap.parse_args()
    if not os.path.exists(TASK):
        sys.exit("no pairwise task — run make_pairwise_task.py first")
    task = json.load(open(TASK, encoding="utf-8"))
    if args.judge:
        if not llm_available():
            sys.exit("no LLM key")
        run_judge(task)
        return
    judge = json.load(open(JUDGE, encoding="utf-8")) if os.path.exists(JUDGE) else {}
    agreement(task, judge)


if __name__ == "__main__":
    main()
