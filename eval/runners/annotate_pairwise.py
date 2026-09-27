"""A3 — annotate the blinded pairwise task. Run this yourself in a terminal.

Each screen is one query and the results the two configurations did NOT agree on. You are not
told which side is which. Answer with the side that better answers the query, or `t` when you
genuinely cannot separate them — a tie is a real answer here and is dropped from the test
rather than forced into a direction.

Progress is saved after every answer, so stopping halfway is fine: re-running resumes.

  python eval/runners/annotate_pairwise.py            # continue where you left off
  python eval/runners/annotate_pairwise.py --restart   # clear verdicts and start over
"""
import argparse
import json
import os
import textwrap

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TASK = os.path.join(ROOT, "eval", "history", "pairwise_task.json")


def show(cards, label, width):
    print(f"\n  ── SIDE {label} " + "─" * max(0, width - 12))
    for c in cards:
        print(f"   #{c['id']}  {c['head']}")
        for line in textwrap.wrap(c["remark"], width - 6):
            print(f"      {line}")
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--restart", action="store_true")
    ap.add_argument("--width", type=int, default=96)
    args = ap.parse_args()
    if not os.path.exists(TASK):
        raise SystemExit("no task — run make_pairwise_task.py first")
    data = json.load(open(TASK, encoding="utf-8"))
    tasks = data["tasks"]
    if args.restart:
        for t in tasks:
            t["verdict"] = ""

    todo = [t for t in tasks if not t["verdict"]]
    print(f"{len(tasks) - len(todo)}/{len(tasks)} already judged. "
          f"{len(todo)} to go.   [1] side 1  [2] side 2  [t] tie  [s] skip  [q] save+quit")
    for n, t in enumerate(todo, 1):
        print("\n" + "=" * args.width)
        print(f"({n}/{len(todo)})  {t['id']}"
              f"{'  [' + t['category'] + ']' if t.get('category') else ''}"
              f"   both sides also returned {t['overlap']} of the same listings")
        print(f"QUERY: {t['query']}")
        show(t["side1"], "1", args.width)
        show(t["side2"], "2", args.width)
        while True:
            a = input("  which side better answers the query? [1/2/t/s/q] ").strip().lower()
            if a in ("1", "2", "t", "s", "q"):
                break
        if a == "q":
            break
        if a != "s":
            t["verdict"] = {"1": "1", "2": "2", "t": "tie"}[a]
        with open(TASK, "w", encoding="utf-8") as fh:      # save after every answer
            json.dump(data, fh, indent=2, ensure_ascii=False)

    done = sum(1 for t in tasks if t["verdict"])
    with open(TASK, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    print(f"\nsaved. {done}/{len(tasks)} judged.  score it: python eval/runners/score_pairwise.py")


if __name__ == "__main__":
    main()
