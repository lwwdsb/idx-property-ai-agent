"""A3 — build a BLINDED pairwise annotation task, spending human judgement only where two
configurations actually disagree.

Why not just label a set properly: a full graded pass over 70 queries x 10 results is 700
judgements, and most of them are wasted. Two configurations agree on the majority of queries,
and on a query where they agree the label cannot change which one wins. Annotating only the
DIFFERENCE cuts the work by about an order of magnitude and puts every judgement on a
decision boundary.

Why blinded: the annotator must not know which side is which, or the expected answer leaks
into the judgement. Sides are swapped by a seeded coin flip per query, recorded separately
from the task file.

Why pairwise instead of grading 0/1/2: "which of these two sets better answers the query" is
a relative judgement and far more stable than an absolute relevance grade — no rubric to
internalise, which is a large part of why the existing graded sets are still verified=false.

  python eval/runners/make_pairwise_task.py --a dense --b bm25
  # -> eval/history/pairwise_task.json      (annotate with annotate_pairwise.py)
  #    eval/history/pairwise_key.json       (the un-blinding key; do not read while annotating)

NOTE both files contain listing remarks, i.e. confidential MLS text. They are written under
eval/history/ (gitignored) and must never be committed or uploaded anywhere.
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "retrieval"))
import pymysql                                    # noqa: E402
from common import load_env                       # noqa: E402
from search import hybrid_search                  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "eval", "datasets")
HIST = os.path.join(ROOT, "eval", "history")


def mysql():
    e = load_env()
    return pymysql.connect(host=e.get("DB_HOST", "127.0.0.1"), port=int(e.get("DB_PORT", 3306)),
                           user=e.get("DB_USER"), password=e.get("DB_PASSWORD"),
                           database=e.get("DB_NAME"), cursorclass=pymysql.cursors.DictCursor)


def cards(conn, ids):
    if not ids:
        return {}
    fmt = ",".join(["%s"] * len(ids))
    with conn.cursor() as cur:
        cur.execute(f"""SELECT id, L_Address, L_City, L_Type_, L_Keyword2, LM_Dec_3, LM_Int2_3,
                               L_SystemPrice, L_Remarks
                        FROM rets_property WHERE id IN ({fmt})""", list(ids))
        rows = cur.fetchall()
    out = {}
    for r in rows:
        price = f"${int(float(r['L_SystemPrice'] or 0)):,}"
        out[int(r["id"])] = {
            "id": int(r["id"]),
            "head": f"{r['L_Address']}, {r['L_City']} — {price} · "
                    f"{r['L_Keyword2']}bd/{r['LM_Dec_3']}ba · {r['L_Type_']}",
            "remark": (r["L_Remarks"] or "").replace("\n", " ")[:420],
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="dense", choices=["dense", "bm25", "hybrid"])
    ap.add_argument("--b", default="bm25", choices=["dense", "bm25", "hybrid"])
    ap.add_argument("--dataset", default="retrieval_large.jsonl")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--max-diff", type=int, default=4,
                    help="show at most this many unique items per side (keeps a judgement small)")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap the queries judged, stratified by category (0 = all)")
    ap.add_argument("--rerank-pool", type=int, nargs=2, metavar=("A", "B"), default=None,
                    help="instead of two modes, compare hybrid at two prefetch depths")
    ap.add_argument("--seed", type=int, default=int(os.environ.get("IDX_EVAL_SEED", 7)))
    args = ap.parse_args()
    if args.a == args.b and not args.rerank_pool:
        sys.exit("--a and --b must differ (or pass --rerank-pool A B)")
    if args.rerank_pool:
        args.a, args.b = f"prefetch{args.rerank_pool[0]}", f"prefetch{args.rerank_pool[1]}"

    cases = [json.loads(l) for l in open(os.path.join(DATA, args.dataset), encoding="utf-8") if l.strip()]
    rnd = random.Random(args.seed)
    if args.limit:
        # Stratify, so capping the work does not also narrow what kind of query is represented.
        by_cat = {}
        for c in cases:
            by_cat.setdefault((c.get("meta") or {}).get("category"), []).append(c)
        for v in by_cat.values():
            rnd.shuffle(v)
        picked, cats = [], sorted(by_cat, key=lambda k: str(k))
        while len(picked) < args.limit and any(by_cat.values()):
            for cat in cats:
                if by_cat[cat]:
                    picked.append(by_cat[cat].pop())
                    if len(picked) >= args.limit:
                        break
        cases = picked
    conn = mysql()
    tasks, key, agreed, n_shown = [], [], 0, 0

    for c in cases:
        if args.rerank_pool:
            pa, pb = args.rerank_pool
            ra = [p.id for p in hybrid_search(c["input"], None, k=args.k, mode="hybrid", prefetch=pa)]
            rb = [p.id for p in hybrid_search(c["input"], None, k=args.k, mode="hybrid", prefetch=pb)]
        else:
            ra = [p.id for p in hybrid_search(c["input"], None, k=args.k, mode=args.a)]
            rb = [p.id for p in hybrid_search(c["input"], None, k=args.k, mode=args.b)]
        only_a, only_b = [i for i in ra if i not in rb], [i for i in rb if i not in ra]
        if not only_a and not only_b:
            agreed += 1
            continue
        only_a, only_b = only_a[:args.max_diff], only_b[:args.max_diff]
        cd = cards(conn, only_a + only_b)
        swap = rnd.random() < 0.5                       # blind: which side is printed as "1"
        side1, side2 = (only_b, only_a) if swap else (only_a, only_b)
        tasks.append({
            "id": c["id"], "query": c["input"],
            "category": (c.get("meta") or {}).get("category"),
            "overlap": len([i for i in ra if i in rb]),
            "side1": [cd[i] for i in side1 if i in cd],
            "side2": [cd[i] for i in side2 if i in cd],
            "verdict": "",                              # "1" | "2" | "tie"
        })
        key.append({"id": c["id"], "side1_is": args.b if swap else args.a,
                    "side2_is": args.a if swap else args.b})
        n_shown += len(side1) + len(side2)
    conn.close()

    with open(os.path.join(HIST, "pairwise_task.json"), "w", encoding="utf-8") as fh:
        json.dump({"a": args.a, "b": args.b, "dataset": args.dataset, "k": args.k,
                   "tasks": tasks}, fh, indent=2, ensure_ascii=False)
    with open(os.path.join(HIST, "pairwise_key.json"), "w", encoding="utf-8") as fh:
        json.dump({"a": args.a, "b": args.b, "key": key}, fh, indent=2)

    full = len(cases) * args.k
    print(f"{args.a} vs {args.b} on {args.dataset} (top-{args.k})\n")
    print(f"  identical top-{args.k}, nothing to judge : {agreed}/{len(cases)} queries")
    print(f"  queries needing a judgement            : {len(tasks)}")
    print(f"  listing cards to read                  : {n_shown}")
    print(f"  a full graded pass would have been     : {full} judgements "
          f"({full / max(1, n_shown):.1f}x more reading, and most of it on results both "
          f"configurations returned)")
    print("\nnext: python eval/runners/annotate_pairwise.py")


if __name__ == "__main__":
    main()
