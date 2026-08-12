"""Cross-model judge agreement check (NOT human validation).

Honestly measures whether a SECOND, different-family judge agrees with the DeepSeek
grades used to build the retrieval set. This gauges label reproducibility/stability
across models — it is NOT human ground truth (a human spot-check remains separate).

Two phases:
  build : sample queries, build pools, get DeepSeek grades (all pool ids, 0/1/2),
          save graded pool locally, and print a BLIND view for the 2nd judge.
  score : given the 2nd judge's grades JSON, compute exact-agreement + weighted Cohen's κ.

  python eval/runners/agreement_check.py build --n 5
  python eval/runners/agreement_check.py score --grades <path-to-json>
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "retrieval"))
from build_retrieval_set import pool_for, fetch_remarks, QUERIES  # noqa: E402
from llm import chat  # noqa: E402

HIST = os.path.join(ROOT, "eval", "history")
GRADED = os.path.join(HIST, "agreement_deepseek.pool.jsonl")   # .pool -> gitignored
BLIND = os.path.join(HIST, "agreement_blind.pool.jsonl")

JUDGE_SYS = (
    "You grade how well each listing satisfies the DISTINCTIVE requirement in a buyer's "
    "query. Be STRICT: 2 = the listing SPECIFICALLY has that feature/style; 1 = loosely "
    "related; 0 = the distinctive feature is ABSENT (even if a nice home). "
    'Grade EVERY id. Return JSON only: {"grades": {"<id>": <0|1|2>, ...}}.'
)


def deepseek_grade_all(query, id_to_text):
    block = "\n".join(f"[{i}] {t}" for i, t in id_to_text.items())
    raw = chat(f"Query: {query}\n\nListings:\n{block}\n\nGrade every listing id.", system=JUDGE_SYS)
    try:
        s, e = raw.index("{"), raw.rindex("}")
        g = json.loads(raw[s:e + 1]).get("grades", {})
        return {str(i): int(g.get(str(i), 0)) for i in id_to_text}
    except Exception:
        return {str(i): 0 for i in id_to_text}


def build(n):
    import random
    random.seed(7)
    qs = random.sample(QUERIES, n)
    graded, blind = [], []
    for qi, q in enumerate(qs, 1):
        ids = pool_for(q)[:12]           # cap pool per query to keep grading manageable
        texts = fetch_remarks(ids)
        ds = deepseek_grade_all(q, texts)
        for i in ids:
            graded.append({"q": q, "id": i, "deepseek": ds[str(i)]})
            blind.append({"q": q, "id": i, "text": texts.get(i, "")})
        print(f"  pooled+graded query {qi}/{n}: {q[:40]} ({len(ids)} listings)")
    os.makedirs(HIST, exist_ok=True)
    with open(GRADED, "w") as f:
        f.write("\n".join(json.dumps(r, ensure_ascii=False) for r in graded) + "\n")
    with open(BLIND, "w") as f:
        f.write("\n".join(json.dumps(r, ensure_ascii=False) for r in blind) + "\n")
    print(f"\nwrote {len(graded)} pairs. BLIND view for the 2nd judge:\n")
    for r in blind:
        print(f'  id={r["id"]}  q="{r["q"]}"\n     {r["text"][:200]}\n')


def weighted_kappa(pairs, cats=(0, 1, 2)):
    """Cohen's quadratic weighted kappa for ordinal grades."""
    n = len(pairs)
    if n == 0:
        return None
    idx = {c: k for k, c in enumerate(cats)}
    K = len(cats)
    O = [[0] * K for _ in range(K)]
    for a, b in pairs:
        O[idx[a]][idx[b]] += 1
    ra = [sum(O[i]) for i in range(K)]
    cb = [sum(O[i][j] for i in range(K)) for j in range(K)]
    W = [[((i - j) ** 2) / ((K - 1) ** 2) for j in range(K)] for i in range(K)]
    E = [[ra[i] * cb[j] / n for j in range(K)] for i in range(K)]
    num = sum(W[i][j] * O[i][j] for i in range(K) for j in range(K))
    den = sum(W[i][j] * E[i][j] for i in range(K) for j in range(K))
    return round(1 - num / den, 4) if den else None


def score(grades_path):
    ds = [json.loads(l) for l in open(GRADED) if l.strip()]
    mine = json.load(open(grades_path))   # { "<id>": grade } — keyed by listing id
    pairs = []
    missing = 0
    for r in ds:
        k = str(r["id"])
        if k in mine:
            pairs.append((int(r["deepseek"]), int(mine[k])))
        else:
            missing += 1
    exact = sum(1 for a, b in pairs if a == b) / len(pairs) if pairs else 0.0
    within1 = sum(1 for a, b in pairs if abs(a - b) <= 1) / len(pairs) if pairs else 0.0
    # binary agreement (relevant>=1 vs not) — matters most for retrieval
    bin_pairs = [((1 if a > 0 else 0), (1 if b > 0 else 0)) for a, b in pairs]
    bin_exact = sum(1 for a, b in bin_pairs if a == b) / len(bin_pairs) if bin_pairs else 0.0
    kappa = weighted_kappa(pairs)

    result = {
        "n_pairs": len(pairs), "missing": missing,
        "exact_agreement": round(exact, 4),
        "within_1_agreement": round(within1, 4),
        "binary_relevant_agreement": round(bin_exact, 4),
        "quadratic_weighted_kappa": kappa,
        "note": "cross-model (DeepSeek judge vs Claude judge), NOT human-verified",
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    with open(os.path.join(HIST, "agreement.metrics.json"), "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--n", type=int, default=5)
    s = sub.add_parser("score"); s.add_argument("--grades", required=True)
    args = ap.parse_args()
    if args.cmd == "build":
        build(args.n)
    else:
        score(args.grades)
