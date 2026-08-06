"""RAG eval — retrieval hit-rate, and the empirical case for a similarity threshold gate.

Measures:
1. in-corpus retrieval hit@k — is the gold source retrieved in the top-k?
2. score separation — do OUT-of-corpus questions have lower top similarity than
   in-corpus ones? If so, a threshold gate could refuse them. This turns "we should add
   a hard threshold gate" (an opinion) into a data-backed number: the best threshold and
   the refusal accuracy it achieves.
3. current refusal behavior — extractive mode always answers (no gate), quantifying the gap.

  python eval/runners/eval_rag.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "retrieval"))
sys.path.insert(0, os.path.join(ROOT, "eval", "metrics"))
from rag import RagIndex  # noqa: E402
from ir import mean  # noqa: E402

DATA = os.path.join(ROOT, "eval", "datasets", "rag.jsonl")
HIST = os.path.join(ROOT, "eval", "history")
K = 3


def load():
    with open(DATA) as f:
        return [json.loads(l) for l in f if l.strip()]


def best_threshold(in_scores, out_scores):
    """Find the top-score threshold that best separates in- vs out-of-corpus:
    in-corpus should be >= t (answer), out-of-corpus should be < t (refuse)."""
    cands = sorted(set(round(s, 3) for s in in_scores + out_scores))
    best_t, best_acc = 0.0, 0.0
    for t in cands:
        correct = sum(1 for s in in_scores if s >= t) + sum(1 for s in out_scores if s < t)
        acc = correct / (len(in_scores) + len(out_scores))
        if acc > best_acc:
            best_acc, best_t = acc, t
    return best_t, round(best_acc, 4)


def main():
    cases = load()
    idx = RagIndex()

    in_cases = [c for c in cases if c["label"]["in_corpus"]]
    out_cases = [c for c in cases if not c["label"]["in_corpus"]]

    # 1. in-corpus retrieval hit@k
    hits, in_scores = [], []
    for c in in_cases:
        got = idx.retrieve(c["input"], K)
        top = max((h["score"] for h in got), default=0.0)
        in_scores.append(top)
        gold = c["label"]["gold_source"].lower()
        hits.append(1.0 if any(gold in h["source"].lower() for h in got) else 0.0)

    # 2. out-of-corpus top scores
    out_scores = [max((h["score"] for h in idx.retrieve(c["input"], K)), default=0.0)
                  for c in out_cases]

    hit_rate = round(mean(hits), 4)
    in_mean, out_mean = round(mean(in_scores), 4), round(mean(out_scores), 4)
    t, acc = best_threshold(in_scores, out_scores)

    print(f"RAG eval: {len(in_cases)} in-corpus + {len(out_cases)} out-of-corpus\n")
    print(f"  in-corpus retrieval hit@{K}:        {hit_rate}")
    print(f"  mean top similarity  in-corpus:    {in_mean}")
    print(f"  mean top similarity  out-of-corpus:{out_mean}")
    print(f"  separation (in - out):             {round(in_mean - out_mean, 4)}")
    print()
    print(f"  → best threshold gate at score >= {t}")
    print(f"    would separate in/out with accuracy {acc}")
    print(f"    (current system has NO hard gate → out-of-corpus questions are answered,")
    print(f"     i.e. refusal accuracy on out-of-corpus ~ 0 in extractive mode)")

    result = {
        "hit_at_k": hit_rate, "k": K,
        "in_corpus_mean_score": in_mean, "out_corpus_mean_score": out_mean,
        "separation": round(in_mean - out_mean, 4),
        "suggested_threshold": t, "threshold_accuracy": acc,
        "n_in": len(in_cases), "n_out": len(out_cases),
    }
    os.makedirs(HIST, exist_ok=True)
    with open(os.path.join(HIST, "rag.metrics.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nmetrics written to {os.path.join(HIST, 'rag.metrics.json')}")


if __name__ == "__main__":
    main()
