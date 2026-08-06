"""Information-retrieval metrics (pure functions, unit-testable).

Given a ranked list of retrieved ids and a graded-relevance judgment
({id: grade}, grade 0/1/2), compute the standard IR quality metrics. These turn
"the results look reasonable" into nDCG@10 / recall@k / MRR numbers.

  from eval.metrics.ir import ndcg_at_k, precision_at_k, recall_at_k, mrr
"""
from math import log2


def _rel(ranked, relevant):
    """Grade for each ranked id (0 if not judged relevant)."""
    return [relevant.get(str(i), 0) for i in ranked]


def precision_at_k(ranked, relevant, k):
    """Fraction of the top-k that are relevant (grade > 0)."""
    if k <= 0:
        return 0.0
    top = _rel(ranked[:k], relevant)
    return sum(1 for g in top if g > 0) / k


def recall_at_k(ranked, relevant, k):
    """Fraction of all relevant items that appear in the top-k."""
    total_rel = sum(1 for g in relevant.values() if g > 0)
    if total_rel == 0:
        return 0.0
    hit = sum(1 for g in _rel(ranked[:k], relevant) if g > 0)
    return hit / total_rel


def mrr(ranked, relevant):
    """Reciprocal rank of the first relevant result (0 if none)."""
    for pos, i in enumerate(ranked, start=1):
        if relevant.get(str(i), 0) > 0:
            return 1.0 / pos
    return 0.0


def dcg_at_k(ranked, relevant, k):
    """Discounted cumulative gain using graded relevance: sum g / log2(pos+1)."""
    return sum(g / log2(pos + 1) for pos, g in enumerate(_rel(ranked[:k], relevant), start=1))


def ndcg_at_k(ranked, relevant, k):
    """DCG normalized by the ideal DCG (best possible ordering). 0..1."""
    dcg = dcg_at_k(ranked, relevant, k)
    ideal = sorted((g for g in relevant.values() if g > 0), reverse=True)[:k]
    idcg = sum(g / log2(pos + 1) for pos, g in enumerate(ideal, start=1))
    return dcg / idcg if idcg > 0 else 0.0


def hit_at_k(ranked, gold_ids, k):
    """1 if any gold id appears in the top-k, else 0 (for RAG source hit-rate)."""
    top = {str(i) for i in ranked[:k]}
    return 1.0 if any(str(g) in top for g in gold_ids) else 0.0


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0
