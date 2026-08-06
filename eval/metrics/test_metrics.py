"""Unit tests for the metric functions — hand-computed expected values.
  python eval/metrics/test_metrics.py
"""
import os
import sys
from math import log2, isclose

sys.path.insert(0, os.path.dirname(__file__))
from ir import precision_at_k, recall_at_k, mrr, ndcg_at_k, hit_at_k  # noqa: E402
from classification import classification_report, exact_set_accuracy  # noqa: E402
from parsing import slot_scores  # noqa: E402

passed = failed = 0


def check(name, cond):
    global passed, failed
    if cond:
        passed += 1; print("✓", name)
    else:
        failed += 1; print("✗", name)


# ---- IR ----
ranked = [10, 20, 30, 40]           # ranked ids
relevant = {"10": 2, "30": 1}       # graded relevance (20,40 not relevant)

check("precision@2 = 1/2 (10 rel, 20 not)", isclose(precision_at_k(ranked, relevant, 2), 0.5))
check("recall@2 = 1/2 (found 10 of {10,30})", isclose(recall_at_k(ranked, relevant, 2), 0.5))
check("recall@4 = 1.0 (found both)", isclose(recall_at_k(ranked, relevant, 4), 1.0))
check("mrr = 1.0 (first result relevant)", isclose(mrr(ranked, relevant), 1.0))
check("mrr = 1/2 when first relevant is rank 2", isclose(mrr([99, 10], relevant), 0.5))
# nDCG@4: DCG = 2/log2(2) + 1/log2(4) = 2 + 0.5 = 2.5 ; IDCG = 2/log2(2)+1/log2(3)=2+0.6309=2.6309
_dcg = 2 / log2(2) + 1 / log2(4)
_idcg = 2 / log2(2) + 1 / log2(3)
check("ndcg@4 matches hand calc", isclose(ndcg_at_k(ranked, relevant, 4), _dcg / _idcg, rel_tol=1e-9))
check("ndcg = 1.0 for ideal order", isclose(ndcg_at_k([10, 30], {"10": 2, "30": 1}, 2), 1.0))
check("hit@k true when gold in top-k", hit_at_k([1, 2, 3], [3], 3) == 1.0)
check("hit@k false when gold below k", hit_at_k([1, 2, 3], [3], 2) == 0.0)

# ---- classification ----
rep = classification_report([("search", "search"), ("market", "market"),
                             ("search", "market"), ("knowledge", "knowledge")])
check("accuracy = 3/4", isclose(rep["accuracy"], 0.75))
check("confusion records search->market", rep["confusion"]["search"].get("market") == 1)
check("multi-intent exact-set acc", isclose(
    exact_set_accuracy([(["search", "market"], ["market", "search"]),
                        (["search"], ["market"])]), 0.5))

# ---- parsing ----
ps = slot_scores([
    {"gold": {"city": "Irvine", "beds": 3}, "pred": {"city": "Irvine", "beds": 3},
     "gold_escalate": False, "pred_escalate": False},
    {"gold": {"city": "Irvine", "beds": 4}, "pred": {"city": "Irvine", "beds": 3},  # beds wrong
     "gold_escalate": True, "pred_escalate": True},
])
check("parse exact_match = 1/2", isclose(ps["exact_match"], 0.5))
check("parse escalation_accuracy = 1.0", isclose(ps["escalation_accuracy"], 1.0))
# slots: gold pairs=4, pred pairs=4, TP = city×2 + beds3(record2 matches beds3? gold beds4)
#   record1: {(city,Irvine),(beds,3)} both match -> TP2
#   record2: gold {(city,Irvine),(beds,4)} pred {(city,Irvine),(beds,3)} -> TP1(city), FP1(beds3), FN1(beds4)
#   TP=3, FP=1, FN=1 -> P=3/4, R=3/4
check("parse slot precision = 0.75", isclose(ps["slot_precision"], 0.75))
check("parse slot recall = 0.75", isclose(ps["slot_recall"], 0.75))

print(f"\n{passed}/{passed + failed} passed")
raise SystemExit(1 if failed else 0)
