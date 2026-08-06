"""Classification metrics for intent evaluation (pure functions).

Accuracy, per-class precision/recall/F1, macro-F1, and a confusion matrix.
Works for single-label intent (the common case); multi-intent is scored with
exact-set-match accuracy plus per-label P/R via the flattened pairs.

  from eval.metrics.classification import classification_report
"""
from collections import defaultdict


def classification_report(pairs, labels=None):
    """pairs = list of (gold, pred) single labels.
    Returns accuracy, macro-F1, per-class P/R/F1, and a confusion matrix."""
    labels = labels or sorted({g for g, _ in pairs} | {p for _, p in pairs})
    n = len(pairs)
    correct = sum(1 for g, p in pairs if g == p)

    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)
    confusion = {g: defaultdict(int) for g in labels}
    for g, p in pairs:
        confusion[g][p] += 1
        if g == p:
            tp[g] += 1
        else:
            fp[p] += 1
            fn[g] += 1

    per_class = {}
    f1s = []
    for c in labels:
        prec = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) else 0.0
        rec = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[c] = {"precision": round(prec, 4), "recall": round(rec, 4),
                        "f1": round(f1, 4), "support": tp[c] + fn[c]}
        f1s.append(f1)

    return {
        "accuracy": round(correct / n, 4) if n else 0.0,
        "macro_f1": round(sum(f1s) / len(f1s), 4) if f1s else 0.0,
        "n": n,
        "per_class": per_class,
        "confusion": {g: dict(row) for g, row in confusion.items()},
    }


def exact_set_accuracy(pairs):
    """For multi-intent: fraction where predicted set == gold set exactly."""
    if not pairs:
        return 0.0
    return round(sum(1 for g, p in pairs if set(g) == set(p)) / len(pairs), 4)
