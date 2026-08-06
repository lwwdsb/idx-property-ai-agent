"""Parse / slot-filling metrics (pure functions).

Compares a predicted structured filter against the gold filter, at the field level:
- exact_match: the whole filter matches exactly (strict)
- slot precision/recall/F1: over the individual (field, value) pairs
- escalation accuracy: did the pipeline's "call the LLM?" decision match the label

  from eval.metrics.parsing import slot_scores
"""


def _pairs(f):
    """Filter dict -> set of (field, value) pairs, ignoring None/unset."""
    return {(k, v) for k, v in (f or {}).items() if v is not None}


def slot_scores(records):
    """records = list of {gold, pred, gold_escalate?, pred_escalate?}.
    Returns exact-match rate, micro slot P/R/F1, and escalation accuracy."""
    exact = 0
    tp = fp = fn = 0
    esc_total = esc_correct = 0
    for r in records:
        g, p = _pairs(r["gold"]), _pairs(r["pred"])
        if g == p:
            exact += 1
        tp += len(g & p)
        fp += len(p - g)
        fn += len(g - p)
        if "gold_escalate" in r and r["gold_escalate"] is not None:
            esc_total += 1
            if bool(r["gold_escalate"]) == bool(r.get("pred_escalate")):
                esc_correct += 1

    n = len(records)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "n": n,
        "exact_match": round(exact / n, 4) if n else 0.0,
        "slot_precision": round(prec, 4),
        "slot_recall": round(rec, 4),
        "slot_f1": round(f1, 4),
        "escalation_accuracy": round(esc_correct / esc_total, 4) if esc_total else None,
    }
