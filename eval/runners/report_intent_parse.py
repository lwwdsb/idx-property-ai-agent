"""Intent + parse eval — METRICS stage.

Reads the predictions written by evalIntentParse.ts and computes metrics with the
single tested metrics library. Prints a summary and returns a dict the top-level
report aggregator can consume.

  python eval/runners/report_intent_parse.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "eval", "metrics"))
from classification import classification_report  # noqa: E402
from parsing import slot_scores  # noqa: E402

HIST = os.path.join(ROOT, "eval", "history")


def _read(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


ROUTABLE = {"search", "market", "recommend", "knowledge", "email"}


def _reject_rate(preds):
    """Of the out-of-domain (gold unknown) cases, fraction correctly judged unknown."""
    out = [p for p in preds if "unknown" in p["gold"]]
    if not out:
        return None, 0
    correct = sum(1 for p in out if p["pred"] == "unknown")
    return round(correct / len(out), 4), len(out)


def _sweep_threshold(preds):
    """Find the global EMBED_THRESHOLD that best separates in-domain (route to the right
    intent) vs out-of-domain (fall to unknown), using the recorded classifier top score.

    We ONLY re-decide samples that went through the embedding fallback (topScore present
    and the current pred came from embedding). At threshold t: if topScore>=t -> predict
    topSkill (if routable), else unknown. Rule-decided samples keep their prediction.
    Score by overall macro-F1 over ALL samples. Returns best t + curve."""
    score_cand = [round(x / 100, 2) for x in range(35, 76)]   # 0.35 .. 0.75
    margin_cand = [round(x / 100, 2) for x in range(0, 21, 1)]  # 0.00 .. 0.20
    ind = [p for p in preds if "unknown" not in p["gold"]]
    ood = [p for p in preds if "unknown" in p["gold"]]

    def accept(p, t, mg):
        """Would the embedding guess be accepted (routed) at (score>=t AND margin>=mg)?"""
        ts, m, tk = p.get("topScore"), p.get("topMargin"), p.get("topSkill")
        return ts is not None and ts >= t and (m if m is not None else 1.0) >= mg and tk in ROUTABLE

    def final_pred(p, t, mg):
        """What the PIPELINE would answer at (t, mg) — the rules decide first, and only a
        sample that actually falls through to the embedding layer is re-decided by the gate.

        All three metrics below are computed from this, on one population. They previously
        were not: in_domain_wrong_reject / ood_reject asked "would the GATE reject this",
        over every sample, while macro_f1 asked what the pipeline would answer. Since the
        regex layer decides ~98% of in-domain traffic before the gate is consulted, the old
        in_domain_wrong_reject charged the threshold for rejecting samples the rules had
        already routed — enough to put production's own 0.58/0.05 in VIOLATION of the 5%
        constraint (0.4578) and hand the recommendation to a point with barely half its OOD
        rejection. Validation anchor: at 0.58/0.05 this definition reproduces the
        independently measured ood_reject_rate (0.6571) exactly."""
        hinge = p.get("topScore") is not None and (p.get("via") == "embedding" or p["pred"] == "unknown")
        return (p["topSkill"] if accept(p, t, mg) else "unknown") if hinge else p["pred"]

    results = []
    for t in score_cand:
        for mg in margin_cand:
            pairs = [(p["gold"][0], final_pred(p, t, mg)) for p in preds]
            f1 = classification_report(pairs)["macro_f1"]
            in_wrong = sum(1 for p in ind if final_pred(p, t, mg) == "unknown") / len(ind) if ind else 0
            ood_rej = sum(1 for p in ood if final_pred(p, t, mg) == "unknown") / len(ood) if ood else 0
            results.append({"t": t, "margin": mg, "macro_f1": round(f1, 4),
                            "in_domain_wrong_reject": round(in_wrong, 4), "ood_reject": round(ood_rej, 4)})

    f1_opt = max(results, key=lambda r: r["macro_f1"])
    ok = [r for r in results if r["in_domain_wrong_reject"] <= 0.05]
    rec = max(ok, key=lambda r: r["ood_reject"]) if ok else f1_opt   # maximize OOD reject under safety

    # A single "recommended" point is misleading when the objective is FLAT: report how wide
    # the optimum is, and how binding the constraint is. Picking one point off a broad
    # plateau with max() is picking arbitrarily, which is how an auto-tuner drifts.
    tied = [r for r in ok if r["ood_reject"] == rec["ood_reject"]]
    return {"f1_optimal": f1_opt, "recommended": rec,
            "constraint": "in-domain wrong-reject <= 5%",
            "n_grid": len(results), "n_feasible": len(ok),
            "constraint_binding": len(ok) < len(results),
            "n_tied_at_optimum": len(tied),
            "tie_span": {"t": [min(r["t"] for r in tied), max(r["t"] for r in tied)],
                         "margin": [min(r["margin"] for r in tied), max(r["margin"] for r in tied)]},
            "in_domain_reachable": sum(
                1 for p in ind
                if p.get("topScore") is not None and (p.get("via") == "embedding" or p["pred"] == "unknown")),
            "n_in_domain": len(ind), "n_ood": len(ood)}


def intent_metrics():
    preds = _read(os.path.join(HIST, "intent.preds.jsonl"))
    # accuracy: predicted intent is IN the gold set (handles multi-intent);
    # confusion/per-class use the primary gold label vs prediction.
    in_set = sum(1 for p in preds if p["pred"] in p["gold"]) / len(preds)
    pairs = [(p["gold"][0], p["pred"]) for p in preds]
    rep = classification_report(pairs)
    rep["accuracy_in_set"] = round(in_set, 4)
    rej, n_out = _reject_rate(preds)
    rep["ood_reject_rate"] = rej
    rep["n_out_of_domain"] = n_out
    rep["threshold_sweep"] = _sweep_threshold(preds)
    # surface the misses for inspection
    rep["misses"] = [{"input": p["input"], "gold": p["gold"], "pred": p["pred"]}
                     for p in preds if p["pred"] not in p["gold"]]
    return rep


def parse_metrics():
    preds = _read(os.path.join(HIST, "parse.preds.jsonl"))
    # escalation decision over ALL cases
    esc = [{"gold": {}, "pred": {}, "gold_escalate": p["gold_escalate"],
            "pred_escalate": p["pred_escalate"]} for p in preds]
    esc_acc = slot_scores(esc)["escalation_accuracy"]
    # regex filter quality over NON-escalate cases (the ones regex is meant to own)
    non_esc = [{"gold": p["gold_filter"], "pred": p["pred_filter"]}
               for p in preds if not p["gold_escalate"]]
    fq = slot_scores(non_esc)
    return {
        "n": len(preds),
        "escalation_accuracy": esc_acc,
        "regex_exact_match_non_escalate": fq["exact_match"],
        "regex_slot_f1_non_escalate": fq["slot_f1"],
        "n_non_escalate": fq["n"],
        "escalation_misses": [{"input": p["input"], "gold": p["gold_escalate"], "pred": p["pred_escalate"]}
                              for p in preds if bool(p["gold_escalate"]) != bool(p["pred_escalate"])],
    }


def main():
    meta = {}
    meta_path = os.path.join(HIST, "intent.meta.json")
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path))
    intent = intent_metrics()
    parse = parse_metrics()

    print("\n=== INTENT ===")
    print(f"  live: llm={meta.get('llmLive')}, embedClassifier={meta.get('classifyLive')}")
    print(f"  accuracy (pred in gold set): {intent['accuracy_in_set']}")
    print(f"  accuracy (primary label):    {intent['accuracy']}   macro-F1: {intent['macro_f1']}   n={intent['n']}")
    print(f"  out-of-domain reject rate:   {intent['ood_reject_rate']}  (n_ood={intent['n_out_of_domain']})")
    sw = intent["threshold_sweep"]
    f1o, rec = sw["f1_optimal"], sw["recommended"]
    print(f"  threshold+margin sweep:")
    print(f"    f1-optimal: t={f1o['t']} margin={f1o['margin']} -> macro-F1 {f1o['macro_f1']}, "
          f"OOD-reject {f1o['ood_reject']}, in-domain-wrong-reject {f1o['in_domain_wrong_reject']}")
    print(f"    RECOMMENDED ({sw['constraint']}): t={rec['t']} margin={rec['margin']} -> "
          f"macro-F1 {rec['macro_f1']}, OOD-reject {rec['ood_reject']}, in-domain-wrong-reject {rec['in_domain_wrong_reject']}")
    if intent["misses"]:
        print(f"  misses ({len(intent['misses'])}):")
        for m in intent["misses"][:12]:
            print(f"    - {m['gold']} != {m['pred']:<10} | {m['input']}")

    print("\n=== PARSE ===")
    print(f"  escalation-decision accuracy (all {parse['n']}): {parse['escalation_accuracy']}")
    print(f"  regex exact-match (non-escalate {parse['n_non_escalate']}): {parse['regex_exact_match_non_escalate']}")
    print(f"  regex slot-F1 (non-escalate): {parse['regex_slot_f1_non_escalate']}")
    if parse["escalation_misses"]:
        print(f"  escalation misses ({len(parse['escalation_misses'])}):")
        for m in parse["escalation_misses"][:12]:
            print(f"    - want escalate={m['gold']} got {m['pred']} | {m['input']}")

    result = {"intent": intent, "parse": parse, "meta": meta}
    out = os.path.join(HIST, "intent_parse.metrics.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nmetrics written to {out}")
    return result


if __name__ == "__main__":
    main()
