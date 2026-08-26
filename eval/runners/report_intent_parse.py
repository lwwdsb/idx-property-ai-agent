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
    cand = [round(x / 100, 2) for x in range(35, 76)]   # 0.35 .. 0.75
    curve = {}
    # in-domain vs OOD samples that hinge on the embedding score
    ind = [p for p in preds if "unknown" not in p["gold"] and p.get("topScore") is not None]
    ood = [p for p in preds if "unknown" in p["gold"] and p.get("topScore") is not None]

    best_f1, best_f1_t = -1.0, None
    for t in cand:
        pairs = []
        for p in preds:
            gold = p["gold"][0]
            ts, tk, via = p.get("topScore"), p.get("topSkill"), p.get("via")
            hinge = ts is not None and (via == "embedding" or p["pred"] == "unknown")
            pred = (tk if (ts >= t and tk in ROUTABLE) else "unknown") if hinge else p["pred"]
            pairs.append((gold, pred))
        f1 = classification_report(pairs)["macro_f1"]
        in_wrong = sum(1 for p in ind if p["topScore"] < t) / len(ind) if ind else 0
        ood_rej = sum(1 for p in ood if p["topScore"] < t) / len(ood) if ood else 0
        curve[t] = {"macro_f1": round(f1, 4), "in_domain_wrong_reject": round(in_wrong, 4),
                    "ood_reject": round(ood_rej, 4)}
        if f1 > best_f1:
            best_f1, best_f1_t = f1, t

    # RECOMMENDED threshold: best macro-F1 SUBJECT TO in-domain wrong-reject <= 5%
    # (raw macro-F1 optimum over-rejects real queries — unacceptable in production).
    ok = [(t, c) for t, c in curve.items() if c["in_domain_wrong_reject"] <= 0.05]
    rec_t = max(ok, key=lambda tc: tc[1]["macro_f1"])[0] if ok else best_f1_t
    return {
        "f1_optimal_threshold": best_f1_t, "f1_optimal_macro_f1": round(best_f1, 4),
        "recommended_threshold": rec_t, "recommended": curve[rec_t],
        "constraint": "in-domain wrong-reject <= 5%",
        "curve": curve,
    }


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
    cur = sw["curve"].get(0.55, {})
    print(f"  threshold sweep:")
    print(f"    f1-optimal t={sw['f1_optimal_threshold']} (macro-F1 {sw['f1_optimal_macro_f1']}) — but over-rejects in-domain")
    print(f"    RECOMMENDED t={sw['recommended_threshold']} ({sw['constraint']}): "
          f"macro-F1 {sw['recommended']['macro_f1']}, OOD-reject {sw['recommended']['ood_reject']}, "
          f"in-domain-wrong-reject {sw['recommended']['in_domain_wrong_reject']}")
    print(f"    current 0.55: macro-F1 {cur.get('macro_f1')}, OOD-reject {cur.get('ood_reject')}")
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
