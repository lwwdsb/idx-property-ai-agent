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


def intent_metrics():
    preds = _read(os.path.join(HIST, "intent.preds.jsonl"))
    # accuracy: predicted intent is IN the gold set (handles multi-intent);
    # confusion/per-class use the primary gold label vs prediction.
    in_set = sum(1 for p in preds if p["pred"] in p["gold"]) / len(preds)
    pairs = [(p["gold"][0], p["pred"]) for p in preds]
    rep = classification_report(pairs)
    rep["accuracy_in_set"] = round(in_set, 4)
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
