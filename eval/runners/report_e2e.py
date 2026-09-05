"""End-to-end eval — METRICS stage.

Reads e2e predictions and reports overall pass rate, a safety check (no email was sent),
and per-assertion / failure detail.

  python eval/runners/report_e2e.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HIST = os.path.join(ROOT, "eval", "history")


def _read(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    preds = _read(os.path.join(HIST, "e2e.preds.jsonl"))
    meta = json.load(open(os.path.join(HIST, "e2e.meta.json"))) if os.path.exists(os.path.join(HIST, "e2e.meta.json")) else {}
    n = len(preds)
    passed = sum(1 for p in preds if p["pass"])

    # per-assertion breakdown
    kinds = {}
    for p in preds:
        for k, v in p["checks"].items():
            d = kinds.setdefault(k, [0, 0])
            d[0] += 1 if v else 0
            d[1] += 1

    result = {
        "pass_rate": round(passed / n, 4) if n else 0.0,
        "passed": passed, "n": n,
        "emails_sent": meta.get("anySent"),
        "safety_ok": meta.get("anySent") == 0,
        "per_assertion": {k: {"passed": d[0], "total": d[1]} for k, d in kinds.items()},
        "live": {"llm": meta.get("llmLive"), "classify": meta.get("classifyLive")},
        "failures": [{"input": p["input"], "note": p["note"], "expect": p["expect"],
                      "got_intent": p["got"]["intent"], "got_skill": p["got"]["skill"],
                      "checks": p["checks"]} for p in preds if not p["pass"]],
    }

    print(f"\n=== END-TO-END ===")
    print(f"  live: llm={result['live']['llm']}, classify={result['live']['classify']}")
    print(f"  pass rate: {result['pass_rate']}  ({passed}/{n})")
    print(f"  SAFETY: emails actually sent = {result['emails_sent']}  -> {'OK (none)' if result['safety_ok'] else 'FAIL!'}")
    print(f"  per-assertion: " + ", ".join(f"{k} {d['passed']}/{d['total']}" for k, d in result["per_assertion"].items()))
    if result["failures"]:
        print(f"  failures ({len(result['failures'])}):")
        for f in result["failures"]:
            bad = [k for k, v in f["checks"].items() if not v]
            print(f"    ✗ [{','.join(bad)}] want {f['expect']} | got intent={f['got_intent']} skill={f['got_skill']} | {f['input']}")

    with open(os.path.join(HIST, "e2e.metrics.json"), "w") as fp:
        json.dump(result, fp, indent=2, ensure_ascii=False)
    print(f"\nmetrics written to {os.path.join(HIST, 'e2e.metrics.json')}")


if __name__ == "__main__":
    main()
