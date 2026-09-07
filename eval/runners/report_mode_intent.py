"""
Intent scoring for the mode-retrieval set — the one place both modes are judged on the SAME
cases, so their intent ability is finally comparable (elsewhere deterministic is scored on
intent.jsonl and auto on agent.jsonl, with different labels and sizes).

  deterministic: classifyIntent's verdict          (preds: regex_intent)
  auto:          which tools the LLM chose         (preds: auto_tools; none == judged OOD)

Needs only the prediction file — no Qdrant, no DB.
Run: python eval/runners/report_mode_intent.py
"""
import json
import os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
GOLD = os.path.join(HERE, "..", "datasets", "mode_retrieval.jsonl")
PREDS = os.path.join(HERE, "..", "history", "mode_retrieval.preds.jsonl")


def rows(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def main():
    if not os.path.exists(PREDS):
        raise SystemExit(f"no predictions at {PREDS} — run: npx tsx eval/runners/evalModeRetrieval.ts")
    gold = {g["id"]: g for g in rows(GOLD)}
    preds = rows(PREDS)

    # auto has no explicit "unknown" class: calling no tool IS its out-of-domain verdict.
    def auto_intents(p):
        tools = [t for t in p.get("auto_tools", []) if t]
        return set(tools) if tools else {"unknown"}

    stats = {m: Counter() for m in ("det", "auto")}
    by_style = {}
    for p in preds:
        g = gold.get(p["id"])
        if not g:
            continue
        want = set(g["gold"].get("intents") or ["unknown"])
        got = {"det": {p.get("regex_intent") or "unknown"}, "auto": auto_intents(p)}
        style = g["style"]
        by_style.setdefault(style, {m: Counter() for m in ("det", "auto")})
        for m in ("det", "auto"):
            # primary: did we get the main intent right (gold's first / any overlap)
            hit = bool(got[m] & want)
            # strict: exactly the gold set (only meaningful for multi-intent rows)
            exact = got[m] == want
            for bucket in (stats[m], by_style[style][m]):
                bucket["n"] += 1
                bucket["hit"] += hit
                bucket["exact"] += exact
                if "unknown" in want:
                    bucket["ood_n"] += 1
                    bucket["ood_ok"] += got[m] == {"unknown"}

    def line(name, c):
        n = c["n"] or 1
        s = f"  {name:<14} n={c['n']:<3} 命中={c['hit']/n:.2f}  完全一致={c['exact']/n:.2f}"
        if c["ood_n"]:
            s += f"  域外拒识={c['ood_ok']}/{c['ood_n']}={c['ood_ok']/c['ood_n']:.2f}"
        return s

    print("=== 全集 ===")
    for m, name in (("det", "确定性"), ("auto", "auto")):
        print(line(name, stats[m]))
    print("\n=== 分层 ===")
    for style in sorted(by_style):
        print(f"[{style}]")
        for m, name in (("det", "确定性"), ("auto", "auto")):
            print(line(name, by_style[style][m]))


if __name__ == "__main__":
    main()
