"""RAG HyDE eval — does a hypothetical-document rewrite lift retrieval, and at what cost?

Compares three ways to build the EMBEDDING for retrieval, on the same rag.jsonl set:
  - baseline : embed the raw question (what ships today)
  - hyde     : embed ONLY the LLM's hypothetical answer passage (classic HyDE)
  - blend    : embed question + hypothetical (robust variant — a bad hypo can't fully derail)

Reports, per variant:
  - in-corpus hit@k (higher = better recall)
  - in/out mean top-similarity + separation, and the best refusal-threshold accuracy
    (HyDE can RAISE out-of-corpus scores too, hurting the gate — this quantifies it,
     answering "需不需要改参数").

Hypotheticals are cached (eval/history/rag_hyde_cache.json) so reruns don't re-call the LLM.
Needs LLM_API_KEY (HyDE generation). Not part of `make eval` (that stays LLM-free/fast).

  python eval/runners/eval_rag_hyde.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "retrieval"))
sys.path.insert(0, os.path.join(ROOT, "eval", "metrics"))
from rag import RagIndex, hyde_passage  # noqa: E402
from llm import llm_available  # noqa: E402
from ir import mean  # noqa: E402

DATA = os.path.join(ROOT, "eval", "datasets", "rag.jsonl")
HIST = os.path.join(ROOT, "eval", "history")
CACHE = os.path.join(HIST, "rag_hyde_cache.json")
K = 3


def load():
    with open(DATA) as f:
        return [json.loads(l) for l in f if l.strip()]


def load_cache():
    if os.path.exists(CACHE):
        with open(CACHE) as f:
            return json.load(f)
    return {}


def best_threshold(in_scores, out_scores):
    """Top-score threshold best separating in-corpus (>=t answer) vs out (<t refuse)."""
    cands = sorted(set(round(s, 3) for s in in_scores + out_scores))
    best_t, best_acc = 0.0, 0.0
    for t in cands:
        correct = sum(1 for s in in_scores if s >= t) + sum(1 for s in out_scores if s < t)
        acc = correct / (len(in_scores) + len(out_scores) or 1)
        if acc > best_acc:
            best_acc, best_t = acc, t
    return round(best_t, 3), round(best_acc, 4)


def hit(idx, question, gold, embed_text):
    got = idx.retrieve(question, K, embed_text=embed_text)
    top = max((h["score"] for h in got), default=0.0)
    ok = any(gold in h["source"].lower() for h in got) if gold else None
    return (1.0 if ok else 0.0), top


def main():
    if not llm_available():
        print("LLM_API_KEY not set — HyDE needs the LLM to generate hypotheticals. Aborting.")
        sys.exit(1)

    cases = load()
    idx = RagIndex()
    cache = load_cache()

    in_cases = [c for c in cases if c["label"]["in_corpus"]]
    out_cases = [c for c in cases if not c["label"]["in_corpus"]]
    print(f"RAG HyDE eval: {len(in_cases)} in-corpus + {len(out_cases)} out-of-corpus, k={K}\n")

    # generate + cache hypotheticals for every question
    n_gen = 0
    for c in cases:
        q = c["input"]
        if q not in cache:
            hypo = hyde_passage(q)
            if not hypo:
                print(f"  ! hypothetical generation failed for: {q!r} — will degrade to baseline")
            cache[q] = hypo or ""
            n_gen += 1
    os.makedirs(HIST, exist_ok=True)
    with open(CACHE, "w") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    print(f"hypotheticals: {len(cache)} cached ({n_gen} newly generated)\n")

    variants = {
        "baseline": lambda q: None,
        "hyde":     lambda q: (cache.get(q) or None),
        "blend":    lambda q: (f"{q}\n{cache[q]}" if cache.get(q) else None),
    }

    rows = {}
    for name, embed_of in variants.items():
        hits, in_top, out_top = [], [], []
        for c in in_cases:
            h, top = hit(idx, c["input"], c["label"]["gold_source"].lower(), embed_of(c["input"]))
            hits.append(h)
            in_top.append(top)
        for c in out_cases:
            _, top = hit(idx, c["input"], "", embed_of(c["input"]))
            out_top.append(top)
        hr = round(mean(hits), 4)
        inm, outm = round(mean(in_top), 4), round(mean(out_top), 4)
        t, acc = best_threshold(in_top, out_top)
        rows[name] = {"hit_at_k": hr, "in_mean": inm, "out_mean": outm,
                      "separation": round(inm - outm, 4), "gate_t": t, "gate_acc": acc}

    base = rows["baseline"]["hit_at_k"]
    hdr = f"{'variant':<9} {'hit@k':>7} {'in_sim':>7} {'out_sim':>8} {'sep':>7} {'gate_t':>7} {'gate_acc':>9}"
    print(hdr)
    print("-" * len(hdr))
    for name, r in rows.items():
        delta = "" if name == "baseline" else f"  ({r['hit_at_k'] - base:+.4f})"
        print(f"{name:<9} {r['hit_at_k']:>7} {r['in_mean']:>7} {r['out_mean']:>8} "
              f"{r['separation']:>7} {r['gate_t']:>7} {r['gate_acc']:>9}{delta}")

    best = max(rows, key=lambda n: (rows[n]["hit_at_k"], rows[n]["separation"]))
    print(f"\n→ best hit@k: {best} ({rows[best]['hit_at_k']}) vs baseline {base}")
    if rows[best]["hit_at_k"] <= base:
        print("  HyDE does NOT beat baseline here → keep it OFF (small factual KB, terse Q already close).")
    else:
        print(f"  HyDE lifts recall by {rows[best]['hit_at_k'] - base:+.4f}; check gate_acc didn't regress.")

    with open(os.path.join(HIST, "rag_hyde.metrics.json"), "w") as f:
        json.dump({"k": K, "n_in": len(in_cases), "n_out": len(out_cases), "variants": rows,
                   "best": best}, f, indent=2)
    print(f"\nmetrics written to {os.path.join(HIST, 'rag_hyde.metrics.json')}")


if __name__ == "__main__":
    main()
