"""Tuning loop #2 — the retrieval knobs (cross-encoder pool size x per-path prefetch depth).

Loop #1 tuned a DETERMINISTIC-ONLY parameter whose answer was already known, to prove the
mechanism. This one tunes SHARED parameters with real headroom, so the gate is stricter: a
shared knob reaches both paths through the same SkillRegistry, and an improvement on one path
that costs the other is not an improvement.

THREE CHANNELS, all objective (the gold is "this exact listing", no judge anywhere):
  generated     171 rows, gold filter + the REAL extractSemanticText output (precomputed by
                genSemanticPreds.ts — no LLM, since the gold filter is ground truth)
  human/regex    39 rows, the DETERMINISTIC path's real extraction (regex_filter/_semantic)
  human/auto     39 rows, the AUTO path's real extraction (auto_filter/_semantic)
The two human channels reuse eval/history/mode_retrieval.preds.jsonl, so the upstream LLM
extraction is FROZEN and paid for once: the sweep varies only the retrieval knobs, and no
candidate costs an LLM call. It also means a difference between channels is attributable to
the knobs rather than to the model having a different day.

Cases whose semantic text is empty are EXCLUDED from the human channels: in production those
never reach Qdrant at all (the skill falls through to structured MySQL), so a retrieval knob
cannot affect them and including them would only dilute the measurement.

topK is NOT swept. tuning's shared.retrieval.topK is merely the default for SearchReq.k, and
every production caller passes k explicitly (skills.ts sends k=10), so changing it moves
nothing — the same shape of finding as RRF's k being fixed inside Qdrant.

Recall per candidate is DETERMINISTIC here (fixed index, fixed inputs, deterministic HNSW and
cross-encoder), so candidates are not repeated for recall; repeats would only re-measure
latency, which is timed separately.

  python eval/runners/loop_retrieval.py                 # measure + decide, change nothing
  python eval/runners/loop_retrieval.py --apply         # write the winner via `npm run tune`
"""
import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from provenance import build as build_stamp        # noqa: E402
from stats import mcnemar_exact                    # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "eval", "datasets")
HIST = os.path.join(ROOT, "eval", "history")
TUNING = os.path.join(ROOT, "config", "tuning.json")
URL = os.environ.get("RETRIEVAL_URL", "http://localhost:8099")
TYPE_DB = {"condo": "Condominium", "townhouse": "Townhouse", "single-family": "SingleFamilyResidence"}
LATENCY_BUDGET_MS = 200        # the line already judged acceptable when rerank went live


def post(path, body, timeout=180):
    req = urllib.request.Request(f"{URL}{path}", method="POST", data=json.dumps(body).encode(),
                                headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def jsonl(name, base=DATA):
    p = os.path.join(base, name)
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if os.path.exists(p) else []


def strip_city(text, city):
    """Mirrors extractSemanticText's first step: the city travels as a hard filter, not as
    semantic text (see eval_knownitem_service.py for what sending the raw query cost)."""
    if not city:
        return text
    out = re.sub(re.escape(city), " ", text, flags=re.I)
    return re.sub(r"\s+", " ", out).strip() or text


def build_channels():
    gen = [c for c in jsonl("mode_retrieval_gen.jsonl") if (c.get("gold") or {}).get("known_item")]
    gpred = {p["id"]: p for p in jsonl("mode_retrieval_gen.preds.jsonl", HIST)}
    hum = {c["id"]: c for c in jsonl("mode_retrieval.jsonl") if (c.get("gold") or {}).get("known_item")}
    preds = {p["id"]: p for p in jsonl("mode_retrieval.preds.jsonl", HIST)}

    ch = {"generated": [], "human/regex": [], "human/auto": []}
    stale_gen = 0
    for c in gen:
        f = c["gold"].get("filter") or {}
        p = gpred.get(c["id"])
        if p is None:
            # Fall back to the crude approximation, but say so: it deletes only the city, while
            # production also strips normalized numbers/units and filler, which on a Chinese
            # query is most of the string.
            stale_gen += 1
            text = strip_city(c["input"], f.get("city"))
        else:
            text = p["semantic"]
            if not text:
                continue          # never reaches Qdrant in production
        ch["generated"].append((c["id"], text, f, c["gold"]["known_item"]))
    if stale_gen:
        print(f"  WARNING: {stale_gen} generated rows have no precomputed semantic text — "
              f"run `npx tsx eval/runners/genSemanticPreds.ts` (using the crude approximation)")
    missing_preds = 0
    for cid, c in hum.items():
        p = preds.get(cid)
        if not p:
            missing_preds += 1
            continue
        for name, fkey, skey in (("human/regex", "regex_filter", "regex_semantic"),
                                 ("human/auto", "auto_filter", "auto_semantic")):
            text = (p.get(skey) or "").strip()
            if not text:
                continue          # never reaches Qdrant in production
            ch[name].append((cid, text, p.get(fkey) or {}, c["gold"]["known_item"]))
    if missing_preds:
        print(f"  note: {missing_preds} human cases have no prediction row "
              f"(run evalModeRetrieval.ts to refresh)")
    return ch


def run(channel, coarse, prefetch, k=10):
    """-> {case_id: hit}, plus per-request latencies."""
    hits, lat = {}, []
    for cid, text, f, gold in channel:
        body = {"text": text, "k": k, "coarse": coarse, "prefetch": prefetch,
                "city": f.get("city"), "max_price": f.get("maxPrice"),
                "min_price": f.get("minPrice"), "min_beds": f.get("beds"),
                "pool": f.get("pool"),
                "ptype": TYPE_DB.get(f.get("propertyType")) if f.get("propertyType") else None}
        t0 = time.perf_counter()
        res = post("/search", body).get("results", [])
        lat.append((time.perf_counter() - t0) * 1000)
        hits[cid] = int(gold in [r.get("listing_id") for r in res])
    return hits, lat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coarse", type=int, nargs="*", default=[10, 15, 20, 30, 50])
    ap.add_argument("--prefetch", type=int, nargs="*", default=[20, 30, 50])
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    cfg = json.load(open(TUNING, encoding="utf-8"))["shared"]["retrieval"]
    inc = (cfg["rerankCoarse"], cfg["prefetch"])
    with urllib.request.urlopen(f"{URL}/health", timeout=10) as r:
        health = json.loads(r.read())
    print(f"service {URL} {health}\nincumbent: coarse={inc[0]} prefetch={inc[1]}\n")

    ch = build_channels()
    for name, rows in ch.items():
        print(f"  channel {name:12} {len(rows)} cases")
    print()

    grid = [(c, p) for c in args.coarse for p in args.prefetch]
    if inc not in grid:
        grid.append(inc)
    results = {}
    for i, (c, p) in enumerate(grid, 1):
        per, lats = {}, []
        for name, rows in ch.items():
            h, l = run(rows, c, p)
            per[name] = h
            lats += l
        recalls = {n: round(sum(h.values()) / len(h), 4) if h else 0 for n, h in per.items()}
        results[(c, p)] = {"hits": per, "recall": recalls,
                           "p50_ms": round(statistics.median(lats), 1)}
        star = "  <- incumbent" if (c, p) == inc else ""
        print(f"  [{i:>2}/{len(grid)}] coarse={c:>3} prefetch={p:>3}  "
              + "  ".join(f"{n.split('/')[-1]}={v:.3f}" for n, v in recalls.items())
              + f"  p50={results[(c,p)]['p50_ms']}ms{star}", flush=True)

    base = results[inc]
    # A shared knob must not trade one path for the other, so every channel is tested
    # separately and any significant loss disqualifies the candidate outright.
    cand = []
    for key, r in results.items():
        if key == inc:
            continue
        verdict = {"coarse": key[0], "prefetch": key[1], "recall": r["recall"],
                   "p50_ms": r["p50_ms"], "channels": {}, "blockers": []}
        for name in ch:
            a, b = base["hits"][name], r["hits"][name]
            ids = [i for i in a if i in b]
            gain = sum(1 for i in ids if b[i] and not a[i])
            loss = sum(1 for i in ids if a[i] and not b[i])
            pv = mcnemar_exact(gain, loss)
            verdict["channels"][name] = {"gain": gain, "loss": loss, "p": round(pv, 6)}
            if loss > gain and pv <= args.alpha:
                verdict["blockers"].append(f"{name}: significantly worse (−{loss - gain}, p={pv:.4f})")
        if r["p50_ms"] > LATENCY_BUDGET_MS:
            verdict["blockers"].append(f"latency p50 {r['p50_ms']}ms over the {LATENCY_BUDGET_MS}ms budget")
        wins = [n for n, v in verdict["channels"].items()
                if v["gain"] > v["loss"] and v["p"] <= args.alpha]
        verdict["significant_wins"] = wins
        if not wins:
            verdict["blockers"].append("no channel improves significantly")
        cand.append(verdict)

    passing = [v for v in cand if not v["blockers"]]
    print("\n" + "=" * 78)
    print(f"incumbent coarse={inc[0]} prefetch={inc[1]}  "
          + "  ".join(f"{n.split('/')[-1]}={v:.3f}" for n, v in base["recall"].items())
          + f"  p50={base['p50_ms']}ms")
    if passing:
        best = max(passing, key=lambda v: sum(v["recall"].values()))
        print(f"DECISION: APPLY coarse={best['coarse']} prefetch={best['prefetch']}  "
              f"wins on {best['significant_wins']}")
    else:
        best = None
        print("DECISION: NO CHANGE — no candidate passes every channel")
        near = sorted(cand, key=lambda v: (len(v["blockers"]), -sum(v["recall"].values())))[:3]
        for v in near:
            print(f"  closest: coarse={v['coarse']:>3} prefetch={v['prefetch']:>3} -> "
                  + "; ".join(v["blockers"]))
    print("=" * 78)

    out = {"at": time.strftime("%Y-%m-%d %H:%M:%S"), "provenance": build_stamp(),
           "incumbent": {"coarse": inc[0], "prefetch": inc[1],
                         "recall": base["recall"], "p50_ms": base["p50_ms"]},
           "channel_sizes": {n: len(r) for n, r in ch.items()},
           "latency_budget_ms": LATENCY_BUDGET_MS,
           "candidates": cand, "decision": "apply" if passing else "no_change",
           "chosen": best}
    with open(os.path.join(HIST, "loop_retrieval.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print("\nwrote eval/history/loop_retrieval.json")

    if best and args.apply:
        for k, v in (("shared.retrieval.rerankCoarse", best["coarse"]),
                     ("shared.retrieval.prefetch", best["prefetch"])):
            subprocess.run(["npm", "run", "-s", "tune", "--", "set", k, str(v)], cwd=ROOT, check=True)
        print("applied (rollback: npm run tune -- rollback). RESTART the retrieval service.")
    elif best:
        print("re-run with --apply to write it")


if __name__ == "__main__":
    main()
