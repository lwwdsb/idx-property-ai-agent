"""Tuning loop #1 — the deterministic router's intent gate (threshold x margin).

This loop exists to VERIFY THE MECHANISM, not to chase a number. It tunes the only
parameter pair that is objective (intent.jsonl carries human labels), blast-radius closed
(auto routes tools by function calling and never consults intent classification), free to
evaluate (local embeddings, no tokens), trivially reversible (two numbers) — and whose
answer is already known, because 0.58/0.05 were swept by hand. A loop that makes decisions
on its own cannot be trusted until it first reproduces one you can check.

  python eval/runners/loop_intent.py                 # measure + decide, change nothing
  python eval/runners/loop_intent.py --repeats 3
  python eval/runners/loop_intent.py --apply         # also write the winner via `npm run tune`

WHY THE GATE IS A PAIRED TEST. Candidate and incumbent are scored on the SAME samples, so
the comparison is paired and the right question is not "is 0.77 bigger than 0.66" but "how
many individual queries changed, and in which direction". With 35 out-of-domain queries a
+0.11 difference can be four of them. McNemar's exact test asks exactly that, and it is the
discipline the retrieval work already arrived at the hard way: an unpaired point estimate on
a small set is how a loop accumulates noise and calls it progress.

THREE WAYS THIS REFUSES TO ACT, each for a different reason:
  flat objective  — a wide tie at the optimum means `max()` is choosing arbitrarily
  not stable      — repeats disagree, so something non-deterministic is leaking in
  not significant — the paired test cannot separate candidate from incumbent
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from provenance import build as build_stamp                      # noqa: E402
from report_intent_parse import _sweep_threshold, _read          # noqa: E402
from stats import mcnemar_exact                                  # noqa: E402  (shared with
#                                          score_pairwise.py — one implementation, no drift)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HIST = os.path.join(ROOT, "eval", "history")
TUNING = os.path.join(ROOT, "config", "tuning.json")
ROUTABLE = {"search", "market", "recommend", "knowledge", "email"}
MAX_TIE = 8          # a plateau wider than this means the optimum is not a point
GRID_STEP = 0.01     # so "wider than a few grid steps" is judged in parameter units


def current_params():
    cfg = json.load(open(TUNING, encoding="utf-8"))["deterministic"]["intent"]
    return round(cfg["embedThreshold"], 2), round(cfg["embedMargin"], 2)


def generate_predictions():
    """Re-run the real pipeline. This is the ONLY source of run-to-run variance here: the
    grid is a deterministic re-scoring of these predictions, so repeating the grid would
    prove nothing. The embedding scores are deterministic; the LLM parse step is not."""
    r = subprocess.run(["npx", "tsx", "eval/runners/evalIntentParse.ts"],
                       cwd=ROOT, capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        raise RuntimeError(f"evalIntentParse failed:\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
    return _read(os.path.join(HIST, "intent.preds.jsonl"))


def gate_decisions(preds, t, mg):
    """Per-sample: would the pipeline answer 'unknown' at (t, mg)? Mirrors final_pred."""
    def accept(p):
        ts, m, tk = p.get("topScore"), p.get("topMargin"), p.get("topSkill")
        return ts is not None and ts >= t and (m if m is not None else 1.0) >= mg and tk in ROUTABLE
    out = {}
    for p in preds:
        hinge = p.get("topScore") is not None and (p.get("via") == "embedding" or p["pred"] == "unknown")
        out[p["id"]] = ((p["topSkill"] if accept(p) else "unknown") if hinge else p["pred"]) == "unknown"
    return out


def one_pass(preds, cur_t, cur_mg):
    """_sweep_threshold gives the candidate + the plateau diagnostics but only summary rows,
    so the incumbent's own scores are rebuilt here from the same per-sample decisions the
    gate uses — one grid pass, and the assert ties the two implementations together."""
    sw = _sweep_threshold(preds)
    ind = [p for p in preds if "unknown" not in p["gold"]]
    ood = [p for p in preds if "unknown" in p["gold"]]
    grid = {}
    for t in [round(x / 100, 2) for x in range(35, 76)]:
        for mg in [round(x / 100, 2) for x in range(0, 21)]:
            d = gate_decisions(preds, t, mg)
            grid[(t, mg)] = {
                "t": t, "margin": mg,
                "in_domain_wrong_reject": round(sum(1 for p in ind if d[p["id"]]) / len(ind), 4),
                "ood_reject": round(sum(1 for p in ood if d[p["id"]]) / len(ood), 4)}
    assert sw["n_grid"] == len(grid), "grid shape disagrees with the sweep's"
    rec = sw["recommended"]
    # Same point, two implementations — a mismatch means gate_decisions and final_pred drifted.
    mine = grid[(rec["t"], rec["margin"])]
    assert mine["ood_reject"] == rec["ood_reject"], "gate_decisions disagrees with final_pred"
    return {"sweep": sw, "recommended": rec, "current": grid[(cur_t, cur_mg)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=2, help="prediction regenerations (variance check)")
    ap.add_argument("--apply", action="store_true", help="write the winner via `npm run tune`")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    cur_t, cur_mg = current_params()
    print(f"incumbent: t={cur_t} margin={cur_mg}   repeats={args.repeats}   alpha={args.alpha}\n")

    passes = []
    for i in range(args.repeats):
        print(f"[pass {i + 1}/{args.repeats}] regenerating predictions...", flush=True)
        preds = generate_predictions()
        p = one_pass(preds, cur_t, cur_mg)
        p["preds"] = preds
        passes.append(p)
        r, c = p["recommended"], p["current"]
        print(f"  candidate t={r['t']} mg={r['margin']} ood={r['ood_reject']} iw={r['in_domain_wrong_reject']}"
              f"   | incumbent ood={c['ood_reject']} iw={c['in_domain_wrong_reject']}")

    # --- stability across repeats -------------------------------------------------
    cands = {(p["recommended"]["t"], p["recommended"]["margin"]) for p in passes}
    cur_ood = {p["current"]["ood_reject"] for p in passes}
    stable = len(cands) == 1 and len(cur_ood) == 1

    last, sw = passes[-1], passes[-1]["sweep"]
    rec, cur = last["recommended"], last["current"]
    span = sw["tie_span"]
    tie_wide = (span["t"][1] - span["t"][0] > MAX_TIE * GRID_STEP
                or span["margin"][1] - span["margin"][0] > MAX_TIE * GRID_STEP)

    # --- paired test on the out-of-domain queries --------------------------------
    ood_ids = [p["id"] for p in last["preds"] if "unknown" in p["gold"]]
    dc = gate_decisions(last["preds"], rec["t"], rec["margin"])
    di = gate_decisions(last["preds"], cur_t, cur_mg)
    b = sum(1 for i in ood_ids if dc[i] and not di[i])      # candidate rejects, incumbent doesn't
    c = sum(1 for i in ood_ids if di[i] and not dc[i])
    pval = mcnemar_exact(b, c)

    ind_ids = [p["id"] for p in last["preds"] if "unknown" not in p["gold"]]
    ind_regress = sum(1 for i in ind_ids if dc[i] and not di[i])   # newly mis-rejected in-domain

    # --- decision ---------------------------------------------------------------
    reasons = []
    if not stable:
        reasons.append(f"NOT STABLE across {args.repeats} passes (candidates={sorted(cands)}, "
                       f"incumbent ood={sorted(cur_ood)}) — something non-deterministic is leaking in")
    if tie_wide:
        reasons.append(f"FLAT OBJECTIVE: {sw['n_tied_at_optimum']} grid points tie at ood={rec['ood_reject']} "
                       f"(t {span['t']}, margin {span['margin']}) — max() is picking arbitrarily, "
                       f"so no single point is defensible")
    if not sw["constraint_binding"]:
        reasons.append(f"CONSTRAINT VACUOUS: all {sw['n_grid']} points satisfy it, because only "
                       f"{sw['in_domain_reachable']}/{sw['n_in_domain']} in-domain samples reach the gate "
                       f"(the regex layer decides the rest) — it cannot discriminate")
    if pval > args.alpha:
        reasons.append(f"NOT SIGNIFICANT: McNemar exact p={pval:.4f} > {args.alpha} on {b + c} discordant "
                       f"of {len(ood_ids)} OOD queries (b={b}, c={c})")
    if ind_regress:
        reasons.append(f"IN-DOMAIN REGRESSION: {ind_regress} in-domain queries newly rejected")
    if (rec["t"], rec["margin"]) == (cur_t, cur_mg):
        reasons.append("candidate == incumbent (nothing to change)")

    keep = not reasons
    print("\n" + "=" * 74)
    print(f"DECISION: {'APPLY' if keep else 'NO CHANGE'}   candidate t={rec['t']} margin={rec['margin']}"
          f"  vs incumbent t={cur_t} margin={cur_mg}")
    print(f"  ood_reject {cur['ood_reject']} -> {rec['ood_reject']}"
          f"   (discordant OOD: +{b} / -{c}, McNemar p={pval:.4f})")
    for r in reasons:
        print(f"  - {r}")
    print("=" * 74)

    out = {"at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
           "provenance": build_stamp(), "repeats": args.repeats, "stable": stable,
           "incumbent": {"t": cur_t, "margin": cur_mg, **{k: cur[k] for k in ("ood_reject", "in_domain_wrong_reject")}},
           "candidate": rec, "sweep_diagnostics": {k: sw[k] for k in
                                                   ("n_grid", "n_feasible", "constraint_binding",
                                                    "n_tied_at_optimum", "tie_span", "in_domain_reachable",
                                                    "n_in_domain", "n_ood")},
           "paired_test": {"test": "mcnemar_exact_two_sided", "b": b, "c": c, "p": round(pval, 6),
                           "alpha": args.alpha},
           "in_domain_regression": ind_regress, "decision": "apply" if keep else "no_change",
           "reasons": reasons}
    with open(os.path.join(HIST, "loop_intent.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(f"\nwrote eval/history/loop_intent.json")

    if keep and args.apply:
        for key, val in (("deterministic.intent.embedThreshold", rec["t"]),
                         ("deterministic.intent.embedMargin", rec["margin"])):
            subprocess.run(["npm", "run", "-s", "tune", "--", "set", key, str(val)], cwd=ROOT, check=True)
        print("applied via `npm run tune` (rollback: npm run tune -- rollback)")
    elif keep:
        print("re-run with --apply to write it")


if __name__ == "__main__":
    main()
