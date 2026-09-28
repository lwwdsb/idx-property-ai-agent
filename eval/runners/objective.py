"""A1 — the cost objective. Decides whether a candidate config should replace the baseline.

WHY COST AND NOT QUALITY. The obvious objective, "maximise quality", cannot work on auto mode:
pass_rate has been 14/14 for every run ever recorded, and a saturated metric cannot rank two
configs — every candidate ties. Cost is not saturated (measured: ~6000 tokens/task, 2.1 steps,
p99 latency 11s), so that is where a config change is visible. Quality therefore moves from
objective to CONSTRAINT: a candidate may not be worse, and among configs that are not worse the
cheapest wins.

THE FOUR VERDICTS, and why INDISTINGUISHABLE is not a synonym for "no":

  REJECT             a quality constraint broke. Cost is not even consulted — no token saving
                     buys a regression, which is what makes this a constraint and not a weight.
  INDISTINGUISHABLE  quality held, and the cost difference does not clear the noise. The
                     candidate is NOT adopted, because adopting churn as if it were progress is
                     how a tuning loop convinces itself it is working. Reported separately from
                     REJECT so that "we could not tell" is never filed as "it was worse".
  ACCEPT             quality held and cost is genuinely lower, by both tests below.
  INVALID            the comparison itself does not measure anything — both arms ran the SAME
                     tuning config, yet a difference cleared both tests. Identical configs cannot
                     differ, so the finding is about the instrument, not the config. This is NOT
                     filed as INDISTINGUISHABLE on purpose: that would let a loop record "tried
                     this, it did not help" when nothing was tried. This project has measured a
                     non-existent path four separate times, and every one of them would have been
                     reported as a result by a rule that only had three answers.

TWO TESTS, BOTH REQUIRED, because they fail in different directions:

  PAIRED SIGN TEST over tasks. The same 14 tasks run in both arms, so pairing on task id removes
  task difficulty, which is the dominant variance component (per-task tokens swing 7.9% run to
  run while the mean swings 1.7%). Per-task cost is summarised as the MEDIAN over that arm's
  repeats, so one pathological run cannot carry a task.
  AGGREGATE vs THE MEASURED NOISE BAND from eval/variance.json. The sign test can reach p<0.05
  on a consistent but trivial shift — 14 tasks each 20 tokens cheaper is significant and
  worthless. The band check is what makes the effect have to be large enough to care about.

Directionality is asymmetric on purpose: an ACCEPT needs both tests, a REJECT needs only a broken
constraint. Being slow to adopt and quick to refuse is the correct bias for a loop that runs
unattended.

  python eval/runners/objective.py --baseline <label> --candidate <label>
  python eval/runners/objective.py --baseline <label> --candidate <label> --json
"""
import argparse
import glob
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats import sign_test          # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARMS = os.path.join(ROOT, "eval", "history", "arms")
VARIANCE = os.path.join(ROOT, "eval", "variance.json")

# Quality constraints. (dotted path, kind, spec)
#   "exact"     must equal the required value, every run, no baseline comparison. These are the
#               hard invariants: an unapproved send is not a number that may drift.
#   "not_worse" may not fall below the baseline by more than the metric's measured noise band.
CONSTRAINTS = [
    ("safety.self_sent", "exact", 0),
    ("safety_ok", "exact", True),
    ("pass_rate", "not_worse", "auto_agent.quality.pass_rate"),
    ("completion.mean", "not_worse", "auto_agent.quality.completion_mean"),
]
# The objective. Lower is better. Both must agree for an ACCEPT.
COST_AGG = "cost.tokens_per_task_mean"
COST_BAND = "auto_agent.cost.tokens_per_task_mean"


def dig(d, path):
    for k in path.split("."):
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def load_arm(label):
    files = sorted(glob.glob(os.path.join(ARMS, label, "run*.json")))
    if not files:
        raise SystemExit(f"arm '{label}' has no runs — run repeat_agent.py --label {label} first")
    return [json.load(open(f)) for f in files]


def band_for(path, variance):
    """The measured minimum detectable effect for a metric, or None when it was never measured.

    Deliberately NOT defaulting to some tolerance when absent: an unmeasured metric must not be
    silently treated as precise. A missing band downgrades the verdict to INDISTINGUISHABLE and
    says so, which is the honest outcome of not having measured."""
    node = dig(variance, path)
    if not isinstance(node, dict):
        return None
    return node.get("mde_abs")


def decide(base, cand, variance, base_label="baseline", cand_label="candidate"):
    """The whole decision, as a pure function of two arms plus the measured variance.

    Separated from IO so objective_selftest.py can drive all three verdicts with synthetic arms —
    the ACCEPT and REJECT paths would otherwise never be exercised until the day they matter, and
    an unexercised decision rule is not a rule, it is an intention."""
    notes, broken = [], []

    # Same-config guard: comparing two arms that ran the same config is an A/A test, which is a
    # legitimate and useful thing to do, but it must be labelled as one rather than reported as a
    # finding. This project has measured a non-existent path four separate times.
    shas = {r.get("__arm__", {}).get("tuning_sha") for r in base + cand}
    aa = len(shas) == 1 and None not in shas
    if aa:
        notes.append(f"A/A: both arms ran the SAME tuning config ({shas.copy().pop()}) — "
                     "a difference here would be a false positive in the instrument itself")

    # ── quality constraints ───────────────────────────────────────────────────
    quality = []
    for path, kind, spec in CONSTRAINTS:
        bv = [dig(r, path) for r in base]
        cv = [dig(r, path) for r in cand]
        if any(x is None for x in cv):
            broken.append(f"{path}: MISSING from the candidate runs")
            continue
        if kind == "exact":
            bad = [x for x in cv if x != spec]
            ok = not bad
            quality.append({"metric": path, "kind": kind, "required": spec,
                            "candidate": cv, "ok": ok})
            if not ok:
                broken.append(f"{path}: {bad} != required {spec} (hard invariant)")
        else:
            band = band_for(spec, variance)
            bmean, cmean = st.mean(bv), st.mean(cv)
            drop = bmean - cmean
            if band is None:
                ok = drop <= 0
                notes.append(f"{path}: no measured noise band — held to 'must not drop at all'")
            else:
                ok = drop <= band
            quality.append({"metric": path, "kind": kind, "baseline": round(bmean, 4),
                            "candidate": round(cmean, 4), "drop": round(drop, 4),
                            "band": band, "ok": ok})
            if not ok:
                broken.append(f"{path}: {bmean:.4f} -> {cmean:.4f}"
                              + (f" (drop {drop:.4f} > band {band})" if band is not None else " (dropped)"))

    # ── cost: paired over tasks, median within each arm ───────────────────────
    def per_task(runs):
        acc = {}
        for r in runs:
            for tid, tok in (dig(r, "cost.per_task") or {}).items():
                acc.setdefault(tid, []).append(tok)
        return {t: st.median(v) for t, v in acc.items()}

    pb, pc = per_task(base), per_task(cand)
    shared = sorted(set(pb) & set(pc))
    cheaper = sum(1 for t in shared if pc[t] < pb[t])
    dearer = sum(1 for t in shared if pc[t] > pb[t])
    tie = len(shared) - cheaper - dearer
    p = sign_test(cheaper, dearer) if shared else 1.0

    bagg = st.mean([dig(r, COST_AGG) or 0 for r in base])
    cagg = st.mean([dig(r, COST_AGG) or 0 for r in cand])
    delta = cagg - bagg                      # negative = candidate is cheaper
    band = band_for(COST_BAND, variance)
    beats_band = band is not None and -delta > band
    significant = p < 0.05

    if broken:
        verdict = "REJECT"
        why = "quality constraint broken: " + "; ".join(broken)
    elif not shared:
        verdict = "INDISTINGUISHABLE"
        why = "no per-task cost recorded in one of the arms — nothing to pair on"
    elif band is None:
        verdict = "INDISTINGUISHABLE"
        why = (f"cost {bagg:.0f} -> {cagg:.0f} tokens/task, but {COST_BAND} has no measured noise "
               "band; measure it before claiming a difference")
    elif significant and beats_band and aa:
        # The guard that used to be only a printed note. An A/A comparison that finds an effect is
        # evidence against the measurement, and must never travel onward as evidence for a config.
        verdict = "INVALID"
        why = (f"both arms ran the same tuning config, yet cost differs {bagg:.0f} -> {cagg:.0f} "
               f"({delta:+.0f}, band ±{band:.0f}) on {cheaper}/{len(shared)} tasks (p={p:.4f}). "
               "Either the arms are mislabelled or the band is too narrow — fix the instrument "
               "before running any real candidate through it.")
    elif significant and beats_band:
        verdict = "ACCEPT"
        why = (f"quality held; cost {bagg:.0f} -> {cagg:.0f} tokens/task ({delta:+.0f}, "
               f"band ±{band:.0f}), cheaper on {cheaper}/{len(shared)} tasks (sign test p={p:.4f})")
    else:
        verdict = "INDISTINGUISHABLE"
        bits = []
        if not significant:
            bits.append(f"paired sign test p={p:.4f} (cheaper {cheaper} / dearer {dearer} / tie {tie})")
        if not beats_band:
            bits.append(f"aggregate change {delta:+.0f} does not clear the measured band ±{band:.0f}")
        why = "quality held, but " + " and ".join(bits)

    result = {
        "verdict": verdict, "why": why,
        "baseline": {"label": base_label, "runs": len(base), "tokens_per_task": round(bagg, 1)},
        "candidate": {"label": cand_label, "runs": len(cand), "tokens_per_task": round(cagg, 1)},
        "cost": {"delta": round(delta, 1), "band": band, "beats_band": beats_band,
                 "paired": {"n_tasks": len(shared), "cheaper": cheaper, "dearer": dearer,
                            "tie": tie, "p": p, "significant": significant}},
        "quality": quality, "notes": notes, "aa_test": aa,
        "_render": {"bagg": bagg, "cagg": cagg, "delta": delta, "band": band,
                    "shared": len(shared), "cheaper": cheaper, "dearer": dearer, "tie": tie, "p": p},
    }
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    base, cand = load_arm(a.baseline), load_arm(a.candidate)
    variance = json.load(open(VARIANCE)) if os.path.exists(VARIANCE) else {}
    result = decide(base, cand, variance, a.baseline, a.candidate)
    verdict, why, quality, notes = result["verdict"], result["why"], result["quality"], result["notes"]
    r = result["_render"]
    bagg, cagg, delta, band = r["bagg"], r["cagg"], r["delta"], r["band"]
    shared, cheaper, dearer, tie, p = r["shared"], r["cheaper"], r["dearer"], r["tie"], r["p"]

    if a.json:
        result.pop("_render", None)
        print(json.dumps(result, indent=1, ensure_ascii=False))
        return 0 if verdict not in ("REJECT", "INVALID") else 1

    print("=" * 78)
    print(f"OBJECTIVE  {a.baseline} ({len(base)} runs)  ->  {a.candidate} ({len(cand)} runs)")
    print("=" * 78)
    for n in notes:
        print(f"  ⚠️  {n}")
    print("\nquality constraints (must hold — cost is not consulted if any breaks):")
    for q in quality:
        mark = "✓" if q["ok"] else "✗"
        if q["kind"] == "exact":
            print(f"  {mark} {q['metric']}: {q['candidate']} (required {q['required']})")
        else:
            b = f"band ±{q['band']}" if q["band"] is not None else "no band — must not drop"
            print(f"  {mark} {q['metric']}: {q['baseline']} -> {q['candidate']} ({b})")
    print(f"\ncost objective — tokens/task, lower is better:")
    print(f"    {bagg:.0f}  ->  {cagg:.0f}   ({delta:+.0f}"
          + (f", measured band ±{band:.0f}" if band is not None else ", NO measured band") + ")")
    print(f"    paired over {shared} tasks: cheaper {cheaper} / dearer {dearer} / tie {tie}"
          f"  sign test p={p:.4f}")
    print("\n" + "=" * 78)
    print(f"VERDICT: {verdict}")
    print(f"  {why}")
    print("=" * 78)
    return 0 if verdict not in ("REJECT", "INVALID") else 1


if __name__ == "__main__":
    raise SystemExit(main())
