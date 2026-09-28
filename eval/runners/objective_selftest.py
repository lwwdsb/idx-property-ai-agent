"""Self-test for the cost objective's decision rule. 0 LLM calls, fully synthetic.

The A/A test (two real arms of the same config) exercises only the INDISTINGUISHABLE path. ACCEPT
and REJECT would otherwise stay unexercised until the day a loop depends on them, and an
unexercised decision rule is not a rule but an intention. Each case here fabricates two arms that
differ in exactly ONE way, so a wrong verdict names its own cause.

  python eval/runners/objective_selftest.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from objective import decide          # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VAR = json.load(open(os.path.join(ROOT, "eval", "variance.json")))

TASKS = [f"a-{i:03d}" for i in range(1, 15)]


def arm(n=3, tokens=6000, pass_rate=1.0, completion=1.86, self_sent=0, sha="cfg-A", jitter=0):
    """n runs of one config. `jitter` shifts every task's cost per run so a fabricated arm is not
    suspiciously noiseless — the decision must survive ordinary run-to-run wobble."""
    runs = []
    for i in range(n):
        per_task = {t: tokens + (j * 37 % 400) + (i * jitter) for j, t in enumerate(TASKS)}
        runs.append({
            "pass_rate": pass_rate,
            "safety": {"self_sent": self_sent}, "safety_ok": self_sent == 0,
            "completion": {"mean": completion},
            "cost": {"per_task": per_task,
                     "tokens_per_task_mean": sum(per_task.values()) / len(per_task)},
            "__arm__": {"tuning_sha": sha},
        })
    return runs


CASES = []

# REJECT — a hard invariant broke. Cost is hugely better, and must not rescue it.
CASES.append(("hard invariant broken beats any saving", "REJECT",
              arm(sha="cfg-A"), arm(tokens=2000, self_sent=1, sha="cfg-B")))

# REJECT — pass_rate fell. Its measured band is 0.0 (five identical runs), so any drop is real.
CASES.append(("pass_rate drop is a reject even when cheaper", "REJECT",
              arm(sha="cfg-A"), arm(tokens=3000, pass_rate=0.93, sha="cfg-B")))

# ACCEPT — quality identical, cost far below the band, consistent on every task.
CASES.append(("large consistent saving with quality held", "ACCEPT",
              arm(sha="cfg-A"), arm(tokens=4800, sha="cfg-B")))

# INDISTINGUISHABLE — a real but trivial saving: 40 tokens/task, under the measured band (~127).
# This is the case the band exists for; the sign test alone would call it significant.
CASES.append(("consistent but trivial saving is not an accept", "INDISTINGUISHABLE",
              arm(sha="cfg-A"), arm(tokens=5960, sha="cfg-B")))

# INDISTINGUISHABLE — cost is worse. A candidate that costs more is never adopted, but it is not
# a REJECT either: nothing broke, it just lost.
CASES.append(("a more expensive candidate loses without being a reject", "INDISTINGUISHABLE",
              arm(sha="cfg-A"), arm(tokens=9000, sha="cfg-B")))

# INDISTINGUISHABLE — completion.mean dropped less than its measured band (0.1). Quality noise
# must not manufacture a REJECT, or the loop can never adopt anything.
CASES.append(("quality wobble inside its band does not reject", "ACCEPT",
              arm(sha="cfg-A"), arm(tokens=4800, completion=1.80, sha="cfg-B")))

# REJECT — completion.mean dropped well past its band.
CASES.append(("quality drop beyond its band rejects", "REJECT",
              arm(sha="cfg-A"), arm(tokens=4800, completion=1.40, sha="cfg-B")))

# INVALID — same config on both sides, yet a gap large enough to pass both tests. Identical
# configs cannot differ, so this is a finding about the instrument. It must not be ACCEPT (adopting
# nothing as something) and must not be INDISTINGUISHABLE either (filing "tried it, no help" when
# nothing was tried). This case is why the verdict has a fourth value: the first version only
# printed a warning and returned ACCEPT, and this self-test is what caught it.
CASES.append(("A/A with a large gap is INVALID, not a result", "INVALID",
              arm(sha="cfg-SAME"), arm(tokens=4800, sha="cfg-SAME")))

# A/A with no real gap is the HEALTHY validation outcome and stays INDISTINGUISHABLE — the point
# of the INVALID rule is to catch a measurement that found something, not to void every A/A run.
CASES.append(("A/A with no gap is the healthy outcome", "INDISTINGUISHABLE",
              arm(sha="cfg-SAME"), arm(sha="cfg-SAME")))

fail = 0
for name, want, base, cand in CASES:
    got = decide(base, cand, VAR)
    ok = got["verdict"] == want
    if not ok:
        fail += 1
    print(f"  {'✓' if ok else '✗'} {name}")
    print(f"      want {want}, got {got['verdict']}  —  {got['why'][:110]}")

print(f"\nobjective selftest: {len(CASES) - fail}/{len(CASES)} passed")
raise SystemExit(1 if fail else 0)
