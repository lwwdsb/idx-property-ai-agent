"""Run the agent eval N times under the CURRENT config and keep every run.

Why this exists: a single run cannot support a claim about cost. The measured noise band
(eval/variance.json) puts the minimum detectable effect at ~2% of tokens per task even with 5
repeats, so an "improvement" read off one run is not evidence. An arm is therefore N runs, stored
together under a label, and `objective.py` compares two labelled arms.

  python eval/runners/repeat_agent.py --label baseline --n 5
  python eval/runners/repeat_agent.py --label progressive-on --n 5

Runs land in eval/history/arms/<label>/run<i>.json (gitignored like the rest of history). The
label also records the tuning config sha, so comparing two arms that secretly shared a config —
the failure this whole apparatus exists to prevent — is detectable after the fact.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARMS = os.path.join(ROOT, "eval", "history", "arms")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--n", type=int, default=5)
    a = ap.parse_args()

    out = os.path.join(ARMS, a.label)
    os.makedirs(out, exist_ok=True)
    cfg = os.path.join(ROOT, "config", "tuning.json")
    sha = hashlib.sha256(open(cfg, "rb").read()).hexdigest()[:12]
    src = os.path.join(ROOT, "eval", "history", "agent.metrics.json")

    for i in range(1, a.n + 1):
        r = subprocess.run(["npx", "tsx", "eval/runners/evalAgent.ts"], cwd=ROOT,
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(f"run {i} FAILED:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
            return 1
        r2 = subprocess.run([sys.executable, "eval/runners/report_agent.py"], cwd=ROOT,
                            capture_output=True, text=True)
        if r2.returncode != 0:
            print(f"report {i} FAILED:\n{r2.stdout[-2000:]}\n{r2.stderr[-2000:]}")
            return 1
        d = json.load(open(src))
        d["__arm__"] = {"label": a.label, "run": i, "tuning_sha": sha,
                        "tuning": json.load(open(cfg))}
        json.dump(d, open(os.path.join(out, f"run{i}.json"), "w"), indent=1)
        tok = (d.get("cost") or {}).get("tokens_total")
        print(f"  run {i}/{a.n}: pass_rate={d.get('pass_rate')} tokens={tok}")

    print(f"\narm '{a.label}': {a.n} runs -> {os.path.relpath(out, ROOT)}  (tuning {sha})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
