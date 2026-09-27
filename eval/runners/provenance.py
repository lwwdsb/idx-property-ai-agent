"""Run provenance stamp — what a comparison between two eval runs is allowed to assume.

Written ONCE per `make eval` (first step), read by make_report.py and by the tuning loop.
Without it, "the metric went from 0.77 to 0.79" is unfalsifiable: the datasets, the code, or
the parameters could have moved instead. The latency benchmark already caught this once —
it was measuring a stale process, not the current code.

Records, for one run:
  git       commit / branch / dirty         -> was the code the same?
  tuning    full config + sha256            -> were the PARAMETERS the same? (the loop's knobs)
  datasets  sha256 + line count per file    -> was the ruler the same? (a changed eval set
                                               makes two numbers incomparable — see the
                                               frozen-main-set / regression-pool split)
  seed      IDX_EVAL_SEED (default 7)       -> same folds/samples for the seeded runners
  env       capability flags, PRESENCE ONLY -> never values; a missing LLM key silently
                                               changes what the suites actually exercise
  versions  python / node

  python eval/runners/provenance.py            # write eval/history/run.stamp.json
  python eval/runners/provenance.py --show     # print the current stamp
"""
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HIST = os.path.join(ROOT, "eval", "history")
DATASETS = os.path.join(ROOT, "eval", "datasets")
TUNING = os.environ.get("IDX_TUNING", "").strip() or os.path.join(ROOT, "config", "tuning.json")
STAMP = os.path.join(HIST, "run.stamp.json")
DEFAULT_SEED = 7


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _dotenv():
    """The runners load .env (via dotenv / retrieval.common.load_env), so the stamp must look
    there too — reading os.environ alone reported a configured LLM key as MISSING and claimed
    the run used fallback paths it did not use."""
    path = os.path.join(ROOT, ".env")
    out = {}
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
    return out


def _service_up(url, timeout=1.5):
    """Whether the warm retrieval service answered. NOT cosmetic: evalIntentParse uses the
    embedding classifier only when :8099 is up, and silently falls back to regex otherwise —
    two runs that differ on this are measuring different systems."""
    try:
        import urllib.request
        with urllib.request.urlopen(f"{url.rstrip('/')}/health", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                              timeout=10).stdout.strip() or None
    except Exception:
        return None


def build():
    dotenv = _dotenv()
    datasets = {}
    for name in sorted(os.listdir(DATASETS)) if os.path.isdir(DATASETS) else []:
        path = os.path.join(DATASETS, name)
        if not os.path.isfile(path):
            continue
        with open(path, "rb") as fh:
            n = sum(1 for line in fh if line.strip())
        datasets[name] = {"sha256": _sha256(path)[:16], "lines": n}

    tuning = None
    if os.path.exists(TUNING):
        tuning = {"path": os.path.relpath(TUNING, ROOT), "sha256": _sha256(TUNING)[:16],
                  "config": json.load(open(TUNING, encoding="utf-8"))}

    dirty = _git("status", "--porcelain")
    return {
        "at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "git": {"commit": (_git("rev-parse", "HEAD") or "")[:12] or None,
                "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
                # A dirty tree means this run is NOT reproducible from the commit alone.
                "dirty": bool(dirty), "dirty_files": len(dirty.splitlines()) if dirty else 0},
        "tuning": tuning,
        "datasets": datasets,
        "seed": int(os.environ.get("IDX_EVAL_SEED", DEFAULT_SEED)),
        # Presence only — never the values. Which of these are set changes what the suites
        # actually exercise (no LLM key -> regex fallback paths, etc).
        "env": {k: bool((os.environ.get(k) or dotenv.get(k) or "").strip()) for k in
                ("LLM_API_KEY", "EMAIL_PASSWORD", "ORCH_TOKEN", "GOOGLE_MAPS_API_KEY")},
        "services": {"retrieval": _service_up(
            os.environ.get("RETRIEVAL_URL") or dotenv.get("RETRIEVAL_URL") or "http://localhost:8099")},
        "versions": {"python": platform.python_version(), "node": _node_version()},
    }


def _node_version():
    try:
        return subprocess.run(["node", "--version"], capture_output=True, text=True,
                              timeout=10).stdout.strip() or None
    except Exception:
        return None


def load_stamp():
    """The stamp for the CURRENT run, or None if `make eval` did not write one."""
    return json.load(open(STAMP, encoding="utf-8")) if os.path.exists(STAMP) else None


def main():
    if "--show" in sys.argv:
        s = load_stamp()
        print(json.dumps(s, indent=2, ensure_ascii=False) if s else "no stamp yet (run make eval)")
        return
    os.makedirs(HIST, exist_ok=True)
    stamp = build()
    tmp = STAMP + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(stamp, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, STAMP)          # atomic, like the tuning writes
    g = stamp["git"]
    print(f"run stamp: {g['commit']}@{g['branch']}"
          f"{' +DIRTY(' + str(g['dirty_files']) + ' files)' if g['dirty'] else ''}"
          f"  seed={stamp['seed']}  datasets={len(stamp['datasets'])}"
          f"  tuning={stamp['tuning']['sha256'] if stamp['tuning'] else 'MISSING'}")


if __name__ == "__main__":
    main()
