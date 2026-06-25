#!/usr/bin/env python3
"""Week 0 stage 5 — config/env validation.

Fails loudly (non-zero exit) if required config is missing or a service is
unreachable, so the app never starts half-configured.

Checks:
  1. required vars present in .env
  2. MySQL reachable + all three tables present with row counts
  3. OpenAI key valid (live call to /v1/models) — skipped with a warning if key blank

Usage:  python3 scripts/check_env.py
"""
import os
import sys
import subprocess
import urllib.request
import urllib.error

# Hard-required now (Week 0). OPENAI_API_KEY is checked separately: it is only
# needed from Week 6 (embeddings), so a blank key is a WARNING, not a failure —
# but if a key IS set it must be valid.
REQUIRED = ["DB_HOST", "DB_PORT", "DB_USER", "DB_PASSWORD", "DB_NAME"]
TABLES = ["rets_property", "california_sold", "rets_openhouse"]
ENV_PATH = os.path.join(os.path.dirname(__file__), "..", ".env")


def load_env(path):
    env = {}
    if not os.path.exists(path):
        sys.exit(f"FATAL: .env not found at {path}")
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()
    return env


def check_required(env):
    missing = [k for k in REQUIRED if not env.get(k)]
    if missing:
        print(f"  MISSING required vars: {', '.join(missing)}")
    else:
        print("  all required vars present")
    return missing


def check_db(env):
    cmd = ["mysql", "-h", env.get("DB_HOST", "127.0.0.1"), "-P", env.get("DB_PORT", "3306"),
           "-u", env.get("DB_USER", "root"), "-N", env.get("DB_NAME", "idx_exchange"),
           "-e", "SELECT " + " UNION ALL SELECT ".join(
               f"'{t}',COUNT(*) FROM {t}" for t in TABLES)]
    proc_env = {**os.environ, "MYSQL_PWD": env.get("DB_PASSWORD", "")}
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, env=proc_env, timeout=30)
    except FileNotFoundError:
        return ["mysql CLI not found"]
    if out.returncode != 0:
        return [out.stderr.strip() or "mysql connection failed"]
    for row in out.stdout.strip().splitlines():
        print("   ", row.replace("\t", ": "))
    return []


def check_openai(env):
    """Returns (errors, warnings). Blank key -> warning (needed from Week 6).
    Set-but-invalid key -> error."""
    key = env.get("OPENAI_API_KEY", "")
    if not key:
        print("  SKIP: OPENAI_API_KEY blank — not needed until Week 6 (embeddings)")
        return [], ["OPENAI_API_KEY not set (required before Week 6)"]
    req = urllib.request.Request("https://api.openai.com/v1/models",
                                 headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            if r.status == 200:
                print("  OpenAI key valid (200 from /v1/models)")
                return [], []
    except urllib.error.HTTPError as e:
        return [f"OpenAI key is set but rejected: HTTP {e.code} (check key/billing)"], []
    except Exception as e:
        return [f"OpenAI check failed: {e}"], []
    return ["OpenAI check: unexpected response"], []


def main():
    env = load_env(ENV_PATH)
    errors, warnings = [], []
    print("[1/3] required vars"); errors += check_required(env)
    print("[2/3] MySQL"); errors += check_db(env)
    print("[3/3] OpenAI"); oe, ow = check_openai(env); errors += oe; warnings += ow
    print()
    if warnings:
        print("WARNINGS:")
        for w in warnings:
            print("  -", w)
        print()
    if errors:
        print("ENV CHECK FAILED:")
        for e in errors:
            print("  -", e)
        sys.exit(1)
    print("ENV CHECK PASSED ✓" + ("  (with warnings)" if warnings else ""))


if __name__ == "__main__":
    main()
