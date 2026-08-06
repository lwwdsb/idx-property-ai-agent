"""Latency benchmark: per-endpoint p50/p95/p99 + a concurrency (tail-latency) test.

Measures the warm-service endpoints (retrieval :8099) and the end-to-end orchestrate
path (:8100). Reports percentiles, not just averages (tail latency is what users feel).

Notes:
- orchestrate needs the shared token; each request uses a UNIQUE userId so the per-user
  rate limit (20/min) and the idempotency cache don't skew timings.
- Run services first (make up) + Qdrant. Run: python eval/runners/bench_latency.py
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import httpx

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HIST = os.path.join(ROOT, "eval", "history")
RET = "http://localhost:8099"
ORCH = "http://localhost:8100"
N = 40           # sequential samples per endpoint
CONC = 20        # concurrent requests for the tail-latency test


def token():
    for line in open(os.path.join(ROOT, ".env")):
        if line.startswith("ORCH_TOKEN="):
            return line.split("=", 1)[1].strip()
    return ""


TOK = token()


def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return 0.0
    i = min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))
    return round(xs[i] * 1000, 1)   # ms


def summarize(name, times):
    return {"endpoint": name, "n": len(times),
            "p50": pct(times, 50), "p95": pct(times, 95), "p99": pct(times, 99),
            "max": round(max(times) * 1000, 1) if times else 0.0}


def time_call(fn):
    t = time.perf_counter()
    fn()
    return time.perf_counter() - t


def bench_seq(name, fn, n=N):
    fn()  # warm
    times = [time_call(fn) for _ in range(n)]
    s = summarize(name, times)
    print(f"  {name:26} p50={s['p50']:>7}ms  p95={s['p95']:>7}ms  p99={s['p99']:>7}ms")
    return s


def main():
    client = httpx.Client(timeout=60.0)

    def search():
        client.post(f"{RET}/search", json={"text": "ocean view craftsman", "city": "Irvine", "k": 5})

    def rag():
        client.post(f"{RET}/rag", json={"question": "what is days on market"})

    def classify():
        client.post(f"{RET}/classify", json={"message": "how's the market in Irvine"})

    _uid = [0]

    def orchestrate_search():
        _uid[0] += 1
        client.post(f"{ORCH}/orchestrate",
                    headers={"Authorization": f"Bearer {TOK}"},
                    json={"userId": f"bench-{_uid[0]}", "message": "在 Irvine 找 3 居室 200万以下"})

    print(f"latency benchmark (sequential, n={N}):\n")
    results = [
        bench_seq("retrieval /search", search),
        bench_seq("retrieval /rag", rag),
        bench_seq("retrieval /classify", classify),
        bench_seq("orchestrate /orchestrate", orchestrate_search),
    ]

    # concurrency / tail-latency: fire CONC orchestrate requests at once
    print(f"\nconcurrency test: {CONC} simultaneous /orchestrate requests")
    base = _uid[0]

    def one(i):
        t = time.perf_counter()
        client.post(f"{ORCH}/orchestrate",
                    headers={"Authorization": f"Bearer {TOK}"},
                    json={"userId": f"conc-{base}-{i}", "message": "Irvine 行情怎么样"})
        return time.perf_counter() - t

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=CONC) as ex:
        conc_times = list(ex.map(one, range(CONC)))
    wall = time.perf_counter() - t0
    conc = summarize("orchestrate @concurrency", conc_times)
    conc["wall_s"] = round(wall, 2)
    print(f"  {CONC} concurrent: wall={conc['wall_s']}s  p50={conc['p50']}ms  "
          f"p95={conc['p95']}ms  p99={conc['p99']}ms  max={conc['max']}ms")

    client.close()
    out = {"sequential": results, "concurrency": conc, "conc_level": CONC}
    os.makedirs(HIST, exist_ok=True)
    with open(os.path.join(HIST, "latency.metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwritten to {os.path.join(HIST, 'latency.metrics.json')}")


if __name__ == "__main__":
    main()
