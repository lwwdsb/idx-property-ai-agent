"""实测噪声带 —— 从一个臂的 N 次重复算出离散度与最小可检测差,写回 eval/variance.json。

为什么要有这个文件:这张表【以前是手算的】。而它恰恰是门禁容差的来源 —— 一个手算、
无从复现、换尺子之后没人记得重算的数,去决定"这个差异算不算回归"。本会话就撞上了:
agent 集 21→26 之后旧带立刻失效,而没有任何东西会提醒。

  mde_abs = 2 × SE_diff = 2 × stdev × sqrt(2/n)
两组各 n 次比较时,小于它的差异不能当成效应。注意它随 n 缩小,所以 n 必须记进表里 ——
拿 n=3 的带去判 n=6 的比较是偷换前提。

取整按【有效数字】而不是小数位:本表指标跨 0.03 到 137000 五个数量级,按小数位取整会把
小端的带存成 0.0,容差规则就退化成"一点都不许掉",在正常噪声上误 REJECT。

  python eval/runners/variance.py --label v4-baseline --section auto_agent
  python eval/runners/variance.py --label v4-baseline --section auto_agent --write
"""
import argparse
import glob
import json
import math
import os
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARMS = os.path.join(ROOT, "eval", "history", "arms")
VAR = os.path.join(ROOT, "eval", "variance.json")

# 组 -> 指标名 -> 在 agent.metrics.json 里的路径
METRICS = {
    "quality": {"pass_rate": ("pass_rate",), "completion_mean": ("completion", "mean")},
    "cost": {k: ("cost", k) for k in ("tokens_total", "tokens_per_task_mean",
                                      "tokens_per_task_p50", "tokens_per_task_max",
                                      "llm_calls_total", "steps_total")},
    "latency": {"p50_ms": ("runtime_metrics", "latency_p50_ms"),
                "p99_ms": ("runtime_metrics", "latency_p99_ms")},
}


def dig(d, path):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def sigfig(x, n=4):
    """有效数字取整。0 原样返回 —— 它不是精度问题,是真的没有离散。"""
    if x == 0 or x is None:
        return 0.0 if x == 0 else None
    return round(x, -int(math.floor(math.log10(abs(x)))) + (n - 1))


def band(values, n):
    mean = st.mean(values)
    sd = st.stdev(values) if len(values) > 1 else 0.0
    mde = 2 * sd * math.sqrt(2 / n)
    return {
        "min": min(values), "max": max(values),
        "mean": sigfig(mean, 6), "stdev": sigfig(sd, 4),
        "cv_pct": round(100 * sd / mean, 2) if mean else 0.0,
        "mde_abs": sigfig(mde, 4),
        "mde_pct": round(100 * mde / mean, 2) if mean else 0.0,
        "values": values,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--section", default="auto_agent")
    ap.add_argument("--write", action="store_true", help="写回 eval/variance.json(默认只打印)")
    a = ap.parse_args()

    runs = sorted(glob.glob(os.path.join(ARMS, a.label, "run*.json")))
    if not runs:
        return print(f"没有找到臂 '{a.label}' 的任何运行") or 1
    docs = [json.load(open(r, encoding="utf-8")) for r in runs]

    # 环境故障的运行不是对配置的测量,必须在算带之前剔掉 —— 但【必须点名】,不能静默。
    # 本会话撞到一次:第 6 次重复里 26 条任务全部 stop=error、LLM 调用数 0、usage 未上报,
    # 服务都正常,是模型那端整体拒了。把它算进去会让 tokens 的 stdev 爆掉,于是门禁容差
    # 被撑得极宽,从此什么回归都拦不住 —— 一次环境抖动永久腐蚀掉这把尺子。
    # 两种无效:整轮没跑起来(usage 未上报 / 0 次调用),以及【有任何一条任务 stop=error】。
    # 后者同样致命而且更隐蔽:那条任务的成本没被计入,整轮的合计就偏低,看起来却完全正常。
    # 本会话实测:余额耗尽时先是某一轮最后一条任务 error(合计少算一条),下一轮才整个死掉。
    def invalid(d):
        c = d.get("cost") or {}
        if not c.get("usage_reported") or c.get("llm_calls_total", 0) == 0:
            return "模型没跑起来"
        n_err = sum(1 for f in d.get("failures", []) if f.get("stop") == "error")
        return f"{n_err} 条任务 stop=error" if n_err else None
    bad = [(os.path.basename(r), d, invalid(d)) for r, d in zip(runs, docs) if invalid(d)]
    if bad:
        print(f"剔除 {len(bad)} 次无效运行(模型没真正跑起来,不是配置的表现):")
        for name, d, why in bad:
            print(f"  {name}: {why}  (pass_rate={d.get('pass_rate')}, "
                  f"llm_calls={(d.get('cost') or {}).get('llm_calls_total')})")
        badnames = {b[0] for b in bad}
        keep = [(r, d) for r, d in zip(runs, docs) if os.path.basename(r) not in badnames]
        runs, docs = [r for r, _ in keep], [d for _, d in keep]
        if len(docs) < 3:
            return print(f"拒绝:只剩 {len(docs)} 次有效运行,算不出可信的带(至少 3 次)") or 1
    n = len(docs)

    shas = {d.get("__arm__", {}).get("tuning_sha") for d in docs}
    if len(shas) != 1:
        return print(f"拒绝:这个臂里的运行用了不同的配置 {shas} —— 那不是一个臂") or 1
    sizes = {d.get("n") for d in docs}
    if len(sizes) != 1:
        return print(f"拒绝:集大小在运行之间变了 {sizes} —— 中途换过尺子") or 1

    out = {"__set__": f"agent.jsonl {sizes.pop()} 条 / {docs[0]['cost']['drives_total']} 段 drive,llm=live"}
    for group, metrics in METRICS.items():
        g = {}
        for name, path in metrics.items():
            vals = [dig(d, path) for d in docs]
            if any(v is None for v in vals):
                print(f"  跳过 {group}.{name}:有运行缺这个字段")
                continue
            g[name] = band(vals, n)
        out[group] = g

    print(f"臂 '{a.label}': {n} 次运行,配置 {shas.pop() if shas else '?'}")
    for group, g in out.items():
        if group.startswith("__"):
            continue
        print(f"  [{group}]")
        for name, b in g.items():
            print(f"    {name:22} mean {b['mean']:<12} stdev {b['stdev']:<10} "
                  f"MDE ±{b['mde_abs']} ({b['mde_pct']}%)")

    if not a.write:
        return print("\n(只打印。加 --write 才写回 eval/variance.json)") or 0

    var = json.load(open(VAR, encoding="utf-8"))
    var[a.section] = out
    var["measured_at"] = __import__("datetime").date.today().isoformat()
    var["n_runs"] = n
    var["tuning_sha"] = docs[0].get("__arm__", {}).get("tuning_sha")
    json.dump(var, open(VAR, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n-> {os.path.relpath(VAR, ROOT)}  (section {a.section}, n={n})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
