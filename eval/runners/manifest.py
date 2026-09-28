"""主集的换尺子检测器。

指标是集上的平均值,所以集一变两次的数字就不可比。本会话撞了两次,两次都只靠人推理才发现:
agent 集 14→21 让成本基线 6049→6449 并作废了已测的两个臂和整条噪声带;memory_select 10→13
让 recall 0.792→0.697 —— 看起来退化 0.095,实际什么都没退化。没有任何机制会拦。

这里就是那个机制。规则只有一条:**frozen 集的内容 sha 变了而声明的 version 没 bump → 失败**。

为什么用"声明的版本"而不是直接比 sha:sha 只能告诉你"变了",而变了之后该做什么是一整套动作
(重测基线、重测噪声带、作废旧臂)。要求手动 bump version,是把"我知道我在换尺子、并且会去
重测"这件事变成一个不能顺手跳过的显式动作。

  python eval/runners/manifest.py            # 校验;不一致则非零退出
  python eval/runners/manifest.py --bump agent.jsonl   # 改完集之后显式换尺子
"""
import argparse
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DS = os.path.join(ROOT, "eval", "datasets")
MANIFEST = os.path.join(DS, "MANIFEST.json")


def sha16(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def lines(path):
    with open(path, encoding="utf-8") as f:
        return sum(1 for _ in f)


def load():
    return json.load(open(MANIFEST, encoding="utf-8"))


def save(man):
    json.dump(man, open(MANIFEST, "w", encoding="utf-8"), indent=2, ensure_ascii=False)


def check():
    man = load()
    sets = man["sets"]
    drifted, missing, untracked = [], [], []
    for name, rec in sets.items():
        p = os.path.join(DS, name)
        if not os.path.exists(p):
            missing.append(name)
            continue
        if not rec.get("frozen", True):
            continue
        cur, n = sha16(p), lines(p)
        if cur != rec["sha256_16"]:
            drifted.append((name, rec["version"], rec["lines"], n))
    for f in sorted(os.listdir(DS)):
        if f.endswith(".jsonl") and f not in sets:
            untracked.append(f)

    print("=" * 78)
    print("主集清单校验")
    print("=" * 78)
    if missing:
        print("\n清单里有、文件不在:")
        for n in missing:
            print(f"  ✗ {n}")
    if untracked:
        print("\n文件在、清单里没有(新集要先登记):")
        for n in untracked:
            print(f"  ✗ {n}")
    if drifted:
        print("\n尺子变了但 version 没 bump —— 这一版的指标与基线【不可比】:")
        for name, ver, was, now in drifted:
            print(f"  ✗ {name}  v{ver}  {was} 条 -> {now} 条")
        print("\n改集是一次显式换尺子。确认要换,就执行:")
        for name, *_ in drifted:
            print(f"    python eval/runners/manifest.py --bump {name}")
        print("  然后必须重跑并重新接受基线(python eval/runners/gate.py --accept),")
        print("  并重测受影响的噪声带(eval/variance.json) —— 旧的臂作废,不要拿来对比。")
    if not (missing or untracked or drifted):
        print("\n全部一致:")
        for name, rec in sets.items():
            tag = "frozen" if rec.get("frozen", True) else "not frozen"
            print(f"  ✓ v{rec['version']}  {name:30s} {rec['lines']:>4} 条  {tag}")
    print("=" * 78)
    ok = not (missing or untracked or drifted)
    print("MANIFEST: PASS" if ok else "MANIFEST: FAIL")
    return 0 if ok else 1


def bump(name):
    man = load()
    if name not in man["sets"]:
        p = os.path.join(DS, name)
        if not os.path.exists(p):
            sys.exit(f"没有这个集: {name}")
        man["sets"][name] = {"version": 1, "sha256_16": sha16(p), "lines": lines(p), "frozen": True}
        save(man)
        print(f"{name}: 新登记为 v1 ({man['sets'][name]['lines']} 条)")
        return 0
    rec = man["sets"][name]
    p = os.path.join(DS, name)
    old_v, old_n = rec["version"], rec["lines"]
    rec["version"] = old_v + 1
    rec["sha256_16"] = sha16(p)
    rec["lines"] = lines(p)
    save(man)
    print(f"{name}: v{old_v} -> v{rec['version']}  ({old_n} -> {rec['lines']} 条)")
    print("提醒:跨版本的指标不可比。现在要做的是")
    print("  1) 重跑受影响的评测并 python eval/runners/gate.py --accept")
    print("  2) 重测噪声带(同配置连跑 N 次)并更新 eval/variance.json")
    print("  3) 作废旧的臂(eval/history/arms/*) —— 它们是在上一把尺子上测的")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bump", metavar="SET")
    a = ap.parse_args()
    return bump(a.bump) if a.bump else check()


if __name__ == "__main__":
    raise SystemExit(main())
