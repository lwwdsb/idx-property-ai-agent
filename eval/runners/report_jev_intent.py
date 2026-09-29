"""给 Jev 的意图臂算分,并和现状、embedding-only 摆在一起。

口径上唯一要小心的地方:residue 配置下 Jev 只回答【规则没定案】的那 34 条,其余 84 条仍由规则
决定。所以必须把两者【组合成完整的 118 条】再算分 —— 只算那 34 条和任何东西都不可比,而生产
形状本来就是"规则 + Jev 兜底"。

指标实现复用 eval/metrics,和意图报告是同一份 —— 换臂不换尺子。

  python eval/runners/report_jev_intent.py all
  python eval/runners/report_jev_intent.py residue
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "eval"))
from metrics.classification import classification_report   # noqa: E402

HIST = os.path.join(ROOT, "eval", "history")


def read(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def score(preds, name):
    """preds: [{gold: [...], pred: str}]。与 report_intent_parse 同口径。"""
    in_set = sum(1 for p in preds if p["pred"] in p["gold"]) / len(preds)
    rep = classification_report([(p["gold"][0], p["pred"]) for p in preds])
    ood = [p for p in preds if "unknown" in p["gold"]]
    ind = [p for p in preds if "unknown" not in p["gold"]]
    return {
        "arm": name, "n": len(preds),
        "accuracy_in_set": round(in_set, 4),
        "macro_f1": round(rep["macro_f1"], 4),
        "ood_reject": round(sum(1 for p in ood if p["pred"] == "unknown") / len(ood), 4) if ood else None,
        # 域内被误判成 unknown —— embedding 就是栽在这个数上(0.458)
        "in_domain_rejected": round(sum(1 for p in ind if p["pred"] == "unknown") / len(ind), 4) if ind else None,
        "per_class": rep["per_class"],
        "misses": [{"input": p["input"], "gold": p["gold"], "pred": p["pred"]}
                   for p in preds if p["pred"] not in p["gold"]],
    }


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "all"
    jev_path = os.path.join(HIST, f"jev_intent.{tag}.preds.jsonl")
    if not os.path.exists(jev_path):
        sys.exit(f"没有 {os.path.relpath(jev_path, ROOT)} —— 先跑 evalJevIntent.ts --config {tag}")
    jev = {r["id"]: r for r in read(jev_path)}
    pipe = {r["id"]: r for r in read(os.path.join(HIST, "intent.preds.jsonl"))}

    # 组合:Jev 答过的用 Jev 的,没答过的保留规则的verdict。tag=all 时 Jev 覆盖全部。
    composed = []
    for cid, p in pipe.items():
        j = jev.get(cid)
        pred = j["pred"] if (j and j.get("pred") not in (None, "error")) else p["pred"]
        composed.append({"id": cid, "input": p["input"], "gold": p["gold"], "pred": pred})

    cur = score([{"id": k, "input": v["input"], "gold": v["gold"], "pred": v["pred"]}
                 for k, v in pipe.items()], "现状(规则+embedding)")
    new = score(composed, f"规则+Jev({tag})" if tag != "all" else "Jev 单独(全量)")

    # embedding-only 那条臂的数字直接从意图报告里取,保证是同一次测量
    emb = None
    mp = os.path.join(HIST, "intent_parse.metrics.json")
    if os.path.exists(mp):
        e = (json.load(open(mp, encoding="utf-8")).get("intent") or {}).get("embedding_only")
        if e:
            g = e["at_production_gate"]
            emb = {"arm": "embedding 单独(放开正则)", "n": e["n"], "accuracy_in_set": g["accuracy"],
                   "macro_f1": g["macro_f1"], "ood_reject": g["ood_reject"],
                   "in_domain_rejected": g["in_domain_rejected"]}

    print("=" * 96)
    print(f"意图识别三臂对比 — 118 条(域内 83 + 域外 35) · Jev 覆盖 {len(jev)} 条")
    print("=" * 96)
    print(f"  {'臂':28} {'n':>4} {'accuracy':>9} {'macro-F1':>9} {'域外拒识':>9} {'域内误拒':>9}")
    for r in [cur, emb, new]:
        if not r:
            continue
        f = lambda v: "n/a" if v is None else f"{v:.4f}"
        print(f"  {r['arm']:28} {r['n']:>4} {f(r['accuracy_in_set']):>9} {f(r['macro_f1']):>9}"
              f" {f(r['ood_reject']):>9} {f(r['in_domain_rejected']):>9}")

    print(f"\n采纳门槛(事先定死,避免事后找理由):域外拒识 ≥ 0.90 且 macro-F1 不低于 {cur['macro_f1']}")
    ok = (new["ood_reject"] or 0) >= 0.90 and new["macro_f1"] >= cur["macro_f1"]
    print(f"  → {'达标' if ok else '未达标'}:域外拒识 {new['ood_reject']} · macro-F1 {new['macro_f1']}")
    if new["misses"]:
        print(f"\n剩余错误 {len(new['misses'])} 条:")
        for m in new["misses"][:12]:
            print(f"  gold={str(m['gold']):22} pred={m['pred']:10} {m['input'][:46]}")

    out = os.path.join(HIST, f"jev_intent.{tag}.metrics.json")
    json.dump({"current": cur, "embedding_only": emb, "jev": new, "threshold_met": ok},
              open(out, "w"), indent=1, ensure_ascii=False)
    print(f"\n-> {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
