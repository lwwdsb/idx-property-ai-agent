"""精排(cross-encoder)在【有效的】分级集上值多少 —— 经由服务测量,而不是在评测里复现。

为什么必须走服务:精排器活在 FastAPI 服务里(warm-load 的 TextCrossEncoder + 粗排去重 + 双层
降级)。在评测里重写一遍就是又一次"测一条生产不走的路",本项目已经踩过五次。

为什么现在才第一次能公平测:
  - 旧的分级池是用【旧索引】的三路各 top-12 建的。精排从 hybrid 粗排 top-20 里挑,能把第 11~20
    名提进 top-10 —— 那些文档从没被判过,按不相关计分。于是"精排 +0.143"里有多少是真提升、
    多少是它恰好选中了池内文档,分不开。
  - 而同一个精排在客观 known-item 上是【负的】(@5 +0.08 但 @10 −0.06),当时的怀疑是
    "CE 与裁判同读 remark 文本、可能有共识水分"。
  - 现在池子并到 RERANK_COARSE 深度,精排能够到的每一条都已判分,覆盖率可验证。

  .venv/bin/python eval/runners/eval_rerank_service.py
  .venv/bin/python eval/runners/eval_rerank_service.py eval/datasets/retrieval_large.jsonl
"""
import json
import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "eval", "metrics"))
from ir import ndcg_at_k, recall_at_k, mrr, precision_at_k, mean   # noqa: E402

SERVICE = os.environ.get("RETRIEVAL_URL", "http://localhost:8099")
K = 10


def search(text, rerank):
    body = json.dumps({"text": text, "k": K, "rerank": rerank}).encode()
    req = urllib.request.Request(f"{SERVICE}/search", data=body,
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return [str(h["listing_id"]) for h in json.load(r)["results"]]


def arm(cases, rerank):
    ndcg, rec, rr, prec, cover, per_q = [], [], [], [], [], {}
    for c in cases:
        rel = c["label"]["relevant"]
        if not rel:
            continue
        ranked = search(c["input"], rerank)
        n = ndcg_at_k(ranked, rel, K)
        ndcg.append(n); per_q[c["id"]] = n
        rec.append(recall_at_k(ranked, rel, K))
        rr.append(mrr(ranked, rel))
        prec.append(precision_at_k(ranked, rel, 5))
        graded = c["label"].get("graded")
        if graded:
            judged = set(str(i) for i in graded)
            cover.append(sum(1 for i in ranked[:K] if i in judged) / len(ranked[:K]) if ranked else 1.0)
    return {"nDCG@10": round(mean(ndcg), 4), "recall@10": round(mean(rec), 4),
            "MRR": round(mean(rr), 4), "precision@5": round(mean(prec), 4),
            "judged_coverage@10": round(mean(cover), 4) if cover else None,
            "n": len(ndcg)}, per_q


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "eval/datasets/retrieval.jsonl"
    path = path if os.path.isabs(path) else os.path.join(ROOT, path)
    cases = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    try:
        with urllib.request.urlopen(f"{SERVICE}/health", timeout=10) as r:
            health = json.load(r)
    except Exception as e:
        sys.exit(f"服务不可达 {SERVICE}: {e}\n先 make up(或起 uvicorn service:app --port 8099)")

    print(f"\n精排测量 — {os.path.basename(path)} {len(cases)} 条 · 服务 {SERVICE} · "
          f"collection={health.get('collection')}\n")
    off, pq_off = arm(cases, False)
    on, pq_on = arm(cases, True)

    print(f"  {'arm':22} {'nDCG@10':>9} {'recall@10':>10} {'MRR':>8} {'P@5':>8} {'judged@10':>10}")
    for name, m in (("hybrid 原序(rerank=off)", off), ("+ CE 精排(生产)", on)):
        cov = m["judged_coverage@10"]
        print(f"  {name:22} {m['nDCG@10']:>9} {m['recall@10']:>10} {m['MRR']:>8} {m['precision@5']:>8}"
              f" {(f'{cov:.3f}' if cov is not None else 'n/a (v1)'):>10}")

    d = round(on["nDCG@10"] - off["nDCG@10"], 4)
    ids = [i for i in pq_off if i in pq_on]
    win = sum(1 for i in ids if pq_on[i] > pq_off[i])
    loss = sum(1 for i in ids if pq_on[i] < pq_off[i])
    tie = len(ids) - win - loss
    diffs = [pq_on[i] - pq_off[i] for i in ids]
    md = mean(diffs)
    if len(diffs) > 1:
        var = sum((x - md) ** 2 for x in diffs) / (len(diffs) - 1)
        se = (var / len(diffs)) ** 0.5
        t = md / se if se else 0.0
    else:
        t = 0.0
    print(f"\n  配对 per-query nDCG@10:{win} 胜 / {tie} 平 / {loss} 负 · meanΔ {md:+.4f} · t≈{t:.2f}"
          + ("  显著" if abs(t) >= 2 else "  落在噪声内"))
    print(f"  精排带来的 nDCG@10 变化:{d:+.4f}")

    cov_off, cov_on = off["judged_coverage@10"], on["judged_coverage@10"]
    if cov_off is not None and cov_on is not None:
        # 覆盖率若两臂不同,差值里就混着"池外惩罚"而不只是质量 —— 这正是旧测量分不开的那部分。
        print(f"\n  判分覆盖率 off {cov_off:.3f} / on {cov_on:.3f}"
              + ("  → 两臂同等被判分,这个差值是质量差,不含池外惩罚"
                 if abs(cov_off - cov_on) < 0.02 else
                 "  ⚠️ 两臂覆盖率不同,差值里混着池外惩罚,不能只当质量看"))

    out = os.path.join(ROOT, "eval", "history", "rerank_service.metrics.json")
    json.dump({"dataset": os.path.basename(path), "collection": health.get("collection"),
               "off": off, "on": on, "delta_ndcg": d,
               "paired": {"win": win, "tie": tie, "loss": loss, "mean_diff": round(md, 4),
                          "t_like": round(t, 2), "significant": abs(t) >= 2}},
              open(out, "w"), indent=1, ensure_ascii=False)
    print(f"\n  -> {os.path.relpath(out, ROOT)}\n")


if __name__ == "__main__":
    main()
