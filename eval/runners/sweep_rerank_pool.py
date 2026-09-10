"""Sweep the cross-encoder's candidate-pool size: CE reranks hybrid top-N down to top-10.
Production currently uses N=30 (RERANK_COARSE). Run: python eval/runners/sweep_rerank_pool.py

扫候选池大小:CE 从 hybrid top-N 里重排出 top-10,N 取多少最好。
生产当前 N=30。注意 N>10 时,11..N 名可能未被标注,被提上来会按 0 分算 —— 对大 N 不利,
所以这个扫描给的是"保守下界",但趋势本身有效。"""
import os, sys, json
ROOT = os.getcwd()
sys.path.insert(0, os.path.join(ROOT, "retrieval"))
sys.path.insert(0, os.path.join(ROOT, "eval", "metrics"))
from search import hybrid_search
from common import load_env
from ir import ndcg_at_k, precision_at_k, mrr, mean
from fastembed.rerank.cross_encoder import TextCrossEncoder
import pymysql

DATA = os.path.join(ROOT, "eval", "datasets", "retrieval.jsonl")
cases = [json.loads(l) for l in open(DATA) if l.strip() and json.loads(l)["label"]["relevant"]]
ce = TextCrossEncoder("Xenova/ms-marco-MiniLM-L-6-v2")
env = load_env()
conn = pymysql.connect(host=env.get("DB_HOST"), port=int(env.get("DB_PORT", 3306)),
                       user=env.get("DB_USER"), password=env.get("DB_PASSWORD"),
                       database=env.get("DB_NAME"), cursorclass=pymysql.cursors.DictCursor)

def remarks(ids):
    if not ids: return {}
    fmt = ",".join(["%s"] * len(ids))
    with conn.cursor() as cur:
        cur.execute(f"SELECT id,L_City,L_Type_,L_Remarks FROM rets_property WHERE id IN ({fmt})", ids)
        return {int(r["id"]): f"{r.get('L_Type_') or ''} in {r.get('L_City') or ''}. {r.get('L_Remarks') or ''}" for r in cur.fetchall()}

NS = [10, 15, 20, 30, 50]
agg = {n: {"ndcg": [], "p5": [], "mrr": [], "d": []} for n in NS}
base = {"ndcg": [], "p5": [], "mrr": []}
for c in cases:
    q, rel = c["input"], c["label"]["relevant"]
    pool = [p.id for p in hybrid_search(q, None, k=max(NS), mode="hybrid")]
    b = pool[:10]
    base["ndcg"].append(ndcg_at_k(b, rel, 10)); base["p5"].append(precision_at_k(b, rel, 5)); base["mrr"].append(mrr(b, rel))
    bn = ndcg_at_k(b, rel, 10)
    for n in NS:
        cand = pool[:n]
        txt = remarks(cand)
        docs = [txt.get(i, "") for i in cand]
        scores = list(ce.rerank(q, docs))
        r = [i for i, _ in sorted(zip(cand, scores), key=lambda x: -x[1])][:10]
        agg[n]["ndcg"].append(ndcg_at_k(r, rel, 10)); agg[n]["p5"].append(precision_at_k(r, rel, 5)); agg[n]["mrr"].append(mrr(r, rel))
        agg[n]["d"].append(ndcg_at_k(r, rel, 10) - bn)
conn.close()

print(f"24 条 judge 分级集 · CE 从 hybrid top-N 重排出 top-10\n")
print(f"  {'配置':<14}{'nDCG@10':>9}{'P@5':>8}{'MRR':>8}{'ΔnDCG':>9}{'胜/平/负':>10}{'t':>7}")
print(f"  {'不重排(基线)':<12}{mean(base['ndcg']):>9.4f}{mean(base['p5']):>8.4f}{mean(base['mrr']):>8.4f}{'—':>9}{'—':>10}{'—':>7}")
for n in NS:
    a = agg[n]; d = a["d"]
    w = sum(1 for x in d if x > 1e-9); l = sum(1 for x in d if x < -1e-9); m = len(d)
    md = mean(d); sd = (sum((x-md)**2 for x in d)/(m-1))**.5 if m > 1 else 0
    t = md/(sd/m**.5) if sd else 0
    print(f"  {'CE top-'+str(n):<14}{mean(a['ndcg']):>9.4f}{mean(a['p5']):>8.4f}{mean(a['mrr']):>8.4f}{md:>+9.4f}{f'{w}/{m-w-l}/{l}':>10}{t:>7.2f}")
