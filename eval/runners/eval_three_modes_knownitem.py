"""dense / bm25 / hybrid compared on KNOWN-ITEM recall (objective labels, no judge).

Cross-checks the graded set, where bm25 has the highest mean — is bm25 genuinely better,
or does the LLM judge reward lexical overlap (it grades by reading the same remarks text)?
Needs Qdrant + the mode_retrieval predictions.  Run: python eval/runners/eval_three_modes_knownitem.py

三路检索(dense/bm25/hybrid)在 KNOWN-ITEM 上的对比 —— 客观标签,无 judge 偏差。
用于交叉验证:judge 分级集上 bm25 均值最高,是 bm25 真的更强,还是 judge 偏爱词面重合?"""
import json, os, sys
ROOT = os.getcwd(); sys.path.insert(0, os.path.join(ROOT, "retrieval"))
from search import hybrid_search, build_filter
TYPE_DB = {"house":"Single Family Residence","condo":"Condominium","townhouse":"Townhouse"}
gold={json.loads(l)["id"]:json.loads(l) for l in open("eval/datasets/mode_retrieval.jsonl") if l.strip()}
preds={json.loads(l)["id"]:json.loads(l) for l in open("eval/history/mode_retrieval.preds.jsonl") if l.strip()}
MODES=["dense","bm25","hybrid"]
agg={m:{5:[0,0],10:[0,0],20:[0,0]} for m in MODES}
per={m:[] for m in MODES}
for cid,g in gold.items():
    tgt=g["gold"].get("known_item")
    if not tgt or cid not in preds: continue
    p=preds[cid]; f=p["auto_filter"]
    text=(p.get("auto_semantic") or "").strip() or g["input"]
    flt=build_filter(city=f.get("city"),max_price=f.get("maxPrice"),min_price=f.get("minPrice"),
        min_beds=f.get("beds"),pool=f.get("pool"),
        ptype=TYPE_DB.get(f.get("propertyType")) if f.get("propertyType") else None)
    for m in MODES:
        ids=[x.id for x in hybrid_search(text, flt, k=20, mode=m)]
        for k in (5,10,20):
            agg[m][k][0]+=int(tgt in ids[:k]); agg[m][k][1]+=1
        per[m].append(int(tgt in ids[:10]))
n=agg["hybrid"][10][1]
print(f"known-item · {n} 条带 target · auto 路径(filter + semantic 文本)\n")
print(f"  {'模式':<9}{'R@5':>8}{'R@10':>8}{'R@20':>8}")
for m in MODES:
    a=agg[m]; print(f"  {m:<9}"+"".join(f"{a[k][0]/a[k][1]:>8.3f}" for k in (5,10,20)))
print("\n  配对 (recall@10, hybrid vs 单路):")
for m in ("dense","bm25"):
    d=[h-s for h,s in zip(per["hybrid"],per[m])]
    w=sum(1 for x in d if x>0); l=sum(1 for x in d if x<0); md=sum(d)/len(d)
    sd=(sum((x-md)**2 for x in d)/(len(d)-1))**.5 if len(d)>1 else 0
    t=md/(sd/len(d)**.5) if sd else 0
    print(f"    hybrid vs {m:<6}: 胜{w}/平{len(d)-w-l}/负{l}  meanΔ {md:+.3f}  t≈{t:.2f}  {'显著' if abs(t)>=2 else '不显著'}")
