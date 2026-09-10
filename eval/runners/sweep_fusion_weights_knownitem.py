"""Sweep dense/bm25 fusion weights against KNOWN-ITEM recall (objective labels).

Production uses Qdrant's built-in RRF, which is equal-weight and has no weight knob; this
reimplements weighted RRF client-side so other ratios can be compared against it.
Needs Qdrant + the mode_retrieval predictions.
  python eval/runners/sweep_fusion_weights_knownitem.py

在 KNOWN-ITEM(客观标签)上扫 dense/bm25 融合权重 —— 生产用的是 Qdrant 内置 RRF(等权、
无权重旋钮),这里用自实现加权 RRF 复现并对比其他配比,验证"等权"是否真的站得住。"""
import json, os, sys
ROOT=os.getcwd(); sys.path.insert(0, os.path.join(ROOT,"retrieval"))
from search import hybrid_search, build_filter
TYPE_DB={"house":"Single Family Residence","condo":"Condominium","townhouse":"Townhouse"}
PREFETCH=30
def wrrf(dl, bl, wd, wb, k, topk=10):
    sc={}
    for r,pid in enumerate(dl): sc[pid]=sc.get(pid,0.0)+wd/(k+r+1)
    for r,pid in enumerate(bl): sc[pid]=sc.get(pid,0.0)+wb/(k+r+1)
    return [p for p,_ in sorted(sc.items(), key=lambda x:-x[1])][:topk]

gold={json.loads(l)["id"]:json.loads(l) for l in open("eval/datasets/mode_retrieval.jsonl") if l.strip()}
preds={json.loads(l)["id"]:json.loads(l) for l in open("eval/history/mode_retrieval.preds.jsonl") if l.strip()}
cases=[]
for cid,g in gold.items():
    tgt=g["gold"].get("known_item")
    if not tgt or cid not in preds: continue
    p=preds[cid]; f=p["auto_filter"]
    text=(p.get("auto_semantic") or "").strip() or g["input"]
    flt=build_filter(city=f.get("city"),max_price=f.get("maxPrice"),min_price=f.get("minPrice"),
        min_beds=f.get("beds"),pool=f.get("pool"),
        ptype=TYPE_DB.get(f.get("propertyType")) if f.get("propertyType") else None)
    dl=[x.id for x in hybrid_search(text, flt, k=PREFETCH, mode="dense")]
    bl=[x.id for x in hybrid_search(text, flt, k=PREFETCH, mode="bm25")]
    nat=[x.id for x in hybrid_search(text, flt, k=10, mode="hybrid")]
    cases.append((tgt, dl, bl, nat))

print(f"known-item · {len(cases)} 条 · 自实现加权 RRF(prefetch={PREFETCH})\n")
nat_r=sum(1 for t,_,_,n in cases if t in n[:10])/len(cases)
print(f"  {'配比 dense:bm25':<18}{'k=4':>9}{'k=60':>9}")
for wd,wb,name in [(1,3,"1:3"),(1,2,"1:2"),(1,1,"1:1 等权"),(2,1,"2:1"),(3,1,"3:1")]:
    row=f"  {name:<18}"
    for k in (4,60):
        hit=sum(1 for t,dl,bl,_ in cases if t in wrrf(dl,bl,wd,wb,k))/len(cases)
        row+=f"{hit:>9.3f}"
    print(row)
print(f"\n  生产实际(Qdrant 内置 RRF,等权): {nat_r:.3f}")
