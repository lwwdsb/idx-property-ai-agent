"""Warm FastAPI retrieval service (Week 10).

Loads the embedding model + RAG index + intent-example vectors ONCE at startup and
keeps them hot, so each request is ~tens of ms instead of paying a ~2-5s model
reload per call (the latency killer of the per-subprocess bridge). The TS
orchestrator talks to this over HTTP.

  uvicorn service:app --port 8099      (from retrieval/, venv active)
Endpoints: /health /classify /rag /validate /recommend
"""
from contextlib import asynccontextmanager

import numpy as np
import pymysql
from fastapi import FastAPI
from pydantic import BaseModel
from fastembed.rerank.cross_encoder import TextCrossEncoder

from common import get_dense, load_env
from rag import RagIndex, answer as rag_answer
from recommend import validate_price, recommend as do_recommend, format_reco
from search import hybrid_search, build_filter

RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"   # small cross-encoder; L-12 gave no gain
RERANK_COARSE = 30                                # coarse pool -> CE rerank -> top-k

# Example utterances per skill — the embedding intent classifier matches a message
# against these (generalizes past regex). Kept small; embedded once at startup.
SKILL_EXAMPLES = {
    "search": [
        "find 3 bedroom homes in Irvine", "houses under 1 million with a pool",
        "condos in San Diego", "在 Irvine 找带泳池的房子", "show me listings in Tustin",
        "looking for a single family home with a big yard", "4 bed 3 bath near the beach",
        "any townhouses for sale in Pasadena", "帮我找一套学区好的房子", "洛杉矶两百万以下的公寓",
        "properties with an ocean view under 3 million", "有没有带车库的独栋"],
    "market": [
        "what is the market like in Irvine", "median sold price in San Diego",
        "price per square foot in Irvine", "Irvine 行情怎么样", "are prices going up",
        "how fast are homes selling here", "is it a buyer's or seller's market",
        "how much have prices appreciated this year", "这个城市房价走势如何", "最近成交价怎么样",
        "average days on market in Tustin"],
    "recommend": [
        "show me similar homes", "recommend listings like this one",
        "comparable properties to this", "跟这套类似的房子", "more homes like that",
        "anything else like the first one", "跟第二个差不多的还有吗", "find me options like this property",
        "what else matches this style"],
    "knowledge": [
        "what does DOM mean", "how is the sold to list ratio calculated",
        "what is price per square foot", "什么是 comp", "explain days on market",
        "what does contingent mean", "define escrow", "how do you compute a price estimate",
        "什么是成交挂牌比", "how many comparables do you need", "which field stores the price"],
    "email": [
        "email the market report to my client", "send this to buyer@example.com",
        "把行情发邮件给经纪人", "can you email the Irvine summary to lisa@x.com",
        "发送这份报告到邮箱", "email these listings to me"],
}

STATE = {}


@asynccontextmanager
async def lifespan(app):
    dense = get_dense()
    STATE["rag"] = RagIndex()                       # embeds the KB once (warm)
    labels, texts = [], []
    for skill, exs in SKILL_EXAMPLES.items():
        for e in exs:
            labels.append(skill); texts.append(e)
    m = np.array(list(dense.embed(texts)), dtype=np.float32)
    STATE["intent_labels"] = labels
    STATE["intent_mat"] = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)
    try:
        STATE["reranker"] = TextCrossEncoder(RERANK_MODEL)   # warm-loaded once
    except Exception:
        STATE["reranker"] = None                             # degrade: search still works, no rerank
    yield
    STATE.clear()


def _remarks(ids):
    """Fetch listing text for rerank candidates (payload has no remarks). Small IN-query by PK."""
    if not ids:
        return {}
    env = load_env()
    conn = pymysql.connect(host=env.get("DB_HOST", "127.0.0.1"), port=int(env.get("DB_PORT", 3306)),
                           user=env.get("DB_USER"), password=env.get("DB_PASSWORD"),
                           database=env.get("DB_NAME"), cursorclass=pymysql.cursors.DictCursor)
    fmt = ",".join(["%s"] * len(ids))
    with conn.cursor() as cur:
        cur.execute(f"SELECT id,L_City,L_Type_,L_Remarks FROM rets_property WHERE id IN ({fmt})", ids)
        rows = cur.fetchall()
    conn.close()
    return {int(r["id"]): f"{r.get('L_Type_') or ''} in {r.get('L_City') or ''}. "
            f"{(r.get('L_Remarks') or '').replace(chr(10), ' ')[:400]}" for r in rows}


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {"ok": True, "rag_chunks": len(STATE["rag"].chunks)}


class ClassifyReq(BaseModel):
    message: str


@app.post("/classify")
def classify(req: ClassifyReq):
    """Embedding-based intent: cosine of the message vs each skill's example set."""
    q = list(get_dense().embed([req.message]))[0].astype(np.float32)
    q = q / (np.linalg.norm(q) + 1e-9)
    scores = STATE["intent_mat"] @ q
    best_by_skill = {}
    for label, s in zip(STATE["intent_labels"], scores):
        best_by_skill[label] = max(best_by_skill.get(label, -1.0), float(s))
    ranked = sorted(best_by_skill.items(), key=lambda kv: kv[1], reverse=True)
    return {"skill": ranked[0][0], "score": ranked[0][1], "ranked": ranked}


class RagReq(BaseModel):
    question: str
    hyde: bool = False   # opt-in HyDE rewrite (auto mode only, set by the TS caller)


@app.post("/rag")
def rag(req: RagReq):
    return rag_answer(req.question, STATE["rag"], hyde=req.hyde)


class ValidateReq(BaseModel):
    city: str | None = None
    sqft: float | None = None
    price: float | None = None


@app.post("/validate")
def validate(req: ValidateReq):
    return validate_price({"city": req.city, "sqft": req.sqft, "price": req.price})


class RecommendReq(BaseModel):
    listing_id: int


@app.post("/recommend")
def recommend(req: RecommendReq):
    try:
        return {"reply": format_reco(do_recommend(req.listing_id))}  # human-readable cards
    except Exception as e:  # Qdrant down etc. — degrade, don't 500
        return {"error": f"recommendation unavailable: {e}"}


class SearchReq(BaseModel):
    text: str                       # semantic query (the soft, unstructured intent)
    city: str | None = None
    max_price: float | None = None
    min_price: float | None = None
    min_beds: float | None = None
    pool: bool | None = None        # tri-state: True=has / False=no / None=don't care
    ptype: str | None = None        # physical L_Type_ value (e.g. "Condominium")
    k: int = 5
    rerank: bool = True             # coarse hybrid top-30 -> cross-encoder rerank -> top-k


@app.post("/search")
def search(req: SearchReq):
    """Hybrid semantic search: hard filters + dense+BM25 RRF (coarse), then optional
    cross-encoder rerank of the top-30 -> top-k (precision). Rerank failure degrades to
    plain hybrid order; Qdrant failure degrades to the caller's MySQL path."""
    try:
        flt = build_filter(req.city, req.max_price, req.min_price, req.min_beds, req.pool, req.ptype)
        reranker = STATE.get("reranker")
        if req.rerank and reranker is not None:
            coarse = hybrid_search(req.text, flt, k=max(RERANK_COARSE, req.k), mode="hybrid")
            try:
                texts = _remarks([int(p.payload["listing_id"]) for p in coarse])
                docs = [texts.get(int(p.payload["listing_id"]), "") for p in coarse]
                scores = list(reranker.rerank(req.text, docs))
                ranked = sorted(zip(coarse, scores), key=lambda x: -x[1])[:req.k]
                return {"results": [{"score": float(s), "reranked": True, **p.payload} for p, s in ranked]}
            except Exception:
                pts = coarse[:req.k]   # rerank failed -> keep hybrid order (乙)
                return {"results": [{"score": p.score, **p.payload} for p in pts]}
        pts = hybrid_search(req.text, flt, k=req.k, mode="hybrid")
        return {"results": [{"score": p.score, **p.payload} for p in pts]}
    except Exception as e:  # Qdrant down — caller degrades to MySQL
        return {"error": f"semantic search unavailable: {e}", "results": []}
