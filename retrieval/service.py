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
from fastapi import FastAPI
from pydantic import BaseModel

from common import get_dense
from rag import RagIndex, answer as rag_answer
from recommend import validate_price, recommend as do_recommend

# Example utterances per skill — the embedding intent classifier matches a message
# against these (generalizes past regex). Kept small; embedded once at startup.
SKILL_EXAMPLES = {
    "search": ["find 3 bedroom homes in Irvine", "houses under 1 million with a pool",
               "condos in San Diego", "在 Irvine 找带泳池的房子", "show me listings in Tustin"],
    "market": ["what is the market like in Irvine", "median sold price in San Diego",
               "price per square foot in Irvine", "Irvine 行情怎么样", "are prices going up"],
    "recommend": ["show me similar homes", "recommend listings like this one",
                  "comparable properties to this", "跟这套类似的房子", "more homes like that"],
    "knowledge": ["what does DOM mean", "how is the sold to list ratio calculated",
                  "what is price per square foot", "什么是 comp", "explain days on market"],
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
    yield
    STATE.clear()


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


@app.post("/rag")
def rag(req: RagReq):
    return rag_answer(req.question, STATE["rag"])


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
        return do_recommend(req.listing_id)
    except Exception as e:  # Qdrant down etc. — degrade, don't 500
        return {"error": f"recommendation unavailable: {e}"}
