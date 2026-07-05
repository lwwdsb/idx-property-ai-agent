"""Shared setup for the hybrid retrieval subsystem (Week 6).

Dense (bge-small, local via fastembed) + sparse BM25 (Qdrant/bm25) vectors stored
in Qdrant, fused with RRF at query time. All local + free — no embedding API key.
Structured fields go into the Qdrant payload so hard filters (city/price/beds) run
as filtered ANN inside Qdrant ("先筛" during HNSW traversal).
"""
import os
from functools import lru_cache

from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding

DENSE_MODEL = "BAAI/bge-small-en-v1.5"   # 384-dim, small + fast, runs on CPU
DENSE_DIM = 384
SPARSE_MODEL = "Qdrant/bm25"
COLLECTION = os.environ.get("QDRANT_COLLECTION", "listings")


def load_env(path=None):
    """Minimal .env loader (DB creds + qdrant url)."""
    path = path or os.path.join(os.path.dirname(__file__), "..", ".env")
    env = {}
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


@lru_cache(maxsize=1)
def get_qdrant():
    env = load_env()
    url = env.get("QDRANT_URL", "http://localhost:6333")
    return QdrantClient(url=url)


@lru_cache(maxsize=1)
def get_dense():
    return TextEmbedding(DENSE_MODEL)


@lru_cache(maxsize=1)
def get_sparse():
    return SparseTextEmbedding(SPARSE_MODEL)


def build_doc_text(r: dict) -> str:
    """The text embedded + BM25-indexed per listing. Remarks carry most of the
    semantic/keyword signal; structured bits add context."""
    parts = [
        r.get("L_Type_") or "",
        f"in {r.get('L_City')}" if r.get("L_City") else "",
        f"{r.get('L_Keyword2')} bed" if r.get("L_Keyword2") else "",
        f"{r.get('LM_Dec_3')} bath" if r.get("LM_Dec_3") else "",
        f"{r.get('LM_Int2_3')} sqft" if r.get("LM_Int2_3") else "",
        r.get("L_Remarks") or "",
    ]
    return " ".join(p for p in parts if p).strip()


def build_payload(r: dict) -> dict:
    """Structured fields for filtering + display (kept in Qdrant payload)."""
    def num(v):
        try:
            return None if v is None or v == "" else float(v)
        except (TypeError, ValueError):
            return None
    return {
        "listing_id": int(r["id"]),
        "mls": r.get("L_DisplayId"),
        "address": r.get("L_Address"),
        "city": r.get("L_City"),
        "type": r.get("L_Type_"),
        "beds": num(r.get("L_Keyword2")),
        "baths": num(r.get("LM_Dec_3")),
        "sqft": num(r.get("LM_Int2_3")),
        "price": num(r.get("L_SystemPrice")),
        "pool": str(r.get("PoolPrivateYN") or "") == "1",
    }


def ensure_collection(client, recreate=False):
    exists = client.collection_exists(COLLECTION)
    if exists and recreate:
        client.delete_collection(COLLECTION)
        exists = False
    if not exists:
        client.create_collection(
            COLLECTION,
            vectors_config={"dense": models.VectorParams(size=DENSE_DIM, distance=models.Distance.COSINE)},
            # IDF modifier => Qdrant computes BM25 IDF across the collection.
            sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
        # payload indexes to make hard filters fast
        for field, schema in [("city", models.PayloadSchemaType.KEYWORD),
                              ("price", models.PayloadSchemaType.FLOAT),
                              ("beds", models.PayloadSchemaType.FLOAT),
                              ("type", models.PayloadSchemaType.KEYWORD),
                              ("pool", models.PayloadSchemaType.BOOL)]:
            client.create_payload_index(COLLECTION, field_name=field, field_schema=schema)
