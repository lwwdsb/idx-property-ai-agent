"""Build the retrieval relevance dataset by POOLING + LLM-assisted grading.

For each semantic query: pool the top-K candidates from dense + BM25 + hybrid (union),
fetch each candidate's remarks locally, and ask the LLM to grade relevance 0/1/2 for
the whole pool in ONE call. Stores ONLY {query, {listing_id: grade}} — never the
confidential remarks text (confidential-data rule).

Labels are source=llm-gen, verified=false; a human spot-checks a sample later and the
eval runner reports the human/LLM agreement rate (honest calibration).

  python eval/runners/build_retrieval_set.py            # build all
  python eval/runners/build_retrieval_set.py --limit 3  # quick test
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "retrieval"))
from common import get_mysql  # noqa: E402
from search import hybrid_search  # noqa: E402
from llm import chat  # noqa: E402

OUT = os.path.join(ROOT, "eval", "datasets", "retrieval.jsonl")
POOL_K = 10   # top-K per mode to pool

# Pure-semantic queries (no hard filter) so dense/bm25/hybrid are compared apples-to-apples
# on ranking quality. Mix of near-synonym, proper-noun, and descriptive intents.
QUERIES = [
    "cozy craftsman bungalow with lots of character",
    "modern home with an open floor plan and natural light",
    "quiet cul-de-sac family home with a big backyard",
    "fixer upper with potential",
    "energy efficient home with solar panels",
    "gourmet kitchen with granite countertops and stainless steel appliances",
    "ADU or guest house or in-law suite",
    "RV parking and a three car garage",
    "ocean view and walk to the beach",
    "fully remodeled kitchen and updated bathrooms",
    "spacious home for a large family",
    "move-in ready turnkey condo",
    "home with a swimming pool and spa",
    "gated community with security",
    "walkable neighborhood near shops and restaurants",
    "mountain views and privacy",
    "historic charm with original details",
    "smart home with modern upgrades",
    "low maintenance yard and single story",
    "investment property with rental potential",
    "bright corner unit with a balcony",
    "horse property with acreage",
    "waterfront with a private dock",
    "new construction never lived in",
]

JUDGE_SYS = (
    "You grade how well each listing satisfies the DISTINCTIVE requirement in a buyer's "
    "query (e.g. 'craftsman bungalow', 'solar panels', 'ocean view', 'ADU'). Be STRICT:\n"
    "- 2 = the listing SPECIFICALLY has that distinctive feature/style (stated in its text).\n"
    "- 1 = loosely/partially related but not clearly the feature.\n"
    "- 0 = the distinctive feature is ABSENT — even if it's a perfectly nice home.\n"
    "A generic nice house that lacks the specific thing asked for is 0, not 1. Most "
    "listings should be 0 or 1; reserve 2 for clear, specific matches.\n"
    'Return JSON only: {"grades": {"<id>": <0|1|2>, ...}}.'
)


def fetch_remarks(ids):
    """id -> short remark snippet (local only, for judging; never stored)."""
    if not ids:
        return {}
    conn = get_mysql()
    out = {}
    with conn.cursor() as cur:
        fmt = ",".join(["%s"] * len(ids))
        cur.execute(f"SELECT id, L_City, L_Type_, L_Keyword2, L_SystemPrice, L_Remarks "
                    f"FROM rets_property WHERE id IN ({fmt})", list(ids))
        for r in cur.fetchall():
            rem = (r.get("L_Remarks") or "")[:280].replace("\n", " ")
            out[int(r["id"])] = (f"{r.get('L_Type_')} in {r.get('L_City')}, "
                                 f"{r.get('L_Keyword2')}bd, ${r.get('L_SystemPrice')} — {rem}")
    conn.close()
    return out


def pool_for(query):
    ids = []
    for mode in ("dense", "bm25", "hybrid"):
        for p in hybrid_search(query, None, k=POOL_K, mode=mode):
            if p.id not in ids:
                ids.append(p.id)
    return ids


def judge(query, id_to_text):
    listing_block = "\n".join(f"[{i}] {t}" for i, t in id_to_text.items())
    prompt = f"Query: {query}\n\nListings:\n{listing_block}\n\nGrade every listing id."
    raw = chat(prompt, system=JUDGE_SYS)
    try:
        start, end = raw.index("{"), raw.rindex("}")
        grades = json.loads(raw[start:end + 1]).get("grades", {})
        return {str(k): int(v) for k, v in grades.items() if int(v) in (0, 1, 2)}
    except Exception as e:
        print(f"  ! judge parse failed for '{query[:40]}': {e}")
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    queries = QUERIES[: args.limit] if args.limit else QUERIES

    records = []
    for n, q in enumerate(queries, 1):
        ids = pool_for(q)
        texts = fetch_remarks(ids)
        grades = judge(q, texts)
        # keep only ids that got a grade > 0 OR are in pool (0s implied for unjudged pool)
        relevant = {i: g for i, g in grades.items() if g > 0}
        records.append({
            "id": f"ret-{n:03d}", "input": q,
            "label": {"relevant": relevant, "pool_size": len(ids)},
            "meta": {"source": "llm-gen", "verified": False},
        })
        print(f"  [{n}/{len(queries)}] {q[:45]:45}  pool={len(ids)} relevant={len(relevant)}")

    with open(OUT, "w") as f:
        f.write("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n")
    print(f"\nwrote {len(records)} queries to {OUT} (ids+grades only, no remarks)")


if __name__ == "__main__":
    main()
