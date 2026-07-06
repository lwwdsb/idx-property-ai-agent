"""Week 7: recommendation + comp-based price validation + verifier.

recommend(listing_id):
  1. similar listings = 0.6 * structured similarity + 0.4 * semantic similarity
     (semantic reuses the Week-6 Qdrant vectors — no recompute).
  2. each candidate is price-validated against recent california_sold comps
     (same city, ±20% living area, last N months), comparing $/sqft.
  3. VERIFIER (the signature guardrail): a deterministic gate withholds any price
     judgment when the comp sample is too small (甲: don't be confidently wrong).
     An optional LLM layer (llm.explain_verdict) adds reasoning ON TOP when a key
     is configured — it never overrides the deterministic floor.
"""
from datetime import date, timedelta

from qdrant_client import models

from common import COLLECTION, get_mysql, get_qdrant
from llm import explain_verdict

MIN_COMPS = 3          # below this -> withhold a verdict
STRONG_COMPS = 8       # at/above this -> high confidence
FAIR_BAND = 5.0        # within ±5% of comp estimate = "in line"


def median(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    m = len(xs) // 2
    return xs[m] if len(xs) % 2 else (xs[m - 1] + xs[m]) / 2


# ---- structured similarity (0..1) ----
def _rel(a, b):
    if not a or not b:
        return 0.0
    return max(0.0, 1.0 - abs(a - b) / a)


def structured_similarity(a: dict, b: dict) -> float:
    price = _rel(a.get("price"), b.get("price"))
    sqft = _rel(a.get("sqft"), b.get("sqft"))
    beds = 1.0 if a.get("beds") == b.get("beds") else (0.5 if a.get("beds") and b.get("beds")
                                                       and abs(a["beds"] - b["beds"]) <= 1 else 0.0)
    city = 1.0 if (a.get("city") or "").lower() == (b.get("city") or "").lower() else 0.0
    return round(0.4 * price + 0.3 * sqft + 0.2 * beds + 0.1 * city, 4)


# ---- comp-based price validation + verifier ----
def validate_price(listing: dict, months: int = 6, area_tol: float = 0.20) -> dict:
    city, sqft, price = listing.get("city"), listing.get("sqft"), listing.get("price")
    frm = (date.today() - timedelta(days=30 * months)).isoformat()
    to = date.today().isoformat()

    sql = ("SELECT ClosePrice, LivingArea FROM california_sold "
           "WHERE City=%s AND ClosePrice>0 AND CloseDate>=%s AND CloseDate<=%s")
    params = [city, frm, to]
    if sqft:
        sql += " AND LivingArea BETWEEN %s AND %s"
        params += [sqft * (1 - area_tol), sqft * (1 + area_tol)]
    conn = get_mysql()
    with conn.cursor() as cur:
        cur.execute(sql, params)
        comps = cur.fetchall()
    conn.close()

    n = len(comps)
    out = {"comp_count": n, "list_price": price}

    # VERIFIER gate (deterministic, the safety floor) ---------------------
    if n < MIN_COMPS:
        out.update(confidence="low", assessment=None,
                   verdict=f"Only {n} comparable sale(s) found — not enough to judge pricing.")
        return out

    comp_median_price = median([c["ClosePrice"] for c in comps])
    ppsf = median([c["ClosePrice"] / c["LivingArea"] for c in comps if c["LivingArea"]])
    if sqft and ppsf:
        estimate = ppsf * sqft
    else:
        estimate = comp_median_price
    delta = (price - estimate) / estimate * 100 if (price and estimate) else None

    if delta is None:
        verdict, assessment = "Listing price unavailable.", None
    elif abs(delta) <= FAIR_BAND:
        verdict, assessment = "Priced in line with recent comparable sales.", "fair"
    elif delta > 0:
        verdict, assessment = f"Asking ~{delta:.0f}% above recent comparable sales.", "above"
    else:
        verdict, assessment = f"Asking ~{-delta:.0f}% below recent comparable sales.", "below"

    out.update(
        confidence="high" if n >= STRONG_COMPS else "medium",
        comp_median_price=round(comp_median_price) if comp_median_price else None,
        comp_median_ppsf=round(ppsf) if ppsf else None,
        estimate=round(estimate) if estimate else None,
        delta_pct=round(delta, 1) if delta is not None else None,
        assessment=assessment, verdict=verdict,
    )
    # reserved LLM reasoning layer (on top of, never replacing, the verdict)
    reasoning = explain_verdict({"listing": listing, "comps": out})
    if reasoning:
        out["explanation"] = reasoning
    return out


# ---- recommendation ----
def get_point(client, listing_id: int):
    pts = client.retrieve(COLLECTION, ids=[listing_id], with_payload=True)
    return pts[0].payload if pts else None


def blend_and_rank(liked: dict, candidates: list, k: int = 5) -> list:
    """Pure ranking core (no I/O, unit-testable): blend 0.6*structured + 0.4*semantic,
    sort, take top-k, and attach a comp-based price check to each.
    `candidates` = list of (payload, semantic_score)."""
    scored = []
    for payload, sem in candidates:
        struct = structured_similarity(liked, payload)
        blended = round(0.6 * struct + 0.4 * float(sem), 4)
        scored.append((blended, struct, float(sem), payload))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [{"listing": pl, "score": b, "structured": s, "semantic": round(sem, 4),
             "pricing": validate_price(pl)} for b, s, sem, pl in scored[:k]]


def recommend(listing_id: int, k: int = 5, pool: int = 40) -> dict:
    client = get_qdrant()
    liked = get_point(client, listing_id)
    if not liked:
        return {"error": f"listing {listing_id} not found in vector store"}

    # semantic candidates: same city + a price band, ranked by the liked vector
    flt = models.Filter(must=[models.FieldCondition(key="city", match=models.MatchValue(value=liked.get("city")))])
    if liked.get("price"):
        flt.must.append(models.FieldCondition(key="price", range=models.Range(
            gte=liked["price"] * 0.6, lte=liked["price"] * 1.6)))
    res = client.query_points(COLLECTION, query=listing_id, using="dense",
                              query_filter=flt, limit=pool, with_payload=True)
    candidates = [(p.payload, p.score) for p in res.points if p.id != listing_id]
    return {"liked": liked, "recommendations": blend_and_rank(liked, candidates, k)}


def format_reco(result: dict) -> str:
    if "error" in result:
        return result["error"]
    liked = result["liked"]
    lines = [f"You liked: {liked.get('address')}, {liked.get('city')} "
             f"(${int(liked['price']):,}, {liked.get('beds')}bd, {liked.get('sqft')}sqft)\n"
             f"Similar listings + price check:"]
    for r in result["recommendations"]:
        pl, pr = r["listing"], r["pricing"]
        price = f"${int(pl['price']):,}" if pl.get("price") else "N/A"
        lines.append(f"\n• {pl.get('address')}, {pl.get('city')} — {price} "
                     f"({pl.get('beds')}bd/{pl.get('sqft')}sqft)  [match {r['score']:.2f}]")
        tag = f" ({pr['confidence']} confidence, {pr['comp_count']} comps)" if pr.get("assessment") else ""
        lines.append(f"    💰 {pr['verdict']}{tag}")
        if pr.get("explanation"):
            lines.append(f"    🧠 {pr['explanation']}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    lid = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    print(format_reco(recommend(lid)))
