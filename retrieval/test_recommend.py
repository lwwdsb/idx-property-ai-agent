"""Correctness checks for the deterministic Week-7 core (no Qdrant needed).
Covers structured similarity, comp-based price validation, and the verifier gate.

  python retrieval/test_recommend.py
"""
from recommend import structured_similarity, validate_price, median, blend_and_rank, MIN_COMPS

passed = failed = 0


def check(name, cond):
    global passed, failed
    if cond:
        passed += 1; print("✓", name)
    else:
        failed += 1; print("✗", name)


# ---- median ----
check("median odd", median([3, 1, 2]) == 2)
check("median even", median([1, 2, 3, 4]) == 2.5)
check("median resists outlier", median([1, 1, 1, 1, 100]) == 1)

# ---- structured similarity ----
base = {"price": 1_500_000, "sqft": 2000, "beds": 3, "city": "Irvine"}
check("identical -> 1.0", structured_similarity(base, base) == 1.0)
check("different city penalized",
      structured_similarity(base, {**base, "city": "San Diego"}) < structured_similarity(base, base))
check("far price lowers score",
      structured_similarity(base, {**base, "price": 3_000_000}) < structured_similarity(base, {**base, "price": 1_550_000}))

# ---- verifier gate: too few comps -> WITHHOLD ----
withheld = validate_price({"city": "NotARealCityXYZ", "sqft": 2000, "price": 1_000_000})
check("unknown city -> withhold (assessment None)", withheld["assessment"] is None)
check("withhold verdict mentions not enough", "not enough" in withheld["verdict"].lower())
check("withhold comp_count < MIN", withheld["comp_count"] < MIN_COMPS)

# ---- comp validation on real data: fair / above / below are deterministic ----
probe = validate_price({"city": "Irvine", "sqft": 2000, "price": 1})  # read the estimate
check("Irvine has enough comps", probe["comp_count"] >= MIN_COMPS)
est = probe["estimate"]
check("estimate computed", est and est > 100_000)

fair = validate_price({"city": "Irvine", "sqft": 2000, "price": est})
above = validate_price({"city": "Irvine", "sqft": 2000, "price": int(est * 1.5)})
below = validate_price({"city": "Irvine", "sqft": 2000, "price": int(est * 0.5)})
check("price == estimate -> fair", fair["assessment"] == "fair")
check("price 1.5x -> above", above["assessment"] == "above" and above["delta_pct"] > 0)
check("price 0.5x -> below", below["assessment"] == "below" and below["delta_pct"] < 0)
check("confidence set when comps sufficient", fair["confidence"] in ("medium", "high"))

# ---- blend + rank core (offline, no Qdrant): 0.6*structured + 0.4*semantic ----
liked = {"price": 1_500_000, "sqft": 2000, "beds": 3, "city": "Irvine"}
cands = [
    ({"address": "A", "price": 1_520_000, "sqft": 2050, "beds": 3, "city": "Irvine"}, 0.70),  # struct high, sem ok
    ({"address": "B", "price": 1_500_000, "sqft": 2000, "beds": 3, "city": "Irvine"}, 0.60),  # struct ~1
    ({"address": "C", "price": 3_000_000, "sqft": 4000, "beds": 6, "city": "Irvine"}, 0.95),  # sem high, struct low
]
ranked = blend_and_rank(liked, cands, k=3)
check("blend returns k results", len(ranked) == 3)
check("blend sorted desc", all(ranked[i]["score"] >= ranked[i + 1]["score"] for i in range(len(ranked) - 1)))
check("structured-similar B beats semantic-only C",
      next(r for r in ranked if r["listing"]["address"] == "B")["score"]
      > next(r for r in ranked if r["listing"]["address"] == "C")["score"])
check("each rec has a price check", all("pricing" in r for r in ranked))
check("blend weight = 0.6*struct+0.4*sem",
      abs(ranked and next(r for r in ranked if r["listing"]["address"] == "B")["score"]
          - (0.6 * 1.0 + 0.4 * 0.60)) < 1e-9)

print(f"\n{passed}/{passed + failed} passed")
raise SystemExit(1 if failed else 0)
