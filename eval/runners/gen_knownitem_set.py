"""Expand the OBJECTIVE known-item set — the only eval set that can grow without human
relevance judgements, because its gold is "this exact listing", not "how relevant is this".

  python eval/runners/gen_knownitem_set.py --n 60 --out eval/datasets/mode_retrieval_gen.jsonl

FOUR RULES THAT KEEP IT HONEST. Each exists because the obvious shortcut biases the set.

1. gold.filter is DERIVED FROM THE LISTING'S OWN COLUMNS, never from the LLM. The hard
   constraints are ground truth by construction, so a parse-F1 measured against them is
   measuring the parser, not another model's guess.

2. The query is a PARAPHRASE, never an excerpt. An LLM writing a query while looking at the
   remark will copy its distinctive words, and word overlap is exactly what rewards BM25 —
   the same confound already found in the LLM-judge labels, re-entering through the query
   side. The prompt forbids reusing proper nouns, project/community names, and rare terms.

3. Candidates are NOT filtered by whether the current system retrieves them. Keeping only
   queries that the live config already answers would make recall@10 approach 1.0 by
   construction and tie the set to today's parameters — it would stop being able to detect
   a regression, which is the one job it has.

4. Every row is tagged source="llm-paraphrase", verified=false, and carries pool_size (how
   many active listings satisfy gold.filter). pool_size is both a difficulty signal and a
   FAIRNESS check: if thousands of listings match the stated constraints, the query does not
   actually identify its target and recall@10 is not a fair question. The tag also lets a
   sweep be re-run on the human subset alone and the conclusions compared — if generated and
   human rows disagree on the DIRECTION of a parameter effect, the generator is a confound
   and this set cannot be trusted. That comparison is the acceptance test, not the row count.
"""
import argparse
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "retrieval"))
import pymysql                          # noqa: E402
from common import load_env             # noqa: E402  (load_env only — importing the rest
from llm import chat, llm_available     # noqa: E402  pulls in fastembed for nothing)


def get_mysql():
    env = load_env()
    return pymysql.connect(
        host=env.get("DB_HOST", "127.0.0.1"), port=int(env.get("DB_PORT", 3306)),
        user=env.get("DB_USER"), password=env.get("DB_PASSWORD"),
        database=env.get("DB_NAME"), cursorclass=pymysql.cursors.DictCursor)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TYPE_LABEL = {"Condominium": "condo", "Townhouse": "townhouse", "SingleFamilyResidence": "single-family"}

# A Chinese row may only be emitted for a city the normalizer can actually resolve
# (src/search/normalize.ts CITY_ALIASES). Otherwise the generated zh subset would lose its
# city constraint while the HUMAN zh rows resolve theirs — the generated set would look
# harder for a reason that has nothing to do with retrieval, which is exactly the kind of
# confound this set exists to avoid. Cities outside the table are emitted in English only.
ZH_CITIES = {
    "Irvine", "Los Angeles", "San Diego", "San Francisco", "San Jose", "Pasadena", "Fullerton",
    "Arcadia", "San Gabriel", "Rowland Heights", "Walnut", "Diamond Bar", "Temple City",
    "Tustin", "Huntington Beach", "Newport Beach", "Anaheim", "Long Beach", "Chino Hills",
    "Monterey Park", "Alhambra", "Cupertino", "Beverly Hills", "Santa Monica",
}
MIN_POOL, MAX_POOL = 8, 400   # below MIN the hard filter alone answers the query, so ranking
                              # is never exercised; above MAX the query does not identify a target

SYSTEM = """You write SEARCH QUERIES that a home buyer would type, from a listing's facts.

HARD RULES
- Write what a buyer WANTS, never a description of a specific listing. No addresses, no MLS
  numbers, no community/project/building names, no street names, no school names.
- Do NOT reuse distinctive or rare words from the remark. PARAPHRASE the gist into ordinary
  buyer language. If the remark says "soaring coffered ceilings", write "high ceilings".
- Include the numeric constraints you are given, phrased naturally ("under 1.6M", "3 bed").
- One sentence, max ~18 words. No quotes, no preamble.

Return STRICT JSON: {"en": "<english query>", "zh": "<the same query in natural Chinese>"}"""


def sample_listings(n, seed, zh_n=0):
    """Diverse across city / price band / type, and only listings with a substantive remark
    (a query cannot describe a listing that describes nothing).

    `zh_n` of them are drawn from ZH_CITIES specifically. Diversity-first sampling spreads
    across hundreds of California cities, of which the normalizer can resolve 24 — so an
    unstratified sample produced ZERO usable Chinese rows. The Chinese share has to be
    sampled for, not filtered for."""
    conn = get_mysql()
    with conn.cursor() as cur:
        cur.execute("""
            SELECT id, L_City, L_Type_, L_Keyword2, LM_Dec_3, LM_Int2_3, L_SystemPrice,
                   PoolPrivateYN, L_Remarks
            FROM rets_property
            WHERE L_Status='Active' AND L_City IS NOT NULL AND L_City<>''
              AND L_SystemPrice > 100000 AND L_Keyword2 > 0
              AND CHAR_LENGTH(L_Remarks) BETWEEN 200 AND 2000
        """)
        rows = cur.fetchall()
    conn.close()

    rnd = random.Random(seed)
    rnd.shuffle(rows)

    def band(p):
        p = float(p)
        return 0 if p < 700_000 else 1 if p < 1_200_000 else 2 if p < 2_000_000 else 3

    def spread(pool, k):
        """Round-robin over (city, price band) so one hot city cannot dominate."""
        buckets = {}
        for r in pool:
            buckets.setdefault((r["L_City"], band(r["L_SystemPrice"])), []).append(r)
        keys = sorted(buckets)
        rnd.shuffle(keys)
        picked = []
        while len(picked) < k and keys:
            for key in list(keys):
                if buckets[key]:
                    picked.append(buckets[key].pop())
                    if len(picked) >= k:
                        break
                else:
                    keys.remove(key)
        return picked

    zh_pool = [r for r in rows if r["L_City"] in ZH_CITIES]
    zh_pick = spread(zh_pool, zh_n)
    taken = {id(r) for r in zh_pick}
    en_pick = spread([r for r in rows if id(r) not in taken], n - len(zh_pick))
    out = zh_pick + en_pick
    rnd.shuffle(out)
    return out


def pool_size(conn, flt):
    where, params = ["L_Status='Active'"], []
    if flt.get("city"):
        where.append("L_City=%s"); params.append(flt["city"])
    if flt.get("beds") is not None:
        where.append("L_Keyword2>=%s"); params.append(flt["beds"])
    if flt.get("maxPrice") is not None:
        where.append("L_SystemPrice<=%s"); params.append(flt["maxPrice"])
    if flt.get("propertyType"):
        inv = {v: k for k, v in TYPE_LABEL.items()}
        where.append("L_Type_=%s"); params.append(inv[flt["propertyType"]])
    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) AS n FROM rets_property WHERE {' AND '.join(where)}", params)
        return int(cur.fetchone()["n"])


_DF = {}          # token -> in how many remarks it appears
_DF_N = 0


def build_df(conn, sample=12000):
    """Document frequency over the remark corpus, so "distinctive" is measured rather than
    guessed. A hand-written stopword list flagged `fireplace`, `laundry`, `private` and
    `family` as lifted words — they are ordinary buyer vocabulary that appears in thousands
    of listings. Rarity is the property that actually distinguishes a copied community name
    from a word any buyer would type."""
    global _DF, _DF_N
    with conn.cursor() as cur:
        cur.execute("SELECT L_Remarks FROM rets_property WHERE L_Status='Active' "
                    "AND CHAR_LENGTH(L_Remarks) > 100 LIMIT %s", (sample,))
        rows = cur.fetchall()
    _DF_N = len(rows)
    for r in rows:
        for t in set(re.findall(r"[a-z][a-z'-]{4,}", (r["L_Remarks"] or "").lower())):
            _DF[t] = _DF.get(t, 0) + 1
    print(f"leakage guard: document frequency over {_DF_N} remarks, {len(_DF)} tokens")


def leaked_tokens(query, remark, city, max_df_ratio=0.004):
    """Tokens the query shares with THIS remark that are RARE in the corpus — i.e. words the
    model copied rather than words a buyer would say. The city is exempt: it is a required
    constraint we asked for, and city names are rare by construction."""
    city_toks = set(re.findall(r"[a-z]+", (city or "").lower()))
    rem = set(re.findall(r"[a-z][a-z'-]{4,}", (remark or "").lower()))
    out = []
    for t in set(re.findall(r"[a-z][a-z'-]{4,}", query.lower())):
        if t in city_toks or t not in rem:
            continue
        if _DF_N and _DF.get(t, 0) / _DF_N <= max_df_ratio:      # rare -> distinctive -> leak
            out.append(t)
    return sorted(out)


def round_price(p):
    """A buyer says "under 1.6M", not "under 1,549,000" — round UP to a natural budget so the
    target still satisfies the constraint."""
    p = float(p)
    step = 50_000 if p < 1_000_000 else 100_000
    return int((int(p / step) + 1) * step)


def loosen(conn, flt, price):
    """Grow the candidate pool until ranking actually matters, without ever excluding the
    target: drop the type constraint first, then widen the budget. A known-item question is
    only informative when several listings satisfy the stated constraints — otherwise the
    hard filter answers it and every configuration scores the same."""
    n = pool_size(conn, flt)
    if n >= MIN_POOL:
        return flt, n
    if "propertyType" in flt:
        trial = {k: v for k, v in flt.items() if k != "propertyType"}
        if pool_size(conn, trial) >= MIN_POOL:
            return trial, pool_size(conn, trial)
        flt = trial
    for mult in (1.3, 1.8, 2.5):
        trial = {**flt, "maxPrice": int(round(price * mult / 50_000) * 50_000)}
        n = pool_size(conn, trial)
        if n >= MIN_POOL:
            return trial, n
    return flt, pool_size(conn, flt)


def build(row, conn):
    beds = int(float(row["L_Keyword2"]))
    ptype = TYPE_LABEL.get(row["L_Type_"])
    flt = {"city": row["L_City"], "beds": beds, "maxPrice": round_price(row["L_SystemPrice"])}
    if ptype:
        flt["propertyType"] = ptype
    flt, _ = loosen(conn, flt, float(row["L_SystemPrice"]))
    ptype = flt.get("propertyType")
    facts = (f"city={flt['city']}; beds={beds}; "
             f"type={ptype or 'home'}; budget=under ${flt['maxPrice']:,}; "
             f"remark gist (PARAPHRASE, do not copy words): {row['L_Remarks'][:600]}")
    raw = chat(facts, system=SYSTEM, temperature=0.4)
    if not raw:
        return None
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return None
    try:
        q = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    en, zh = (q.get("en") or "").strip(), (q.get("zh") or "").strip()
    if not en:
        return None
    leaked = leaked_tokens(en, row["L_Remarks"], flt["city"])
    return {"listing": row, "filter": flt, "en": en, "zh": zh, "leaked": leaked,
            "pool": pool_size(conn, flt)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=int(os.environ.get("IDX_EVAL_SEED", 7)))
    ap.add_argument("--out", default="eval/datasets/mode_retrieval_gen.jsonl")
    ap.add_argument("--zh-share", type=float, default=0.4, help="fraction emitted as Chinese")
    args = ap.parse_args()
    if not llm_available():
        sys.exit("no LLM key — this generator needs one (LLM_API_KEY)")

    rows = sample_listings(args.n, args.seed, zh_n=int(args.n * args.zh_share))
    print(f"sampled {len(rows)} listings across "
          f"{len({r['L_City'] for r in rows})} cities\n")
    conn = get_mysql()
    build_df(conn)
    rnd = random.Random(args.seed)
    out, skipped = [], {"no_json": 0, "leaked": 0, "pool_out_of_band": 0}
    for i, row in enumerate(rows, 1):
        b = build(row, conn)
        if not b:
            skipped["no_json"] += 1; continue
        if b["leaked"]:
            skipped["leaked"] += 1
            print(f"  [{i:>3}] SKIP leaked {b['leaked']}  {b['en'][:50]}")
            continue
        if not (MIN_POOL <= b["pool"] <= MAX_POOL):
            skipped["pool_out_of_band"] += 1
            print(f"  [{i:>3}] SKIP pool={b['pool']} outside [{MIN_POOL},{MAX_POOL}]")
            continue
        zh = bool(b["zh"]) and b["filter"]["city"] in ZH_CITIES
        out.append({
            "id": f"mrg-{len(out) + 1:03d}",
            "input": b["zh"] if zh else b["en"],
            "style": "generated",
            "lang": "zh" if zh else "en",
            "gold": {"filter": b["filter"], "semantic": None,
                     "intents": ["search"], "known_item": int(row["id"])},
            "meta": {"source": "llm-paraphrase", "verified": False,
                     "pool_size": b["pool"], "price": float(row["L_SystemPrice"])},
        })
        print(f"  [{i:>3}] {'zh' if zh else 'en'} pool={b['pool']:>4}  {out[-1]['input'][:62]}")
    conn.close()

    path = os.path.join(ROOT, args.out)
    with open(path, "w", encoding="utf-8") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nwrote {len(out)} rows -> {args.out}   skipped: {skipped}")
    print("NOTE verified=false. Acceptance is not the row count — it is whether a sweep on "
          "these rows agrees in DIRECTION with the human rows (eval/runners/compare_sources.py).")


if __name__ == "__main__":
    main()
