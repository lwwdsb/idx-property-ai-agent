"""Offline ingest: MySQL active listings -> Qdrant (dense + BM25 sparse + payload).

Usage:
  python retrieval/ingest.py                 # full ingest
  python retrieval/ingest.py --limit 500     # small test batch
  python retrieval/ingest.py --city Irvine   # one city
  python retrieval/ingest.py --recreate      # drop + rebuild the collection
"""
import argparse
import sys
import time

import pymysql
from qdrant_client import models

from common import (COLLECTION, build_doc_text, build_payload, ensure_collection,
                    get_dense, get_qdrant, get_sparse, load_env)

FIELDS = ("id, L_DisplayId, L_Address, L_City, L_Type_, L_Keyword2, LM_Dec_3, "
          "LM_Int2_3, L_SystemPrice, PoolPrivateYN, L_Remarks")


def fetch_rows(limit=None, city=None):
    env = load_env()
    conn = pymysql.connect(
        host=env.get("DB_HOST", "127.0.0.1"), port=int(env.get("DB_PORT", 3306)),
        user=env.get("DB_USER", "root"), password=env.get("DB_PASSWORD", ""),
        database=env.get("DB_NAME", "idx_exchange"), cursorclass=pymysql.cursors.DictCursor)
    sql = f"SELECT {FIELDS} FROM rets_property WHERE L_Status='Active'"
    params = []
    if city:
        sql += " AND L_City=%s"; params.append(city)
    if limit:
        sql += " LIMIT %s"; params.append(int(limit))
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    conn.close()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--city")
    ap.add_argument("--recreate", action="store_true")
    ap.add_argument("--resume", action="store_true", help="skip ids already in Qdrant (interruptible)")
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    client = get_qdrant()
    ensure_collection(client, recreate=args.recreate)

    rows = fetch_rows(args.limit, args.city)
    if args.resume and not args.recreate:
        done_ids, offset = set(), None
        while True:
            pts, offset = client.scroll(COLLECTION, limit=10000, offset=offset,
                                        with_payload=False, with_vectors=False)
            done_ids.update(p.id for p in pts)
            if offset is None:
                break
        before = len(rows)
        rows = [r for r in rows if int(r["id"]) not in done_ids]
        print(f"resume: {len(done_ids)} already ingested, {before - len(rows)} skipped")
    print(f"fetched {len(rows)} listings to embed; upserting...")
    if not rows:
        print("nothing to do — already complete.")
        return

    dense, sparse = get_dense(), get_sparse()
    t0 = time.time()
    done = 0
    for i in range(0, len(rows), args.batch):
        chunk = rows[i:i + args.batch]
        texts = [build_doc_text(r) for r in chunk]
        dvecs = list(dense.embed(texts))
        svecs = list(sparse.embed(texts))
        points = []
        for r, d, s in zip(chunk, dvecs, svecs):
            points.append(models.PointStruct(
                id=int(r["id"]),
                vector={
                    "dense": d.tolist(),
                    "bm25": models.SparseVector(indices=s.indices.tolist(), values=s.values.tolist()),
                },
                payload=build_payload(r),
            ))
        client.upsert(COLLECTION, points=points, wait=True)
        done += len(points)
        sys.stdout.write(f"\r  {done}/{len(rows)}  ({done/(time.time()-t0):.0f}/s)")
        sys.stdout.flush()
    print(f"\ndone in {time.time()-t0:.0f}s. collection '{COLLECTION}' count:",
          client.count(COLLECTION).count)


if __name__ == "__main__":
    main()
