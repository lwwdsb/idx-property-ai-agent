/**
 * Query layer (Week 3): SearchFilter -> parameterized SQL -> ListingRow[].
 *
 * Design points:
 *  - All SQL is parameterized (`?`) — never concatenate user input (injection guard).
 *  - Column names come from the field dictionary via col() — single source of truth.
 *  - Keyword search uses the L_Remarks FULLTEXT index (MATCH...AGAINST), not LIKE.
 *  - The ≤50 result cap is enforced HERE, centrally (handbook guardrail) — callers
 *    can't request more.
 *  - A COUNT gives a structural signal (empty / too-many) to drive clarification (Q6).
 */
import type { RowDataPacket } from 'mysql2/promise';
import { query } from '../db.js';
import { col } from '../../schema/columns.js';
import { type ListingRow, mapListingRow } from './listingRow.js';
import type { SearchFilter } from './filters.js';

export const MAX_RESULTS = 50;
const TOO_MANY = 200;

const RP = 'rets_property' as const;

/** Normalized propertyType -> physical L_Type_ value. */
const TYPE_DB: Record<string, string> = {
  'condo': 'Condominium',
  'townhouse': 'Townhouse',
  'single-family': 'SingleFamilyResidence',
};

const SELECT_LISTINGS = `SELECT
  id,
  ${col('listingId', RP)}    AS listingId,
  ${col('mlsNumber', RP)}    AS mls,
  ${col('address', RP)}      AS address,
  ${col('city', RP)}         AS city,
  ${col('zip', RP)}          AS zip,
  ${col('propertyType', RP)} AS type,
  ${col('beds', RP)}         AS beds,
  ${col('baths', RP)}        AS baths,
  ${col('livingArea', RP)}   AS sqft,
  ${col('price', RP)}        AS price,
  ${col('photoCount', RP)}   AS photoCount,
  ${col('yearBuilt', RP)}    AS yearBuilt,
  ${col('pool', RP)}         AS pool,
  ${col('latitude', RP)}     AS lat,
  ${col('longitude', RP)}    AS lng
FROM ${RP}`;

function buildWhere(f: SearchFilter): { sql: string; params: unknown[] } {
  const clauses: string[] = [`${col('status', RP)} = 'Active'`];
  const params: unknown[] = [];
  if (f.city) { clauses.push(`${col('city', RP)} = ?`); params.push(f.city); }
  if (f.beds != null) { clauses.push(`${col('beds', RP)} >= ?`); params.push(f.beds); }
  if (f.baths != null) { clauses.push(`${col('baths', RP)} >= ?`); params.push(f.baths); }
  if (f.maxPrice != null) { clauses.push(`${col('price', RP)} <= ?`); params.push(f.maxPrice); }
  if (f.minPrice != null) { clauses.push(`${col('price', RP)} >= ?`); params.push(f.minPrice); }
  if (f.propertyType && TYPE_DB[f.propertyType]) {
    clauses.push(`${col('propertyType', RP)} = ?`); params.push(TYPE_DB[f.propertyType]);
  }
  if (f.pool === true) { clauses.push(`${col('pool', RP)} = '1'`); }
  else if (f.pool === false) { clauses.push(`COALESCE(${col('pool', RP)}, '') <> '1'`); }
  if (f.minSqft != null) { clauses.push(`${col('livingArea', RP)} >= ?`); params.push(f.minSqft); }
  if (f.keywords) {
    clauses.push(`MATCH(${col('remarks', RP)}) AGAINST (? IN NATURAL LANGUAGE MODE)`);
    params.push(f.keywords);
  }
  return { sql: clauses.join(' AND '), params };
}

export interface SearchResult {
  rows: ListingRow[];
  total: number;
  page: number;
  limit: number;
}

export async function searchActiveListings(
  filter: SearchFilter,
  page = 1,
  limit = 10,
): Promise<SearchResult> {
  const cappedLimit = Math.min(Math.max(1, limit), MAX_RESULTS);
  const safePage = Math.max(1, page);
  const offset = (safePage - 1) * cappedLimit;
  const { sql: where, params } = buildWhere(filter);

  const countRows = await query<RowDataPacket & { n: number }>(
    `SELECT COUNT(*) AS n FROM ${RP} WHERE ${where}`,
    params,
  );
  const total = Number(countRows[0]?.n ?? 0);

  const rows = await query<RowDataPacket>(
    `${SELECT_LISTINGS} WHERE ${where} ORDER BY ${col('price', RP)} ASC LIMIT ? OFFSET ?`,
    [...params, cappedLimit, offset],
  );

  return { rows: rows.map(mapListingRow), total, page: safePage, limit: cappedLimit };
}

/** Structural signal for the orchestrator's clarification decision (Q6). */
export function searchSignal(total: number): { signal: 'empty' | 'too_many'; clarification: string } | null {
  if (total === 0) {
    return { signal: 'empty', clarification: 'No matches — want to relax the budget or widen the area?' };
  }
  if (total > TOO_MANY) {
    return { signal: 'too_many', clarification: `That's ${total} listings — narrow it down by budget, beds, or property type?` };
  }
  return null;
}

// ---- Sold comps (feeds Week 5 market stats + Week 7 comp validation) ----

const CS = 'california_sold' as const;

export interface SoldComp {
  city: string | null;
  closeDate: string | null;
  closePrice: number | null;
  sqft: number | null;
  beds: number | null;
  baths: number | null;
  dom: number | null;
}

function isoDaysAgo(months: number): string {
  const d = new Date();
  d.setMonth(d.getMonth() - months);
  return d.toISOString().slice(0, 10);
}
const isoToday = () => new Date().toISOString().slice(0, 10);

/**
 * Recent sold transactions in a city. CloseDate is a VARCHAR in ISO format, so
 * string comparison works for the range; the upper bound = today excludes the
 * known dirty future dates (e.g. year 2072). Internal use -> higher cap than the
 * user-facing ≤50.
 */
export async function getSoldComps(city: string, months = 6, limit = 500): Promise<SoldComp[]> {
  const rows = await query<RowDataPacket>(
    `SELECT
       ${col('city', CS)}        AS city,
       ${col('closeDate', CS)}   AS closeDate,
       ${col('closePrice', CS)}  AS closePrice,
       ${col('livingArea', CS)}  AS sqft,
       ${col('beds', CS)}        AS beds,
       ${col('baths', CS)}       AS baths,
       ${col('daysOnMarket', CS)} AS dom
     FROM ${CS}
     WHERE ${col('city', CS)} = ?
       AND ${col('closePrice', CS)} > 0
       AND ${col('closeDate', CS)} >= ?
       AND ${col('closeDate', CS)} <= ?
     ORDER BY ${col('closeDate', CS)} DESC
     LIMIT ?`,
    [city, isoDaysAgo(months), isoToday(), Math.min(limit, 2000)],
  );
  const num = (v: unknown) => (v == null || v === '' ? null : Number(v));
  return rows.map((r) => ({
    city: r.city ?? null,
    closeDate: r.closeDate ?? null,
    closePrice: num(r.closePrice),
    sqft: num(r.sqft),
    beds: num(r.beds),
    baths: num(r.baths),
    dom: num(r.dom),
  }));
}
