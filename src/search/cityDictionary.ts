/**
 * Known-city dictionary: the set of cities we actually have listings for.
 *
 * Used as a cheap, deterministic OUTPUT sanity check on parsing: a regex/LLM that
 * "confidently" extracts a city that isn't real (e.g. "USC", a mis-capture) is a
 * false positive our under-parse confidence signal can't catch. Validating the
 * extracted city against this set turns "confidently wrong" into a detectable
 * signal that triggers LLM re-analysis or a clarification.
 *
 * Loaded once and cached. Fails OPEN (accepts the city) if the DB can't be reached,
 * so a hiccup never blocks a search (乙).
 */
import type { RowDataPacket } from 'mysql2/promise';
import { query } from '../db.js';
import { col } from '../../schema/columns.js';

let citySet: Set<string> | null = null;

/** Distinct, lowercased cities present in the active-listings table (cached). */
export async function knownCitySet(): Promise<Set<string>> {
  if (citySet) return citySet;
  const rp = 'rets_property' as const;
  const rows = await query<RowDataPacket & { c: string }>(
    `SELECT DISTINCT ${col('city', rp)} AS c FROM ${rp} WHERE ${col('city', rp)} <> ''`,
  );
  citySet = new Set(rows.map((r) => String(r.c).trim().toLowerCase()).filter(Boolean));
  return citySet;
}

/** True if we can serve this city. Fails OPEN on any error (don't block on a hiccup). */
export async function isKnownCity(city: string): Promise<boolean> {
  try {
    return (await knownCitySet()).has(city.trim().toLowerCase());
  } catch {
    return true;
  }
}
