/**
 * Structured search filter — the target of NL parsing (Week 2) and the input to
 * the query layer (Week 3). Field names mirror the semantic concepts in
 * schema/columns.ts (city/beds/baths/price...).
 */

export interface SearchFilter {
  city?: string;
  /** Interpreted as ">= beds" by the query layer. */
  beds?: number;
  /** ">= baths". */
  baths?: number;
  maxPrice?: number;
  minPrice?: number;
  /** Normalized: 'condo' | 'townhouse' | 'single-family'. */
  propertyType?: string;
  pool?: boolean;
  minSqft?: number;
  /** Free-text remark search via FULLTEXT (e.g. "ocean view", "remodeled kitchen"). */
  keywords?: string;
  /** Proximity constraint (e.g. "within 30 min of downtown LA"). When present, the
   * search skill consults the maps client to filter/rank by commute — a CODE decision
   * driven by whether this slot exists, not by the LLM choosing to call a tool. */
  proximity?: {
    to: string;              // destination text (place / address)
    withinMinutes?: number;  // max commute minutes (if a time was given)
    mode?: 'driving' | 'transit' | 'walking';
  };
}

export const FILTER_KEYS: ReadonlyArray<keyof SearchFilter> = [
  'city', 'beds', 'baths', 'maxPrice', 'minPrice', 'propertyType', 'pool', 'minSqft', 'keywords', 'proximity',
];

/**
 * A patch carries per-field intent (Q12 multi-turn design):
 *   value     -> set/replace
 *   null      -> clear the field
 *   undefined -> leave unchanged
 * Modeling updates as patches (not append) prevents contradictory accumulation
 * like "3 beds AND 4 beds" when the user changes their mind.
 */
export type FilterPatch = { [K in keyof SearchFilter]?: SearchFilter[K] | null };

/** Apply a patch to a base filter (immutable). */
export function mergeFilter(base: SearchFilter, patch: FilterPatch): SearchFilter {
  const out: SearchFilter = { ...base };
  for (const key of FILTER_KEYS) {
    if (!(key in patch)) continue;
    const v = patch[key as keyof FilterPatch];
    if (v === undefined) continue;
    if (v === null) {
      delete out[key];
    } else {
      // index-safe assignment
      (out as Record<string, unknown>)[key] = v;
    }
  }
  return out;
}

/** Number of fields actually set — a structural signal for confidence. */
export function filledCount(f: SearchFilter): number {
  return FILTER_KEYS.filter((k) => f[k] !== undefined).length;
}

/**
 * Structural listing constraints only (beds/baths/price/type/sqft/pool/proximity) — excludes
 * `keywords`, which is free-text/semantic residue. Used to decide "this is a search that just
 * needs a city": a real constraint (e.g. "3 bed under 1M") qualifies, but a bare keyword blob
 * (e.g. an LLM parse hallucinating "joke") does not — that should defer to the OOD gate.
 */
export function structuralCount(f: SearchFilter): number {
  return FILTER_KEYS.filter((k) => k !== 'keywords' && k !== 'city' && f[k] !== undefined).length;
}

/** Human-readable summary for the "current filter" line shown to users (Q12 transparency). */
export function summarizeFilter(f: SearchFilter): string {
  const parts: string[] = [];
  if (f.city) parts.push(f.city);
  if (f.propertyType) parts.push(f.propertyType);
  if (f.beds !== undefined) parts.push(`${f.beds}+ bd`);
  if (f.baths !== undefined) parts.push(`${f.baths}+ ba`);
  if (f.minSqft !== undefined) parts.push(`${f.minSqft}+ sqft`);
  if (f.minPrice !== undefined) parts.push(`≥$${fmtPrice(f.minPrice)}`);
  if (f.maxPrice !== undefined) parts.push(`≤$${fmtPrice(f.maxPrice)}`);
  if (f.pool === true) parts.push('pool');
  else if (f.pool === false) parts.push('no pool');
  if (f.keywords) parts.push(`"${f.keywords}"`);
  if (f.proximity) {
    const t = f.proximity.withinMinutes ? `≤${f.proximity.withinMinutes}min ` : 'near ';
    parts.push(`${t}${f.proximity.to}`);
  }
  return parts.length ? parts.join(' · ') : '(no filters)';
}

function fmtPrice(n: number): string {
  if (n >= 1_000_000) return `${parseFloat((n / 1_000_000).toFixed(2))}M`;
  if (n >= 1_000) return `${Math.round(n / 1_000)}K`;
  return String(n);
}
