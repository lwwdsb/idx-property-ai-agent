/**
 * ListingRow — the unified DTO returned by the query layer and consumed by
 * search / recommendation / WhatsApp / email (one shape, no per-channel rewrites).
 *
 * Grounding (Q5): every user-facing fact (address, price, beds...) comes from a
 * real DB row via this DTO and `formatListingCard`. The LLM never authors these
 * values — it only adds language around them.
 */
import type { RowDataPacket } from 'mysql2/promise';

export interface ListingRow {
  id: number;
  listingId: string | null;
  mls: string | null;
  address: string | null;
  city: string | null;
  zip: string | null;
  type: string | null;
  beds: number | null;
  baths: number | null;
  sqft: number | null;
  price: number | null;
  photoCount: number | null;
  yearBuilt: number | null;
  pool: boolean;
  lat: number | null;
  lng: number | null;
  /** Commute minutes to a requested destination — filled by the search skill only
   * when a proximity constraint was parsed (else undefined). */
  commuteMinutes?: number | null;
}

/** SELECT aliases (physical column -> DTO field) come from the field dictionary,
 * so mapping stays a single source of truth. Built in searchListings.ts. */
export function mapListingRow(r: RowDataPacket): ListingRow {
  const num = (v: unknown): number | null =>
    v == null || v === '' || Number.isNaN(Number(v)) ? null : Number(v);
  return {
    id: Number(r.id),
    listingId: r.listingId ?? null,
    mls: r.mls ?? null,
    address: r.address ?? null,
    city: r.city ?? null,
    zip: r.zip ?? null,
    type: r.type ?? null,
    beds: num(r.beds),
    baths: num(r.baths),
    sqft: num(r.sqft),
    price: num(r.price),
    photoCount: num(r.photoCount),
    yearBuilt: num(r.yearBuilt),
    pool: String(r.pool ?? '') === '1',
    lat: num(r.lat),
    lng: num(r.lng),
  };
}

const money = (n: number | null): string =>
  n == null ? 'N/A' : `$${n.toLocaleString('en-US')}`;

/** Compact WhatsApp/email-friendly card. Facts only, all from the row. */
export function formatListingCard(row: ListingRow): string {
  const head = [row.address, row.city].filter(Boolean).join(', ') || `Listing ${row.mls ?? row.id}`;
  const specs = [
    row.beds != null ? `${row.beds} bd` : null,
    row.baths != null ? `${row.baths} ba` : null,
    row.sqft != null ? `${row.sqft.toLocaleString('en-US')} sqft` : null,
  ].filter(Boolean).join(' · ');
  const tags = [
    row.type ?? null,
    row.yearBuilt != null ? `built ${row.yearBuilt}` : null,
    row.pool ? 'pool' : null,
    row.photoCount != null ? `${row.photoCount} photos` : null,
  ].filter(Boolean).join(' · ');
  return [
    `🏡 ${head} — ${money(row.price)}`,
    specs && `   ${specs}`,
    tags && `   ${tags}`,
    row.mls && `   MLS# ${row.mls}`,
  ].filter(Boolean).join('\n');
}
