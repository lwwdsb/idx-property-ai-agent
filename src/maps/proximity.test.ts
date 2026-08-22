/**
 * Proximity filtering/ranking test (offline: a FAKE maps client is injected, no key /
 * network). Verifies applyProximity annotates commute, filters by withinMinutes, sorts
 * by commute, and degrades gracefully.
 *   npm run test:proximity
 */
import assert from 'node:assert/strict';
import { applyProximity } from '../orchestrator/skills.js';
import type { MapsClient } from './mapsClient.js';
import type { ListingRow } from '../search/listingRow.js';
import { closePool } from '../db.js';

function row(id: number, lat: number | null, lng: number | null): ListingRow {
  return { id, listingId: null, mls: `M${id}`, address: `${id} St`, city: 'LA', zip: null,
    type: 'Condominium', beds: 3, baths: 2, sqft: 1000, price: 900000, photoCount: null,
    yearBuilt: null, pool: false, lat, lng };
}

// fake client: commute minutes = the listing id (so id order = commute order)
const fake = (commute: Record<number, number | null>, available = true): MapsClient => ({
  available,
  async geocode() { return { lat: 34, lng: -118 }; },
  async commuteMinutes(origins) { return origins.map((o) => ({ id: o.id, minutes: commute[o.id] ?? null })); },
});

let pass = 0, fail = 0;
async function t(name: string, fn: () => Promise<void>) {
  try { await fn(); pass++; console.log('✓', name); }
  catch (e) { fail++; console.error('✗', name, '\n   ', (e as Error).message.split('\n')[0]); }
}

await t('annotates commute + sorts by commute ascending', async () => {
  const rows = [row(1, 34, -118), row(2, 34, -118), row(3, 34, -118)];
  const out = await applyProximity(rows, { to: 'downtown' }, fake({ 1: 40, 2: 10, 3: 25 }));
  assert.deepEqual(out.map((r) => r.id), [2, 3, 1]);           // sorted by minutes
  assert.equal(out[0]!.commuteMinutes, 10);
});

await t('filters by withinMinutes', async () => {
  const rows = [row(1, 34, -118), row(2, 34, -118), row(3, 34, -118)];
  const out = await applyProximity(rows, { to: 'downtown', withinMinutes: 30 }, fake({ 1: 40, 2: 10, 3: 25 }));
  assert.deepEqual(out.map((r) => r.id), [2, 3]);             // 40min dropped
});

await t('no key / unavailable -> returns original rows unchanged (乙)', async () => {
  const rows = [row(1, 34, -118)];
  const out = await applyProximity(rows, { to: 'x' }, fake({}, false));
  assert.deepEqual(out, rows);
});

await t('listings without coords -> skipped (returns original)', async () => {
  const rows = [row(1, null, null), row(2, null, null)];
  const out = await applyProximity(rows, { to: 'x' }, fake({ 1: 5 }));
  assert.deepEqual(out, rows);                                // none geocodable -> unchanged
});

await closePool();
console.log(`\n${pass}/${pass + fail} passed`);
if (fail) process.exitCode = 1;
