/**
 * Week 3 integration tests — run against the real local idx_exchange DB.
 * Run: npm run test:db   (requires MySQL up + tables imported)
 */
import assert from 'node:assert/strict';
import { searchActiveListings, searchSignal, getSoldComps, MAX_RESULTS } from './searchListings.js';
import { formatListingCard } from './listingRow.js';
import { closePool } from '../db.js';

type Case = { name: string; fn: () => void | Promise<void> };
const cases: Case[] = [];
const t = (name: string, fn: Case['fn']) => cases.push({ name, fn });

t('city + beds filter returns matching active listings', async () => {
  const { rows, total } = await searchActiveListings({ city: 'Irvine', beds: 3 });
  assert.ok(rows.length > 0, 'expected some Irvine 3bd listings');
  assert.ok(total >= rows.length);
  for (const r of rows) {
    assert.equal((r.city ?? '').toLowerCase(), 'irvine');
    assert.ok(r.beds != null && r.beds >= 3, `beds ${r.beds} should be >=3`);
  }
});

t('maxPrice is respected', async () => {
  const { rows } = await searchActiveListings({ city: 'Irvine', maxPrice: 1_000_000 });
  for (const r of rows) assert.ok(r.price != null && r.price <= 1_000_000, `price ${r.price}`);
});

t('pool filter returns only pool listings', async () => {
  const { rows } = await searchActiveListings({ city: 'Irvine', pool: true });
  assert.ok(rows.length > 0);
  for (const r of rows) assert.equal(r.pool, true);
});

t('propertyType maps condo -> Condominium', async () => {
  const { rows } = await searchActiveListings({ city: 'Irvine', propertyType: 'condo' });
  assert.ok(rows.length > 0);
  for (const r of rows) assert.equal(r.type, 'Condominium');
});

t('case-insensitive city match', async () => {
  const { total } = await searchActiveListings({ city: 'irvine' });
  assert.ok(total > 0, 'lowercase city should still match');
});

t('result cap enforced at <=50 regardless of requested limit', async () => {
  const { rows } = await searchActiveListings({ city: 'Irvine' }, 1, 999);
  assert.ok(rows.length <= MAX_RESULTS, `got ${rows.length}`);
});

t('FULLTEXT keyword search works', async () => {
  const { rows, total } = await searchActiveListings({ keywords: 'pool spa' }, 1, 5);
  assert.ok(total > 0 && rows.length > 0, 'expected FULLTEXT matches for "pool spa"');
});

t('searchSignal: empty / too_many / ok', () => {
  assert.equal(searchSignal(0)!.signal, 'empty');
  assert.equal(searchSignal(5000)!.signal, 'too_many');
  assert.equal(searchSignal(40), null);
});

t('getSoldComps returns recent positive-price sales for a city', async () => {
  const comps = await getSoldComps('San Diego', 6);
  assert.ok(comps.length > 0, 'expected San Diego sold comps');
  const today = new Date().toISOString().slice(0, 10);
  for (const c of comps.slice(0, 50)) {
    assert.equal((c.city ?? '').toLowerCase(), 'san diego');
    assert.ok(c.closePrice != null && c.closePrice > 0);
    assert.ok(c.closeDate != null && c.closeDate <= today, `future date ${c.closeDate}`);
  }
});

t('formatListingCard renders facts from the row', async () => {
  const { rows } = await searchActiveListings({ city: 'Irvine', beds: 3 }, 1, 1);
  const card = formatListingCard(rows[0]!);
  assert.match(card, /🏡/);
  assert.match(card, /\$/); // a price was rendered
});

(async () => {
  let pass = 0, fail = 0;
  for (const c of cases) {
    try { await c.fn(); pass++; console.log('✓', c.name); }
    catch (e) { fail++; console.error('✗', c.name, '\n   ', (e as Error).message.split('\n')[0]); }
  }
  await closePool();
  console.log(`\n${pass}/${pass + fail} passed`);
  if (fail) process.exitCode = 1;
})();
