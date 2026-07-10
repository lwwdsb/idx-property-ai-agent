/**
 * Week 2 eval suite — runs offline (no LLM key needed).
 * Covers EN/中文, fuzzy phrasing, missing fields, money parsing, patch-merge,
 * structural clarification, and the LLM-escalation path (via a mock client).
 *
 * Run: npm test
 */
import assert from 'node:assert/strict';
import { parseMoney, regexParse } from './regexParse.js';
import { parseQuery } from './parseQuery.js';
import { mergeFilter, summarizeFilter } from './filters.js';
import type { LLMClient } from '../llm/client.js';

type Case = { name: string; fn: () => void | Promise<void> };
const cases: Case[] = [];
const t = (name: string, fn: Case['fn']) => cases.push({ name, fn });

const mockLLM = (patch: Record<string, unknown>): LLMClient => ({
  available: true,
  async parseFilters() { return patch as never; },
});
const brokenLLM: LLMClient = { available: true, async parseFilters() { throw new Error('boom'); } };

// ---- money parsing ----
t('parseMoney: suffixes & formats', () => {
  assert.equal(parseMoney('1.5M'), 1_500_000);
  assert.equal(parseMoney('900k'), 900_000);
  assert.equal(parseMoney('$1,200,000'), 1_200_000);
  assert.equal(parseMoney('1.5 million'.replace(' ', '')), 1_500_000);
  assert.equal(parseMoney('150万'), 1_500_000);
});

// ---- regex extraction (EN) ----
t('EN: full query', () => {
  const f = regexParse('in Irvine find 3 bedroom homes with a pool under 1.5 million');
  assert.deepEqual(f, { city: 'Irvine', beds: 3, pool: true, maxPrice: 1_500_000 });
});
t('EN: beds/baths + city + max $k', () => {
  const f = regexParse('3 bed 2 bath in San Diego under $900k');
  assert.equal(f.city, 'San Diego'); assert.equal(f.beds, 3); assert.equal(f.baths, 2);
  assert.equal(f.maxPrice, 900_000);
});
t('EN: condo + price range', () => {
  const f = regexParse('condo in Los Angeles between 500k and 800k');
  assert.equal(f.propertyType, 'condo'); assert.equal(f.minPrice, 500_000); assert.equal(f.maxPrice, 800_000);
  assert.equal(f.city, 'Los Angeles');
});
t('EN: townhouse + over (minPrice)', () => {
  const f = regexParse('townhouse over 1m near Pasadena');
  assert.equal(f.propertyType, 'townhouse'); assert.equal(f.minPrice, 1_000_000); assert.equal(f.city, 'Pasadena');
});
t('EN: sqft is NOT parsed as price', () => {
  const f = regexParse('single family home with at least 2000 sqft in Irvine');
  assert.equal(f.propertyType, 'single-family'); assert.equal(f.minSqft, 2000);
  assert.equal(f.minPrice, undefined); assert.equal(f.maxPrice, undefined);
  assert.equal(f.city, 'Irvine');
});
t('EN: compact 4br 3ba', () => {
  const f = regexParse('4br 3ba in Tustin');
  assert.equal(f.beds, 4); assert.equal(f.baths, 3); assert.equal(f.city, 'Tustin');
});

// ---- regex extraction (中文) ----
t('中文: full query', () => {
  const f = regexParse('在 Irvine 找 3 居室带泳池、150万以下公寓');
  assert.equal(f.city, 'Irvine'); assert.equal(f.beds, 3); assert.equal(f.pool, true);
  assert.equal(f.maxPrice, 1_500_000); assert.equal(f.propertyType, 'condo');
});
t('中文: 以上 = minPrice', () => {
  const f = regexParse('San Jose 200万以上的房子');
  assert.equal(f.minPrice, 2_000_000);
});

// ---- pool is tri-state (negation must not be a false positive) ----
t('EN: "without a pool" -> pool false', () => {
  const f = regexParse('3 bed in Irvine without a pool');
  assert.equal(f.pool, false); assert.equal(f.city, 'Irvine'); assert.equal(f.beds, 3);
});
t('中文: 不要泳池 -> false, 带泳池 -> true', () => {
  assert.equal(regexParse('在 Irvine 找不要泳池的房子').pool, false);
  assert.equal(regexParse('在 Irvine 找带泳池的房子').pool, true);
});
t('summarizeFilter shows "no pool" for false', () => {
  assert.equal(summarizeFilter({ city: 'Irvine', pool: false }), 'Irvine · no pool');
});

// ---- city validation (output sanity check) + risk-word escalation ----
const knownCities = (...set: string[]) => (c: string) =>
  set.map((s) => s.toLowerCase()).includes(c.trim().toLowerCase());

t('city validation: rejects a non-real city, keeps constraints, clarifies', async () => {
  const r = await parseQuery('在 USC 附近找 3 居室', { isKnownCity: knownCities('irvine', 'san diego') });
  assert.equal(r.confidence, 'low');
  assert.match(r.clarification!, /USC/);        // echoes what it couldn't serve
  assert.equal(r.filter.city, undefined);
  assert.equal(r.filter.beds, 3);               // other constraints preserved
});
t('city validation: accepts a real city', async () => {
  const r = await parseQuery('3 bed in Irvine', { isKnownCity: knownCities('irvine') });
  assert.equal(r.confidence, 'high'); assert.equal(r.filter.city, 'Irvine');
});
t('city validation: LLM rescues a city regex missed, then validated', async () => {
  const r = await parseQuery('homes under 1m near the USC campus', {
    isKnownCity: knownCities('los angeles'), llm: mockLLM({ city: 'Los Angeles' }),
  });
  assert.equal(r.confidence, 'high'); assert.equal(r.filter.city, 'Los Angeles');
});
t('risk word (negation) escalates to LLM even when regex already parsed a city', async () => {
  let called = false;
  const spy: LLMClient = { available: true, async parseFilters() { called = true; return {} as never; } };
  await parseQuery('3 bed in Irvine without a pool', { isKnownCity: knownCities('irvine'), llm: spy });
  assert.equal(called, true);                   // ② negation -> double-check via LLM
});

// ---- parseQuery: confidence / clarification (structural) ----
t('parseQuery: city present -> high', async () => {
  const r = await parseQuery('3 bed in Irvine under 1M');
  assert.equal(r.confidence, 'high'); assert.equal(r.filter.city, 'Irvine'); assert.equal(r.clarification, undefined);
});
t('parseQuery: missing city -> low + ask city, keeps constraints', async () => {
  const r = await parseQuery('3 bedroom under 1M');
  assert.equal(r.confidence, 'low'); assert.match(r.clarification!, /city/i);
  assert.equal(r.filter.beds, 3); assert.equal(r.filter.maxPrice, 1_000_000);
});
t('parseQuery: empty -> low + ask anything', async () => {
  const r = await parseQuery('hello there');
  assert.equal(r.confidence, 'low'); assert.match(r.clarification!, /Tell me a city/i);
});

// ---- parseQuery: LLM escalation only when city missing ----
t('parseQuery: escalates to LLM when regex misses city', async () => {
  const r = await parseQuery('homes under 1m in the bay area', { llm: mockLLM({ city: 'San Francisco' }) });
  assert.equal(r.source, 'llm'); assert.equal(r.confidence, 'high'); assert.equal(r.filter.city, 'San Francisco');
  assert.equal(r.filter.maxPrice, 1_000_000); // regex constraint preserved
});
t('parseQuery: does NOT call LLM when regex already has city', async () => {
  let called = false;
  const spy: LLMClient = { available: true, async parseFilters() { called = true; return {} as never; } };
  await parseQuery('2 bed in Irvine', { llm: spy });
  assert.equal(called, false); // Q13 cost budget: skip LLM on the cheap path
});
t('parseQuery: LLM failure degrades to regex result (乙)', async () => {
  const r = await parseQuery('3 bed under 1M in nowhere-lowercase', { llm: brokenLLM });
  assert.equal(r.confidence, 'low'); assert.equal(r.filter.beds, 3); // no throw
});

// ---- multi-turn patch merge (Q12) ----
t('merge: "actually 4 beds" overrides 3 (replace, not accumulate)', () => {
  const base = { city: 'Irvine', beds: 3 };
  const next = regexParse('actually 4 bedrooms');
  const merged = mergeFilter(base, next);
  assert.equal(merged.beds, 4); assert.equal(merged.city, 'Irvine');
});
t('merge: null clears a field', () => {
  assert.deepEqual(mergeFilter({ beds: 3, city: 'Irvine' }, { beds: null }), { city: 'Irvine' });
});

// ---- summary line (Q12 transparency) ----
t('summarizeFilter', () => {
  assert.equal(summarizeFilter({ city: 'Irvine', beds: 3, maxPrice: 1_500_000, pool: true }),
    'Irvine · 3+ bd · ≤$1.5M · pool');
  assert.equal(summarizeFilter({}), '(no filters)');
});

// ---- runner ----
(async () => {
  let pass = 0, fail = 0;
  for (const c of cases) {
    try { await c.fn(); pass++; console.log('✓', c.name); }
    catch (e) { fail++; console.error('✗', c.name, '\n   ', (e as Error).message.split('\n')[0]); }
  }
  console.log(`\n${pass}/${pass + fail} passed`);
  if (fail) process.exitCode = 1;
})();
