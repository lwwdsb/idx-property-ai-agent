/**
 * Week 5 market stats tests (real DB + pure median unit checks).
 * Run: npm run test:market
 */
import assert from 'node:assert/strict';
import { getMarketStats, median } from './marketStats.js';
import { closePool } from '../db.js';

type Case = { name: string; fn: () => void | Promise<void> };
const cases: Case[] = [];
const t = (name: string, fn: Case['fn']) => cases.push({ name, fn });

t('median: odd/even/empty, resists outliers', () => {
  assert.equal(median([3, 1, 2]), 2);
  assert.equal(median([1, 2, 3, 4]), 2.5);
  assert.equal(median([]), null);
  // an average would be dragged high by the outlier; median is not
  assert.equal(median([1, 1, 1, 1, 100]), 1);
});

t('getMarketStats returns sane Irvine numbers', async () => {
  const s = await getMarketStats('Irvine', 12);
  assert.ok(s.count > 30, `expected many Irvine sales, got ${s.count}`);
  assert.ok(s.medianPrice != null && s.medianPrice > 100_000, `median price ${s.medianPrice}`);
  assert.ok(s.medianPricePerSqft != null && s.medianPricePerSqft > 0);
  assert.ok(s.medianDom != null && s.medianDom >= 0);
  assert.ok(s.medianListToSoldPct != null && s.medianListToSoldPct > 50 && s.medianListToSoldPct < 200);
});

t('window bounds exclude dirty future dates', async () => {
  const s = await getMarketStats('Irvine', 12);
  assert.ok(s.window.to <= new Date().toISOString().slice(0, 10));
  for (const p of s.trend) assert.ok(p.month <= s.window.to.slice(0, 7), `future month ${p.month}`);
});

t('trend is chronological and within window length', async () => {
  const s = await getMarketStats('Irvine', 12);
  assert.ok(s.trend.length <= 12);
  for (let i = 1; i < s.trend.length; i++) {
    assert.ok(s.trend[i - 1]!.month < s.trend[i]!.month, 'months not sorted');
  }
});

t('unknown city -> zero count, no crash', async () => {
  const s = await getMarketStats('NotARealCityXYZ', 12);
  assert.equal(s.count, 0);
  assert.equal(s.medianPrice, null);
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
