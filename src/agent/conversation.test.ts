/**
 * Week 4 multi-turn agent tests (uses the real DB for the results path).
 * Run: npm run test:agent
 */
import assert from 'node:assert/strict';
import { handleSearchTurn, isReset } from './conversation.js';
import { InMemorySessionStore } from './session.js';
import { closePool } from '../db.js';

type Case = { name: string; fn: () => void | Promise<void> };
const cases: Case[] = [];
const t = (name: string, fn: Case['fn']) => cases.push({ name, fn });

t('isReset detects EN/中文 reset phrases', () => {
  assert.equal(isReset('start over'), true);
  assert.equal(isReset('重新找'), true);
  assert.equal(isReset('换一个'), true);
  assert.equal(isReset('3 bed in Irvine'), false);
});

t('in-memory store honors TTL (expired entries return null)', async () => {
  const store = new InMemorySessionStore(-1); // any age exceeds a negative TTL
  await store.set('u', { filter: { city: 'Irvine' }, step: 1, updatedAt: Date.now() });
  assert.equal(await store.get('u'), null);
  // and a live store returns the session
  const live = new InMemorySessionStore(60_000);
  await live.set('u', { filter: { city: 'Irvine' }, step: 1, updatedAt: Date.now() });
  assert.equal((await live.get('u'))?.filter.city, 'Irvine');
});

t('missing city -> clarify, keeps constraints (no DB hit)', async () => {
  const store = new InMemorySessionStore();
  const r = await handleSearchTurn('u', '3 bedroom under 1M', { store });
  assert.equal(r.kind, 'clarify');
  assert.match(r.reply, /city/i);
  assert.equal(r.filter.beds, 3);
  assert.equal(r.filter.maxPrice, 1_000_000);
});

t('clarify then provide city -> merges and searches', async () => {
  const store = new InMemorySessionStore();
  const r1 = await handleSearchTurn('u', '3 bedroom under 2M', { store });
  assert.equal(r1.kind, 'clarify');
  const r2 = await handleSearchTurn('u', 'in Irvine', { store });
  assert.notEqual(r2.kind, 'clarify');
  assert.equal(r2.filter.city, 'Irvine');
  assert.equal(r2.filter.beds, 3);          // carried over from turn 1
  assert.equal(r2.filter.maxPrice, 2_000_000);
});

t('multi-turn refine: override beds + add pool, keep city', async () => {
  const store = new InMemorySessionStore();
  await handleSearchTurn('u', '3 bed in Irvine under 3M', { store });
  const r2 = await handleSearchTurn('u', 'actually 4 beds', { store });
  assert.equal(r2.filter.beds, 4);          // replaced, not accumulated
  assert.equal(r2.filter.city, 'Irvine');   // retained
  const r3 = await handleSearchTurn('u', 'with a pool', { store });
  assert.equal(r3.filter.pool, true);
  assert.equal(r3.filter.beds, 4);
  assert.equal(r3.filter.city, 'Irvine');
});

t('reset wipes accumulated filter', async () => {
  const store = new InMemorySessionStore();
  await handleSearchTurn('u', '3 bed in Irvine under 2M', { store });
  const r = await handleSearchTurn('u', 'start over', { store });
  assert.equal(r.kind, 'reset');
  assert.deepEqual(r.filter, {});
  const next = await handleSearchTurn('u', 'condos in San Diego under 800k', { store });
  assert.equal(next.filter.city, 'San Diego');
  assert.equal(next.filter.beds, undefined); // old beds gone
});

t('results path returns cards + current-filter line', async () => {
  const store = new InMemorySessionStore();
  const r = await handleSearchTurn('u', '3 bed in Irvine with a pool under 3M', { store });
  assert.ok(['results', 'too_many'].includes(r.kind));
  assert.match(r.reply, /Current filter:/);
  assert.ok((r.rows?.length ?? 0) > 0);
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
