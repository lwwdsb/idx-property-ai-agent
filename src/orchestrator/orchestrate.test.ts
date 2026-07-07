/**
 * Week 9 orchestrator tests. Routing / recipes / clarify verified offline; the
 * Python skills (recommend, knowledge, validate) use a fake bridge, so no
 * Qdrant/Python needed. search + market execute against the real DB.
 *
 * Run: npm run test:orch
 */
import assert from 'node:assert/strict';
import { orchestrate } from './orchestrate.js';
import { buildRegistry } from './skills.js';
import type { PythonBridge } from './bridge.js';
import { InMemoryDraftStore } from '../email/drafts.js';
import { config } from '../config.js';
import { closePool } from '../db.js';

const calls: string[] = [];
let classifyReturn = { skill: 'unknown', score: 0.2 };
const fakeBridge: PythonBridge = {
  async classify(m) { calls.push(`classify:${m}`); return classifyReturn; },
  async rag(q) { calls.push(`rag:${q}`); return `RAG_ANSWER for "${q}"\nSources: terms.md`; },
  async recommend(id) { calls.push(`recommend:${id}`); return `RECS_FOR ${id}`; },
  async validate(l) { calls.push(`validate:${l.city}`); return JSON.stringify({ verdict: 'Priced in line with recent comparable sales.' }); },
};
const opts = { registry: buildRegistry(fakeBridge, new InMemoryDraftStore()), bridge: fakeBridge };
const OPERATOR = config.email.allowlist[0] ?? 'op';

type Case = { name: string; fn: () => Promise<void> };
const cases: Case[] = [];
const t = (name: string, fn: Case['fn']) => cases.push({ name, fn });

// ---- routing ----
t('routes a listing query -> search', async () => {
  const r = await orchestrate('u', '在 Irvine 找 3 居室 150万以下', opts);
  assert.equal(r.intent, 'search');
  assert.match(r.reply, /Current filter|match/i);
});
t('routes market query -> market (city-first)', async () => {
  const r = await orchestrate('u', 'Irvine 行情怎么样', opts);
  assert.equal(r.intent, 'market');
  assert.match(r.reply, /median|sales/i);
});
t('routes recommend query -> recommend skill via bridge', async () => {
  const r = await orchestrate('u', '跟 1174456906 类似的房子', opts);
  assert.equal(r.intent, 'recommend');
  assert.match(r.reply, /RECS_FOR 1174456906/);
});
t('routes knowledge question -> RAG via bridge', async () => {
  const r = await orchestrate('u', 'what does DOM mean?', opts);
  assert.equal(r.intent, 'knowledge');
  assert.match(r.reply, /RAG_ANSWER/);
});
t('"days on market" definition is knowledge, not market (substring trap)', async () => {
  const r = await orchestrate('u', 'what is days on market?', opts);
  assert.equal(r.intent, 'knowledge');
});

// ---- email drafting (routes, drafts pending, never sends) ----
t('email intent -> drafts a pending email, does not send', async () => {
  const r = await orchestrate(OPERATOR, 'email the Irvine market report to client@example.com', opts);
  assert.equal(r.intent, 'email');
  assert.match(r.reply, /pending your approval/i);
  assert.match(r.reply, /client@example.com/);
});

// ---- compound recipe (search -> validate) ----
t('compound: search + price validate chained', async () => {
  const r = await orchestrate('u', '帮我在 Irvine 找 3 居室，顺便看看贵不贵', opts);
  assert.equal(r.intent, 'compound');
  assert.match(r.reply, /price check/i);
  assert.match(r.reply, /in line with recent comparable/i); // from fake validate
  assert.ok(calls.some((c) => c.startsWith('validate:')), 'validate was invoked');
});

// ---- embedding fallback when regex is unsure ----
t('regex-miss phrasing -> routed via embedding classifier', async () => {
  classifyReturn = { skill: 'market', score: 0.9 };   // service would say "market"
  const r = await orchestrate('u', 'give me the lay of the land please', opts); // no regex keyword
  assert.equal(r.intent, 'market');
  assert.ok(calls.some((c) => c.startsWith('classify:')), 'embedding classifier consulted');
  classifyReturn = { skill: 'unknown', score: 0.2 };  // reset
});
t('low embedding score -> still clarify (floor)', async () => {
  classifyReturn = { skill: 'market', score: 0.3 };   // below threshold
  const r = await orchestrate('u', 'zzz qqq', opts);
  assert.equal(r.intent, 'unknown');
  classifyReturn = { skill: 'unknown', score: 0.2 };
});

// ---- clarify / fallback ----
t('missing city -> clarify (no skill run)', async () => {
  const r = await orchestrate('u', '3 bedroom under 1M', opts);
  assert.match(r.reply, /city/i);
});
t('gibberish -> unknown clarify', async () => {
  const r = await orchestrate('u', 'hello there friend', opts);
  assert.equal(r.intent, 'unknown');
  assert.match(r.reply, /search listings|market stats|recommend|questions/i);
});
t('recommend without an id -> asks for id', async () => {
  const r = await orchestrate('u', 'show me similar homes', opts);
  assert.equal(r.intent, 'recommend');
  assert.match(r.reply, /id|MLS/i);
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
