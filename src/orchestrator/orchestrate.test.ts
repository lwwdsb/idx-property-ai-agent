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
import type { LLMClient } from '../llm/client.js';
import { config } from '../config.js';
import { closePool } from '../db.js';

const noLLM: LLMClient = { available: false, async parseFilters() { return {}; } };
// a planner LLM that decomposes into per-skill sub-queries
const planLLM: LLMClient = {
  available: true,
  async parseFilters() { return {}; },
  async planSkills() { return [{ skill: 'search', query: '3 bed in Irvine' }, { skill: 'market', query: 'Irvine' }]; },
};

const calls: string[] = [];
let classifyReturn = { skill: 'unknown', score: 0.2 };
let searchReturn: Array<Record<string, unknown>> = [];
const fakeBridge: PythonBridge = {
  async classify(m) { calls.push(`classify:${m}`); return classifyReturn; },
  async rag(q) { calls.push(`rag:${q}`); return `RAG_ANSWER for "${q}"\nSources: terms.md`; },
  async recommend(id) { calls.push(`recommend:${id}`); return `RECS_FOR ${id}`; },
  async validate(l) { calls.push(`validate:${l.city}`); return JSON.stringify({ verdict: 'Priced in line with recent comparable sales.' }); },
  async search(p) { calls.push(`search:${p.text}|city=${p.city}`); return searchReturn as never; },
};
const draftStore = new InMemoryDraftStore();
const sentBox: string[] = [];
const send = async (m: { to: string }) => { sentBox.push(m.to); };
const opts = { registry: buildRegistry(fakeBridge, draftStore), bridge: fakeBridge, draftStore, send };
const OPERATOR = config.email.allowlist[0] ?? 'op';
const OUTSIDER = '+19999999999';

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
t('semantic search: soft query -> Qdrant hybrid (with hard filter)', async () => {
  searchReturn = [{ score: 0.9, address: '1 View Ln', city: 'Irvine', price: 1400000, beds: 3, type: 'SingleFamilyResidence' }];
  const r = await orchestrate('u', '在 Irvine 找有山景的工匠风老宅 150万以下', opts);
  assert.equal(r.intent, 'search');
  assert.ok(calls.some((c) => c.startsWith('search:')), 'went to semantic search');
  assert.match(r.reply, /semantic:|View Ln/);
  searchReturn = [];
});
t('pure structured query -> MySQL (no semantic call)', async () => {
  calls.length = 0;
  const r = await orchestrate('u', '在 Irvine 找 3 居室 150万以下', opts);
  assert.equal(r.intent, 'search');
  assert.ok(!calls.some((c) => c.startsWith('search:')), 'no semantic residual -> stays on MySQL');
});
t('routes recommend query -> recommend skill via bridge', async () => {
  const r = await orchestrate('u', '跟 1174456906 类似的房子', opts);
  assert.equal(r.intent, 'recommend');
  assert.match(r.reply, /RECS_FOR 1174456906/);
});
t('recommend by reference ("跟第一个类似的") resolves from last search', async () => {
  await orchestrate('refuser', '在 Irvine 找 3 居室 200万以下', opts);   // seeds lastResults
  const r = await orchestrate('refuser', '跟第一个类似的房子', opts);
  assert.equal(r.intent, 'recommend');
  assert.match(r.reply, /RECS_FOR \d+/);                                 // resolved to a real id, no raw id typed
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
t('WhatsApp "approve N" sends (deterministic command, not the LLM)', async () => {
  sentBox.length = 0;
  const draft = await orchestrate(OPERATOR, 'email the Irvine market report to buyer@example.com', opts);
  const id = draft.reply.match(/approve (\d+)/i)![1];
  const r = await orchestrate(OPERATOR, `approve ${id}`, opts);
  assert.equal(r.skill, 'email-approve');
  assert.match(r.reply, /✅ Sent/);
  assert.deepEqual(sentBox, ['buyer@example.com']);   // delivered to the right recipient
});
t('non-operator cannot approve a draft', async () => {
  const draft = await orchestrate(OPERATOR, 'email the Irvine market report to x@example.com', opts);
  const id = draft.reply.match(/approve (\d+)/i)![1];
  sentBox.length = 0;
  const r = await orchestrate(OUTSIDER, `approve ${id}`, opts);
  assert.match(r.reply, /only the operator/i);
  assert.equal(sentBox.length, 0);                     // not sent
});
t('"cancel N" cancels a pending draft', async () => {
  const draft = await orchestrate(OPERATOR, 'email the Irvine market report to y@example.com', opts);
  const id = draft.reply.match(/approve (\d+)/i)![1];
  const r = await orchestrate(OPERATOR, `cancel ${id}`, opts);
  assert.match(r.reply, /cancelled/i);
});

// ---- compound recipe (search -> validate) ----
t('multi-intent (search + market) -> planner runs both skills', async () => {
  // no-LLM planner => deterministic detected set [search, market]
  const r = await orchestrate('u', '在 Irvine 找 3 居室，再看看这个城市的行情', { ...opts, llm: noLLM });
  assert.equal(r.intent, 'compound');
  assert.match(r.reply, /Current filter|match/i);        // search part
  assert.match(r.reply, /median|sales|No recent/i);       // market part
  assert.ok((r.skill ?? '').includes('+'), 'composed multiple skills');
});
t('planner: LLM decomposes into per-skill sub-queries, runs both', async () => {
  const r = await orchestrate('u', '在 Irvine 找 3 居室，再看看行情', { ...opts, llm: planLLM });
  assert.equal(r.intent, 'compound');
  assert.ok((r.skill ?? '').includes('search') && (r.skill ?? '').includes('market'), 'both skills ran');
  assert.match(r.reply, /Current filter|match/i);
  assert.match(r.reply, /median|sales|No recent/i);
});
t('single intent does NOT trigger the planner', async () => {
  const r = await orchestrate('u', '在 Irvine 找 3 居室 200万以下', { ...opts, llm: noLLM });
  assert.equal(r.intent, 'search');                       // stays single, no compound
});
t('"days on market" question does NOT falsely plan (substring trap)', async () => {
  const r = await orchestrate('u', 'what is days on market?', { ...opts, llm: noLLM });
  assert.equal(r.intent, 'knowledge');                    // not market, not compound
});

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
