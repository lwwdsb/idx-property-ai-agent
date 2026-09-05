/**
 * Multi-query tests (offline): variant generation (+ degrade) and RRF fusion.
 * Run: npx tsx src/search/multiQuery.test.ts
 */
import assert from 'node:assert/strict';
import { expandQuery, rrfFuse } from './multiQuery.js';
import type { LLMClient, ChatMessage } from '../llm/client.js';

let pass = 0, fail = 0;
async function check(name: string, fn: () => Promise<void> | void) {
  try { await fn(); pass++; console.log('✓', name); }
  catch (e) { fail++; console.log('✗', name, (e as Error).message); }
}

const fakeLLM = (reply: string, throws = false): LLMClient => ({
  available: true,
  async parseFilters() { return {}; },
  async chatWithTools() {
    if (throws) throw new Error('ETIMEDOUT');
    return { content: reply, toolCalls: [], raw: { role: 'assistant', content: '' } as ChatMessage };
  },
});

await check('expandQuery: original + parsed variants (deduped, capped)', async () => {
  const r = await expandQuery('ocean view craftsman',
    fakeLLM('["coastal arts-and-crafts home", "sea-facing bungalow"]'), 3);
  assert.equal(r[0], 'ocean view craftsman');
  assert.ok(r.includes('coastal arts-and-crafts home') && r.includes('sea-facing bungalow'));
  assert.equal(r.length, 3);
});
await check('expandQuery: LLM down -> original only (degrade)', async () => {
  assert.deepEqual(await expandQuery('ocean view', fakeLLM('', true), 3), ['ocean view']);
});
await check('expandQuery: no llm -> original only', async () => {
  assert.deepEqual(await expandQuery('ocean view', undefined, 3), ['ocean view']);
});
await check('rrfFuse: items ranked high across lists win; union deduped', () => {
  const A = [{ id: 1 }, { id: 2 }, { id: 3 }];
  const B = [{ id: 2 }, { id: 1 }, { id: 4 }];
  const fused = rrfFuse([A, B], (x) => x.id);
  assert.deepEqual(fused.slice(0, 2).map((x) => x.id).sort(), [1, 2]);   // 1 & 2 top (high in both)
  assert.equal(fused.length, 4);                                          // deduped union
});
await check('rrfFuse: same id across lists is deduped', () => {
  assert.equal(rrfFuse([[{ id: 1 }], [{ id: 1 }], [{ id: 2 }]], (x) => x.id).length, 2);
});

console.log(`\n${pass}/${pass + fail} passed`);
if (fail) process.exit(1);
