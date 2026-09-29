/**
 * Multi-query tests (offline): variant generation (+ degrade) and RRF fusion.
 * Run: npx tsx src/search/multiQuery.test.ts
 */
import assert from 'node:assert/strict';
import { expandQuery, rrfFuse, translateSemantic, hasCJK } from './multiQuery.js';
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


await check('hasCJK: 只认中日韩,不误伤英文与数字', () => {
  assert.equal(hasCJK('mid-century with ADU'), false);
  assert.equal(hasCJK('3-bed 1.6M'), false);
  assert.equal(hasCJK('中古风'), true);
  assert.equal(hasCJK('mixed 中古风 style'), true);
});

await check('translateSemantic: 英文原样返回,不白花一次调用', async () => {
  let called = false;
  const llm = { available: true, async parseFilters() { return {}; },
    async chatWithTools() { called = true; return { content: 'x', toolCalls: [], raw: { role: 'assistant' as const, content: 'x' } }; } };
  assert.equal(await translateSemantic('ocean view craftsman', llm as never), 'ocean view craftsman');
  assert.equal(called, false, '英文残余不该触发任何调用');
});

await check('translateSemantic: 没有 LLM 时返回空串 —— 降级结构化而不是送噪声', async () => {
  // 空串是给调用方的信号:走纯结构化搜索。宁可丢掉特征,也不要拿中文去搜英文语料 ——
  // 后者不是"效果差一点",而是把一次语义检索变成噪声,还把路由带错(非空残余=走 hybrid)。
  assert.equal(await translateSemantic('中古风 学区好', undefined), '');
});

await check('translateSemantic: 翻译结果仍含中文 -> 按失败处理', async () => {
  const llm = { available: true, async parseFilters() { return {}; },
    async chatWithTools() { return { content: '中古风 style', toolCalls: [], raw: { role: 'assistant' as const, content: '' } }; } };
  assert.equal(await translateSemantic('中古风', llm as never), '', '半成品不许送进索引');
});

console.log(`\n${pass}/${pass + fail} passed`);
if (fail) process.exit(1);
