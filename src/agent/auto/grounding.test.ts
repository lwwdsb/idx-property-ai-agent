/**
 * Grounding gate tests (offline; a fake LLM is injected — no real calls).
 * Verifies the runtime gate's three paths: LLM-rewrite fixes it, deterministic strip
 * as fallback, and a grounded reply passes through untouched (zero LLM cost).
 * Run: npx tsx src/agent/auto/grounding.test.ts
 */
import assert from 'node:assert/strict';
import { checkGrounding, observationBlob, stripUngrounded, groundFinal } from './grounding.js';
import type { ChatMessage, LLMClient } from '../../llm/client.js';

let pass = 0, fail = 0;
async function check(name: string, fn: () => void | Promise<void>) {
  try { await fn(); pass++; console.log('✓', name); }
  catch (e) { fail++; console.log('✗', name, (e as Error).message); }
}

const fakeLLM = (reply: string, onCall?: () => void): LLMClient => ({
  available: true,
  async parseFilters() { return {}; },
  async chatWithTools() { onCall?.(); return { content: reply, toolCalls: [], raw: {} as ChatMessage }; },
});

await check('checkGrounding: id present -> grounded', () => {
  assert.equal(checkGrounding('see MLS# 1157867537', 'listing 1157867537 $1M').ungrounded.length, 0);
});
await check('checkGrounding: id absent -> ungrounded', () => {
  assert.deepEqual(checkGrounding('MLS# 9999999999', 'listing 1157867537').ungrounded, ['9999999999']);
});
await check('checkGrounding: no ids -> nothing to check', () => {
  assert.equal(checkGrounding('median is $1.5M', 'blah').idCount, 0);
});
await check('observationBlob: joins only tool messages', () => {
  const blob = observationBlob([
    { role: 'tool', content: 'A 111111' }, { role: 'assistant', content: 'B' }, { role: 'tool', content: 'C 222222' },
  ] as ChatMessage[]);
  assert.ok(blob.includes('111111') && blob.includes('222222') && !blob.includes('B'));
});
await check('stripUngrounded: marks ids [unverified]', () => {
  assert.equal(stripUngrounded('id 1234567', ['1234567']), 'id [unverified]');
});

await check('groundFinal: LLM rewrite removes the hallucinated id', async () => {
  const msgs = [{ role: 'tool', content: 'real listing 1157867537' }] as ChatMessage[];
  const out = await groundFinal('fake 9999999999 and real 1157867537', msgs, fakeLLM('the real one is 1157867537'));
  assert.ok(!out.includes('9999999999'));
});
await check('groundFinal: strip fallback when LLM keeps hallucinating', async () => {
  const msgs = [{ role: 'tool', content: 'real 1157867537' }] as ChatMessage[];
  const out = await groundFinal('cite 9999999999', msgs, fakeLLM('still cite 9999999999'));
  assert.ok(out.includes('[unverified]') && !out.includes('9999999999'));
});
await check('groundFinal: grounded reply passes through, no LLM call', async () => {
  const msgs = [{ role: 'tool', content: 'real 1157867537' }] as ChatMessage[];
  let called = false;
  const out = await groundFinal('the one is 1157867537', msgs, fakeLLM('X', () => { called = true; }));
  assert.equal(out, 'the one is 1157867537');
  assert.equal(called, false);
});

console.log(`\n${pass}/${pass + fail} passed`);
if (fail) process.exit(1);
