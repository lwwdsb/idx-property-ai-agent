/**
 * Auto loop resilience test (offline): when the LLM fails even after retry+breaker,
 * runAgent must NOT throw — it checkpoints and returns a graceful 'error' result.
 * Run: npx tsx src/agent/auto/loop.test.ts
 */
import assert from 'node:assert/strict';
import { runAgent, retryAgentRun } from './loop.js';
import { InMemoryAgentRunStore } from './runStore.js';
import { buildRegistry } from '../../orchestrator/skills.js';
import { pythonBridge } from '../../orchestrator/bridge.js';
import { InMemoryDraftStore } from '../../email/drafts.js';
import type { LLMClient } from '../../llm/client.js';

let pass = 0, fail = 0;
async function check(name: string, fn: () => Promise<void>) {
  try { await fn(); pass++; console.log('✓', name); }
  catch (e) { fail++; console.log('✗', name, (e as Error).message); }
}

// simulates the LLM being down AFTER client-level retry+breaker already gave up
const failingLLM: LLMClient = {
  available: true,
  async parseFilters() { return {}; },
  async chatWithTools() { throw new Error('ETIMEDOUT'); },
};

await check('LLM failure -> graceful abort (no throw) + checkpoint saved', async () => {
  const store = new InMemoryAgentRunStore();
  const registry = buildRegistry(pythonBridge, new InMemoryDraftStore());
  const r = await runAgent('find homes in Irvine', { userId: 'u', registry, llm: failingLLM, store, maxSteps: 3 });
  assert.equal(r.stopReason, 'error');                     // did NOT throw
  assert.match(r.reply, /couldn't finish|unavailable|busy/i);
  const run = await store.get(r.runId!);                   // state checkpointed
  assert.ok(run && run.state.task === 'find homes in Irvine');
});

await check('interrupted run retries from checkpoint (dependency recovered) -> done', async () => {
  const store = new InMemoryAgentRunStore();
  const registry = buildRegistry(pythonBridge, new InMemoryDraftStore());
  const r1 = await runAgent('what is DOM', { userId: 'u2', registry, llm: failingLLM, store, maxSteps: 3 });
  assert.equal(r1.stopReason, 'error');
  assert.equal((await store.get(r1.runId!))!.status, 'interrupted');   // checkpointed as interrupted

  const okLLM: LLMClient = {
    available: true,
    async parseFilters() { return {}; },
    async chatWithTools() { return { content: 'DOM = days on market.', toolCalls: [], raw: { role: 'assistant', content: '' } }; },
  };
  const r2 = await retryAgentRun(r1.runId!, { registry, llm: okLLM, store });
  assert.equal(r2.stopReason, 'final');                 // resumed from checkpoint and finished
  assert.match(r2.reply, /days on market/i);
  assert.equal((await store.get(r1.runId!))!.status, 'done');
});

console.log(`\n${pass}/${pass + fail} passed`);
if (fail) process.exit(1);
