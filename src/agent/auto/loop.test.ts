/**
 * Auto loop resilience test (offline): when the LLM fails even after retry+breaker,
 * runAgent must NOT throw — it checkpoints and returns a graceful 'error' result.
 * Run: npx tsx src/agent/auto/loop.test.ts
 */
import assert from 'node:assert/strict';
import { runAgent } from './loop.js';
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

console.log(`\n${pass}/${pass + fail} passed`);
if (fail) process.exit(1);
