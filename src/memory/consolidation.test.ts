/**
 * Memory-consolidation sub-agent tests (offline; scripted fake LLM, in-memory runStore).
 * Verifies it writes memories AND that its capability domain is isolated (memory tools only,
 * userId not a tool parameter, business/unknown tools rejected).
 * Run: npx tsx src/memory/consolidation.test.ts
 */
import assert from 'node:assert/strict';
import { rmSync } from 'node:fs';
import { runConsolidation, MEMORY_TOOLS } from './consolidation.js';
import { loadProfile } from './profile.js';
import { InMemoryAgentRunStore, type AgentRunState } from '../agent/auto/runStore.js';
import type { LLMClient, ChatMessage, ToolSpec } from '../llm/client.js';

let pass = 0, fail = 0;
async function check(name: string, fn: () => Promise<void> | void) {
  try { await fn(); pass++; console.log('✓', name); }
  catch (e) { fail++; console.log('✗', name, (e as Error).message); }
}

interface Step { content?: string; toolCalls?: Array<{ name: string; arguments: Record<string, unknown> }>; }
function scriptedLLM(steps: Step[], onTools?: (t: ToolSpec[]) => void): LLMClient {
  let i = 0;
  return {
    available: true,
    async parseFilters() { return {}; },
    async chatWithTools(_messages: ChatMessage[], tools: ToolSpec[]) {
      onTools?.(tools);
      const s = steps[i++] ?? { content: 'done' };
      return {
        content: s.content ?? '',
        toolCalls: (s.toolCalls ?? []).map((t, ix) => ({ id: `c${ix}`, name: t.name, arguments: t.arguments })),
        raw: { role: 'assistant', content: s.content ?? '' } as ChatMessage,
      };
    },
  };
}

const uid = 'test-consolidate-user';
const cleanup = () => { rmSync(`data/profiles/test-consolidate-user.json`, { force: true }); rmSync(`data/profiles/test-consolidate-user.md`, { force: true }); };

const seededStore = async () => {
  const store = new InMemoryAgentRunStore();
  const state: AgentRunState = {
    task: 'find homes', progressive: false, step: 1, activeToolNames: [], trace: [],
    memory: { constraints: {}, facts: [], drafts: [] },
    messages: [
      { role: 'user', content: 'find 3 bed homes in Irvine under 2M' },
      { role: 'assistant', content: 'here are some Irvine listings...' },
    ],
  };
  await store.create(uid, state);
  return store;
};

await check('writes a memory from recent sessions (read -> add -> final)', async () => {
  cleanup();
  const store = await seededStore();
  const llm = scriptedLLM([
    { toolCalls: [{ name: 'read_recent_sessions', arguments: { limit: 5 } }] },
    { toolCalls: [{ name: 'add_memory', arguments: { name: 'irvine-3bed', description: 'looks for 3-bed in Irvine', type: 'semantic', content: 'wants 3-bedroom homes in Irvine', salience: 0.7 } }] },
    { content: 'Stored 1 preference.' },
  ]);
  const summary = await runConsolidation(uid, { llm, runStore: store });
  assert.equal(summary, 'Stored 1 preference.');
  const p = loadProfile(uid);
  assert.equal(p.memories.length, 1);
  assert.equal(p.memories[0]!.name, 'irvine-3bed');
  cleanup();
});

await check('ISOLATION: loop is handed only the memory-tool whitelist', async () => {
  cleanup();
  const store = await seededStore();
  let seen: string[] = [];
  const llm = scriptedLLM([{ content: 'nothing to do' }], (tools) => { seen = tools.map((t) => t.name); });
  await runConsolidation(uid, { llm, runStore: store });
  assert.deepEqual(seen, ['read_recent_sessions', 'list_memories', 'add_memory', 'forget_memory']);
  assert.ok(!seen.includes('search') && !seen.includes('email'));   // no business tools reachable
  cleanup();
});

await check('consolidation: promotes episodics to semantic and forgets the redundant ones', async () => {
  cleanup();
  const store = await seededStore();
  // seed two episodics, then the agent promotes them into one semantic + forgets both
  const { loadProfile: lp, addMemory: am, saveProfile: sp } = await import('./profile.js');
  const seed = lp(uid);
  am(seed, { name: 'ev-a', description: 'looked at Irvine schools', type: 'episodic', content: 'a' });
  am(seed, { name: 'ev-b', description: 'looked at Irvine schools again', type: 'episodic', content: 'b' });
  sp(seed);
  const llm = scriptedLLM([
    { toolCalls: [{ name: 'add_memory', arguments: { name: 'cares-schools', description: 'cares about school zones', type: 'semantic', content: 'prioritizes good school zones', salience: 0.7, mergedFrom: ['ev-a', 'ev-b'] } }] },
    { toolCalls: [{ name: 'forget_memory', arguments: { name: 'ev-a' } }, { name: 'forget_memory', arguments: { name: 'ev-b' } }] },
    { content: 'Promoted 2 episodics into 1 semantic.' },
  ]);
  await runConsolidation(uid, { llm, runStore: store });
  const p = loadProfile(uid);
  assert.equal(p.memories.length, 1);
  assert.equal(p.memories[0]!.name, 'cares-schools');
  assert.deepEqual(p.memories[0]!.mergedFrom, ['ev-a', 'ev-b']);   // lineage kept
  cleanup();
});

await check('ISOLATION: add_memory has no userId parameter (cannot address another user)', () => {
  const addTool = MEMORY_TOOLS.find((t) => t.name === 'add_memory')!;
  const props = (addTool.parameters as { properties: Record<string, unknown> }).properties;
  assert.ok(!('userId' in props) && !('user' in props));
});

await check('ISOLATION: an unknown/business tool call is rejected, not executed', async () => {
  cleanup();
  const store = await seededStore();
  const llm = scriptedLLM([
    { toolCalls: [{ name: 'email', arguments: { to: 'x@y.com' } }] },   // pretend the model tries a business tool
    { content: 'ok' },
  ]);
  await runConsolidation(uid, { llm, runStore: store });   // must not throw / not send anything
  assert.equal(loadProfile(uid).memories.length, 0);        // nothing written by a bogus tool
  cleanup();
});

console.log(`\n${pass}/${pass + fail} passed`);
if (fail) process.exit(1);
