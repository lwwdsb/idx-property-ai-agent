/**
 * Memory-consolidation sub-agent tests (offline; scripted fake LLM, in-memory runStore).
 * Verifies it writes memories AND that its capability domain is isolated (memory tools only,
 * userId not a tool parameter, business/unknown tools rejected).
 * Run: npx tsx src/memory/consolidation.test.ts
 */
import assert from 'node:assert/strict';
import { rmSync } from 'node:fs';
import { runConsolidation, MEMORY_TOOLS } from './consolidation.js';
import { loadProfile, addMemory, saveProfile } from './profile.js';
import { InMemoryAgentRunStore, type AgentRunState } from '../agent/auto/runStore.js';
import type { LLMClient, ChatMessage, ToolSpec } from '../llm/client.js';

let pass = 0, fail = 0;
async function check(name: string, fn: () => Promise<void> | void) {
  try { await fn(); pass++; console.log('✓', name); }
  catch (e) { fail++; console.log('✗', name, (e as Error).message); }
}

interface Step { content?: string; toolCalls?: Array<{ name: string; arguments: Record<string, unknown> }>; }
function scriptedLLM(steps: Step[], onTools?: (t: ToolSpec[]) => void, onMessages?: (m: ChatMessage[]) => void): LLMClient {
  let i = 0;
  return {
    available: true,
    async parseFilters() { return {}; },
    async chatWithTools(_messages: ChatMessage[], tools: ToolSpec[]) {
      onTools?.(tools);
      onMessages?.(_messages);
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
  const run = await store.create(uid, state);
  await store.save(run.id, { status: 'done' });   // only COMPLETED runs are eligible
  return store;
};

/** Last tool observation the model was handed (to assert what it actually got to read). */
const lastToolText = (msgs: ChatMessage[]) =>
  [...msgs].reverse().find((m) => m.role === 'tool')?.content as string | undefined;

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


// The backstop lives in a `finally` precisely so it survives the paths where the LLM never
// reached its merge/forget phase. Both tests seed a memory that decay MUST evict.
const seedWeakAndStrong = () => {
  const p = loadProfile(uid);
  addMemory(p, { name: 'weak', description: 'd', type: 'semantic', content: 'c', salience: 0.01 });
  addMemory(p, { name: 'keep', description: 'd', type: 'semantic', content: 'c', salience: 0.9 });
  saveProfile(p);
};

await check('compaction still runs when the LLM burns its step budget', async () => {
  cleanup();
  const store = await seededStore();
  seedWeakAndStrong();
  const llm = scriptedLLM([   // never yields a final turn -> exhausts the budget
    { toolCalls: [{ name: 'list_memories', arguments: {} }] },
    { toolCalls: [{ name: 'list_memories', arguments: {} }] },
    { toolCalls: [{ name: 'list_memories', arguments: {} }] },
  ]);
  const out = await runConsolidation(uid, { llm, runStore: store, maxSteps: 2 });
  assert.match(out, /step budget/);
  const after = loadProfile(uid);
  assert.ok(!after.memories.find((m) => m.name === 'weak'), 'low-score memory should be compacted away');
  assert.ok(after.memories.find((m) => m.name === 'keep'), 'high-score memory must survive');
  cleanup();
});

await check('compaction still runs when the LLM throws', async () => {
  cleanup();
  const store = await seededStore();
  seedWeakAndStrong();
  const boom: LLMClient = {
    available: true,
    async parseFilters() { return {}; },
    async chatWithTools() { throw new Error('provider down'); },
  };
  await assert.rejects(runConsolidation(uid, { llm: boom, runStore: store }), /provider down/);
  const after = loadProfile(uid);
  assert.ok(!after.memories.find((m) => m.name === 'weak'), 'low-score memory should be compacted away');
  cleanup();
});


await check('digests only completed runs; an awaiting_approval run is not consolidated', async () => {
  cleanup();
  const store = new InMemoryAgentRunStore();
  const mk = (text: string): AgentRunState => ({
    task: text, progressive: false, step: 1, activeToolNames: [], trace: [],
    memory: { constraints: {}, facts: [], drafts: [] },
    messages: [{ role: 'user', content: text }, { role: 'assistant', content: 'ok' }],
  });
  const done = await store.create(uid, mk('find 3 bed homes in Irvine'));
  await store.save(done.id, { status: 'done' });
  const pending = await store.create(uid, mk('email the agent about 123 Main St'));
  await store.save(pending.id, { status: 'awaiting_approval' });   // email NOT sent yet

  let seen: string | undefined;
  const llm = scriptedLLM(
    [{ toolCalls: [{ name: 'read_recent_sessions', arguments: { limit: 5 } }] }, { content: 'ok' }],
    undefined, (m) => { seen = lastToolText(m) ?? seen; },
  );
  await runConsolidation(uid, { llm, runStore: store });

  assert.ok(seen?.includes('Irvine'), 'the completed run should be readable');
  assert.ok(!seen?.includes('123 Main St'), 'the awaiting_approval run must not be handed to the model');
  assert.equal(loadProfile(uid).lastConsolidatedRunId, done.id, 'watermark stops at the completed run');
  assert.ok(pending.id > done.id, 'the skipped run is newer — it was excluded by status, not by order');
  cleanup();
});

await check('watermark makes a second pass see nothing new (no re-reading)', async () => {
  cleanup();
  const store = await seededStore();
  const first = scriptedLLM([{ toolCalls: [{ name: 'read_recent_sessions', arguments: {} }] }, { content: 'ok' }]);
  await runConsolidation(uid, { llm: first, runStore: store });
  const mark = loadProfile(uid).lastConsolidatedRunId;
  assert.ok(mark && mark > 0, 'first pass advances the watermark');

  let seen: string | undefined;
  const second = scriptedLLM(
    [{ toolCalls: [{ name: 'read_recent_sessions', arguments: {} }] }, { content: 'ok' }],
    undefined, (m) => { seen = lastToolText(m) ?? seen; },
  );
  await runConsolidation(uid, { llm: second, runStore: store });
  assert.match(seen ?? '', /No new completed sessions/);
  assert.equal(loadProfile(uid).lastConsolidatedRunId, mark, 'watermark unchanged when nothing new');
  cleanup();
});

console.log(`\n${pass}/${pass + fail} passed`);
if (fail) process.exit(1);
