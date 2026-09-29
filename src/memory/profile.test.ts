/**
 * Long-term classified-memory tests (offline): facts (immediate learning) + semantic/
 * episodic entries (add/merge/touch/select) + store round-trip.
 * Run: npx tsx src/memory/profile.test.ts
 */
import assert from 'node:assert/strict';
import { rmSync } from 'node:fs';
import {
  freshProfile, saveMemories, saveUsage, loadProfile, renderMd,
  addMemory, touchMemory, selectMemories, semanticMemories, episodicMemories, profileHint,
  compactMemories, forgetMemory, memoryDerivedFilter, slotConflicts,
} from './profile.js';
import type { SearchFilter } from '../search/filters.js';

let pass = 0, fail = 0;
async function check(name: string, fn: () => void | Promise<void>) {
  try { await fn(); pass++; console.log('✓', name); }
  catch (e) { fail++; console.log('✗', name, (e as Error).message); }
}
const f = (o: Partial<SearchFilter>) => o as SearchFilter;

// ── facts (structured, 0 LLM) ──

// ── semantic / episodic entries ──
await check('memory: addMemory creates then merges (salience max, sourceRuns union)', () => {
  const p = freshProfile('u');
  addMemory(p, { name: 'schools', description: 'cares about schools', type: 'semantic', content: 'likes good school zones', salience: 0.5, sourceRuns: [1] });
  addMemory(p, { name: 'schools', description: 'cares about schools', type: 'semantic', content: 'likes good school zones', salience: 0.8, sourceRuns: [2] });
  assert.equal(p.memories.length, 1);
  assert.equal(p.memories[0]!.salience, 0.8);
  assert.deepEqual(p.memories[0]!.sourceRuns, [1, 2]);
});
await check('memory: same name + OPPOSITE content = replace (salience follows new, not max)', () => {
  const p = freshProfile('u');
  addMemory(p, { name: 'schools', description: 'd', type: 'semantic', content: 'cares about schools', salience: 0.9 });
  addMemory(p, { name: 'schools', description: 'd', type: 'semantic', content: 'does NOT care about schools', salience: 0.4 });
  assert.equal(p.memories.length, 1);
  assert.ok(p.memories[0]!.content.includes('does NOT'));   // new content wins (recency)
  assert.equal(p.memories[0]!.salience, 0.4);               // replaced, NOT max(0.9, 0.4)
});
await check('memory: touchMemory bumps useCount', () => {
  const p = freshProfile('u');
  addMemory(p, { name: 'ev1', description: 'drafted Irvine report', type: 'episodic', content: 'x' });
  touchMemory(p, 'ev1');
  assert.equal(p.memories[0]!.useCount, 1);
});
await check('memory: an EMPTY llm selection means "none", not "fall back to everything"', async () => {
  let p = freshProfile('u');
  p = addMemory(p, { name: 'a', description: 'd', type: 'semantic', content: 'c', salience: 0.9 });
  p = addMemory(p, { name: 'b', description: 'd', type: 'semantic', content: 'c', salience: 0.8 });
  const saysNone: any = { available: true, async chatWithTools() { return { content: '[]', toolCalls: [], raw: {} }; } };
  assert.deepEqual((await selectMemories(p.memories, 'hello', saysNone)).map((m) => m.name), [],
    'a valid "none" must inject nothing — otherwise a greeting drags the whole profile into the prompt');
  const garbage: any = { available: true, async chatWithTools() { return { content: 'sorry, what?', toolCalls: [], raw: {} }; } };
  assert.equal((await selectMemories(p.memories, 'hello', garbage)).length, 2,
    'an UNPARSEABLE reply is a real failure and must still degrade to the deterministic rank');
});
await check('memory: selectMemories fallback ranks by salience (no LLM)', async () => {
  const p = freshProfile('u');
  addMemory(p, { name: 'hi', description: 'important', type: 'episodic', content: 'A', salience: 0.9 });
  addMemory(p, { name: 'lo', description: 'minor', type: 'episodic', content: 'B', salience: 0.2 });
  const sel = await selectMemories(episodicMemories(p), 'anything', undefined, { episodic: 1 });
  assert.equal(sel.length, 1); assert.equal(sel[0]!.name, 'hi');
});
await check('memory: semantic is selected too (so useCount is a real signal)', async () => {
  const p = freshProfile('u');
  addMemory(p, { name: 'sem', description: 'prefers old homes', type: 'semantic', content: 'S', salience: 0.8 });
  addMemory(p, { name: 'ev', description: 'drafted a report', type: 'episodic', content: 'E', salience: 0.8 });
  const sel = await selectMemories(p.memories, 'anything', undefined);
  assert.deepEqual(sel.map((m) => m.name).sort(), ['ev', 'sem'], 'both types are eligible');
  sel.forEach((m) => touchMemory(p, m.name));
  // Wholesale injection made this impossible: everything rose together, so order never moved.
  assert.equal(p.memories.find((m) => m.name === 'sem')!.useCount, 1);
});
await check('memory: caps apply PER TYPE — episodics cannot crowd out preferences', async () => {
  const p = freshProfile('u');
  for (let i = 0; i < 8; i++) {
    addMemory(p, { name: `ev${i}`, description: 'event', type: 'episodic', content: 'E', salience: 0.9 });
  }
  addMemory(p, { name: 'sem', description: 'pref', type: 'semantic', content: 'S', salience: 0.1 });
  const sel = await selectMemories(p.memories, 'anything', undefined, { semantic: 5, episodic: 3 });
  assert.equal(sel.filter((m) => m.type === 'episodic').length, 3);
  assert.ok(sel.some((m) => m.name === 'sem'), 'a low-salience preference still gets its own slot');
});
await check('profileHint: 记忆必须【被选中】才注入;没选中就什么都不注入', () => {
  const p = freshProfile('u');
  addMemory(p, { name: 'schools', description: 'schools', type: 'semantic', content: 'likes good schools' });
  addMemory(p, { name: 'ev', description: 'event', type: 'episodic', content: 'drafted a report' });
  // 注入没被选中的东西,正是当初让 useCount 失去意义的原因。事实层删除之后,profileHint 里已经
  // 没有"总是注入"的部分 —— 空档案给出空串,而不是一段没有内容的模板。
  assert.equal(profileHint(p), '', '什么都没选中 -> 不注入任何东西');
  const withSel = profileHint(p, semanticMemories(p));
  assert.ok(withSel.includes('likes good schools'));
  assert.ok(!withSel.includes('drafted a report'), '只注入选中的那些');
});

// ── store round-trip ──
await check('save/load round-trip + md 渲染记忆与声明式 slots', () => {
  const uid = 'test-profile-user';
  const p = freshProfile(uid);
  addMemory(p, { name: 'budget', description: '预算上限 200 万', type: 'semantic',
    content: '超过 200 万不考虑', slots: { maxPrice: 2_000_000 } as never });
  addMemory(p, { name: 'schools', description: 'cares about schools', type: 'semantic', content: 'likes good school zones' });
  saveMemories(p); saveUsage(p);   // one call per writer-owned file
  const loaded = loadProfile(uid);
  assert.equal(semanticMemories(loaded).length, 2);
  const md = renderMd(loaded);
  assert.ok(md.includes('cares about schools'), 'md 列出语义记忆');
  assert.ok(md.includes('maxPrice'), 'md 列出声明式 slots —— 这是现在唯一的跨会话软默认来源,必须可审查');
  for (const suffix of ['facts.json', 'memories.json', 'usage.json', 'md']) {
    rmSync(`data/profiles/${uid}.${suffix}`, { force: true });
  }
});

await check('split store: the two writers do not clobber each other', () => {
  const uid = 'test-split-user';
  const clean = () => { for (const x of ['facts.json', 'memories.json', 'usage.json', 'md']) rmSync(`data/profiles/${uid}.${x}`, { force: true }); };
  clean();
  // chat path writes a fact; consolidation path writes a memory — from SEPARATE loads, the way
  // two concurrent writers would. Under one shared file the second save would erase the first.
  const chat = freshProfile(uid);
  addMemory(chat, { name: 'pref-a', description: 'd', type: 'semantic', content: 'c' });
  saveMemories(chat);

  const agent = loadProfile(uid);
  addMemory(agent, { name: 'm1', description: 'd', type: 'semantic', content: 'c' });
  agent.lastConsolidatedRunId = 7;
  saveMemories(agent);

  const stale = freshProfile(uid);            // 一个从没见过对方写入的 writer
  touchMemory(stale, 'pref-a');
  saveUsage(stale);

  const merged = loadProfile(uid);
  assert.equal(merged.memories.length, 2, '两个 writer 的记忆都在,互不覆盖');
  assert.equal(merged.lastConsolidatedRunId, 7, 'watermark survives too');
  clean();
});

await check('round-trip: declared slots and usage counters survive a reload', () => {
  const uid = 'test-roundtrip-user';
  const clean = () => { for (const x of ['facts.json', 'memories.json', 'usage.json', 'md']) rmSync(`data/profiles/${uid}.${x}`, { force: true }); };
  clean();
  // SLOTS ARE THE WHOLE DETERMINISTIC CHANNEL NOW. memoryDerivedFilter reads declared slots and
  // nothing else — prose is not parsed — so a slot lost on reload does not degrade the filter, it
  // silently empties it, and the memory still LOOKS intact in profile.md. The existing round-trip
  // case predates the field and stores a memory without any, so nothing covered this.
  const w = freshProfile(uid);
  addMemory(w, { name: 'wants-pool', description: 'd', type: 'semantic', content: '要带泳池',
    slots: { pool: true, maxPrice: 2_000_000 } as never });
  saveMemories(w);
  touchMemory(w, 'wants-pool');
  touchMemory(w, 'wants-pool');
  saveUsage(w);

  const r = loadProfile(uid);
  assert.deepEqual(r.memories[0]!.slots, { pool: true, maxPrice: 2_000_000 }, 'slots survive the reload');
  assert.deepEqual(memoryDerivedFilter(r.memories), { pool: true, maxPrice: 2_000_000 },
    'and still reach the deterministic filter after a reload');
  // usage lives in its own file written by the chat path; the reload has to re-attach it by name.
  assert.equal(r.memories[0]!.useCount, 2, 'usage counters survive and re-attach');
  clean();
});

await check('a cross-name slot contradiction survives reload AND keeps the field silent', () => {
  const uid = 'test-conflict-user';
  const clean = () => { for (const x of ['facts.json', 'memories.json', 'usage.json', 'md']) rmSync(`data/profiles/${uid}.${x}`, { force: true }); };
  clean();
  // The conflict is derived, not stored, so it has to be re-derived after a reload. If it were not,
  // a restart would resurrect the arbitrary first-writer-wins resolution this replaced.
  const w = freshProfile(uid);
  addMemory(w, { name: 'wants-pool', description: 'd', type: 'semantic', content: '要泳池', slots: { pool: true } as never });
  addMemory(w, { name: 'no-pool-please', description: 'd', type: 'semantic', content: '不要泳池', slots: { pool: false } as never });
  saveMemories(w);

  const r = loadProfile(uid);
  assert.equal(r.memories.length, 2, 'both memories survive — neither is deleted on a heuristic');
  assert.equal(slotConflicts(r.memories).length, 1, 'the contradiction is re-detected after reload');
  assert.deepEqual(memoryDerivedFilter(r.memories), {}, 'and the conflicted field stays silent');
  clean();
});

await check('compact: evicts over-capacity, keeping top-N by score', () => {
  const p = freshProfile('u');
  for (let i = 0; i < 32; i++) addMemory(p, { name: `s${i}`, description: 'd', type: 'semantic', content: 'c', salience: i / 32 });
  const { removed } = compactMemories(p, { maxSemantic: 30, minScore: 0 });   // isolate capacity
  assert.equal(removed.length, 2);
  assert.equal(semanticMemories(p).length, 30);
  assert.ok(!p.memories.find((m) => m.name === 's0'));   // lowest salience evicted
});
await check('compact: decay-evicts very-low score; forgetMemory removes by name', () => {
  const p = freshProfile('u');
  addMemory(p, { name: 'keep', description: 'd', type: 'semantic', content: 'c', salience: 0.9 });
  addMemory(p, { name: 'weak', description: 'd', type: 'semantic', content: 'c', salience: 0.01 });
  compactMemories(p, { minScore: 0.08 });
  assert.ok(p.memories.find((m) => m.name === 'keep') && !p.memories.find((m) => m.name === 'weak'));
  assert.equal(forgetMemory(p, 'keep'), true);
  assert.equal(p.memories.length, 0);
});

console.log(`\n${pass}/${pass + fail} passed`);
if (fail) process.exit(1);
