/**
 * Long-term classified-memory tests (offline): facts (immediate learning) + semantic/
 * episodic entries (add/merge/touch/select) + store round-trip.
 * Run: npx tsx src/memory/profile.test.ts
 */
import assert from 'node:assert/strict';
import { rmSync } from 'node:fs';
import {
  freshProfile, learnFromFilter, preferredFilter, saveFacts, saveMemories, saveUsage, loadProfile, renderMd,
  addMemory, touchMemory, selectMemories, semanticMemories, episodicMemories, profileHint,
  compactMemories, forgetMemory,
} from './profile.js';
import type { SearchFilter } from '../search/filters.js';

let pass = 0, fail = 0;
async function check(name: string, fn: () => void | Promise<void>) {
  try { await fn(); pass++; console.log('✓', name); }
  catch (e) { fail++; console.log('✗', name, (e as Error).message); }
}
const f = (o: Partial<SearchFilter>) => o as SearchFilter;

// ── facts (structured, 0 LLM) ──
await check('fact: new field low-confidence; repetition reinforces past threshold', () => {
  let p = freshProfile('u');
  p = learnFromFilter(p, f({ city: 'Irvine' }));
  assert.ok(p.prefs.city!.confidence < 0.5);
  for (let i = 0; i < 2; i++) p = learnFromFilter(p, f({ city: 'Irvine' }));
  assert.ok(p.prefs.city!.confidence >= 0.5 && p.prefs.city!.seen === 3);
});
await check('fact: conflicting value erodes then replaces; preferredFilter = high-conf only', () => {
  let p = freshProfile('u');
  for (let i = 0; i < 3; i++) p = learnFromFilter(p, f({ city: 'Irvine', beds: 3 }));
  p = learnFromFilter(p, f({ maxPrice: 2000000 }));   // low conf
  const def = preferredFilter(p);
  assert.equal(def.city, 'Irvine'); assert.equal(def.beds, 3); assert.equal(def.maxPrice, undefined);
});

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
await check('profileHint: facts always; memories ONLY if they were selected', () => {
  let p = freshProfile('u');
  for (let i = 0; i < 3; i++) p = learnFromFilter(p, f({ city: 'Irvine' }));
  addMemory(p, { name: 'schools', description: 'schools', type: 'semantic', content: 'likes good schools' });
  addMemory(p, { name: 'ev', description: 'event', type: 'episodic', content: 'drafted a report' });
  // Nothing selected -> no memories injected, semantic included. Injecting what was never
  // selected is exactly what made useCount meaningless.
  const bare = profileHint(p);
  assert.ok(bare.includes('Irvine'), 'facts are still always injected');
  assert.ok(!bare.includes('likes good schools') && !bare.includes('drafted a report'));
  const withSel = profileHint(p, semanticMemories(p));
  assert.ok(withSel.includes('likes good schools'));
});

// ── store round-trip ──
await check('save/load round-trip + md renders facts and memories', () => {
  const uid = 'test-profile-user';
  let p = freshProfile(uid);
  for (let i = 0; i < 3; i++) p = learnFromFilter(p, f({ city: 'Irvine' }));
  addMemory(p, { name: 'schools', description: 'cares about schools', type: 'semantic', content: 'likes good school zones' });
  saveFacts(p); saveMemories(p); saveUsage(p);   // one call per writer-owned file
  const loaded = loadProfile(uid);
  assert.equal(loaded.prefs.city!.value, 'Irvine');
  assert.equal(semanticMemories(loaded).length, 1);
  const md = renderMd(loaded);
  assert.ok(md.includes('Irvine') && md.includes('cares about schools'));
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
  learnFromFilter(chat, f({ city: 'Irvine' }));
  saveFacts(chat);

  const agent = loadProfile(uid);
  addMemory(agent, { name: 'm1', description: 'd', type: 'semantic', content: 'c' });
  agent.lastConsolidatedRunId = 7;
  saveMemories(agent);

  const stale = freshProfile(uid);            // a writer that never saw the memory at all
  learnFromFilter(stale, f({ beds: 3 }));
  saveFacts(stale);

  const merged = loadProfile(uid);
  assert.equal(merged.prefs.beds!.value, 3, 'latest fact write wins');
  assert.equal(merged.memories.length, 1, 'memory survives a facts-only write');
  assert.equal(merged.lastConsolidatedRunId, 7, 'watermark survives too');
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
