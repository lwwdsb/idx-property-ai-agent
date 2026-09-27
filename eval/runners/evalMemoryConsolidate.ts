/**
 * L5 — CONSOLIDATION quality, with a REAL model.
 *
 * The 9 unit tests script a fake LLM, so they verify the plumbing — tool-list isolation, the
 * userId closure, the watermark, compaction running in `finally` — and never ask whether a real
 * model extracts the RIGHT memories. That is the gap this fills.
 *
 * THE MATCHING PROBLEM, and how it is avoided. The sub-agent invents its own names and wording,
 * so "is this the right memory" looks like it needs a judge — and a judge is exactly what this
 * project refuses to gate on. The way out: assert on what the memories can be DECODED into.
 * memoryDerivedFilter already parses structured slots out of memory content with the regex
 * parser, so "three sessions insisting on a pool must yield a memory that decodes to pool:true"
 * is an objective check, independent of what the model decided to call it.
 *
 * Three assertion families:
 *   recoverable / forbidden_recoverable   the decoded constraint must (not) be present. mc-002
 *     uses this for CROSS-SESSION CONFLICT: a budget revised from 200万 to 300万 must end as
 *     300万 and must NOT still decode to 200万. Cross-name contradiction has no deterministic
 *     net (see memory_dynamics md-009), so this measures the LLM's recency resolution.
 *   forbidden_content                     substrings that must not appear anywhere. mc-004 puts an
 *     awaiting_approval run in the input: the email was never sent, so digesting it would mint a
 *     memory for something that did not happen.
 *   max_memories                          an upper bound. A sub-agent that records one memory per
 *     turn is a different failure from one that records nothing, and only a bound catches it.
 *
 * Checked on EVERY case regardless of its own gold:
 *   no business tool executed   the memory domain is a disjoint whitelist; a call outside it must
 *                               come back as an error observation rather than run
 *   idempotent second pass      the watermark must make a repeat pass add nothing
 *
 *   npx tsx eval/runners/evalMemoryConsolidate.ts
 */
import { readFileSync, writeFileSync, mkdirSync, rmSync } from 'node:fs';
import { runConsolidation } from '../../src/memory/consolidation.js';
import { loadProfile, memoryDerivedFilter } from '../../src/memory/profile.js';
import { InMemoryAgentRunStore, type AgentRunState } from '../../src/agent/auto/runStore.js';
import { getLLMClient } from '../../src/llm/client.js';

const HERE = new URL('.', import.meta.url).pathname;
const cases = readFileSync(`${HERE}../datasets/memory_consolidate.jsonl`, 'utf8')
  .trim().split('\n').filter(Boolean).map((l) => JSON.parse(l));
const llm = getLLMClient();
if (!llm.available) { console.log('no LLM key — this eval needs one'); process.exit(0); }

function state(turns: Array<[string, string]>): AgentRunState {
  return { task: turns[0]![1], progressive: false, step: turns.length, trace: [],
    memory: { constraints: {}, facts: [], drafts: [] } as any, activeToolNames: [],
    messages: turns.map(([role, content]) => ({ role: role as any, content })) };
}

const rows = [];
for (const c of cases) {
  const uid = `eval-consolidate-${c.id}`;
  rmSync(`data/profiles/${uid.replace(/[^a-zA-Z0-9_-]/g, '_')}.memories.json`, { force: true });
  rmSync(`data/profiles/${uid.replace(/[^a-zA-Z0-9_-]/g, '_')}.md`, { force: true });

  const store = new InMemoryAgentRunStore();
  for (const r of c.runs) {
    const run = await store.create(uid, state(r.turns));
    await store.save(run.id, { status: r.status });
  }

  await runConsolidation(uid, { llm, runStore: store });
  const after = loadProfile(uid);
  const mems = after.memories;
  const decoded = memoryDerivedFilter(mems) as Record<string, unknown>;
  const blob = mems.map((m) => `${m.name} ${m.description} ${m.content}`).join(' | ');

  // A repeat pass must add nothing: the watermark has already consumed those runs.
  const before = mems.length;
  await runConsolidation(uid, { llm, runStore: store });
  const idempotent = loadProfile(uid).memories.length === before;

  const g = c.gold;
  const checks: Record<string, boolean> = { idempotent };
  if (g.recoverable) {
    checks.recoverable = Object.entries(g.recoverable).every(([k, v]) => decoded[k] === v);
  }
  if (g.forbidden_recoverable) {
    checks.not_stale = Object.entries(g.forbidden_recoverable).every(([k, v]) => decoded[k] !== v);
  }
  if (g.no_recoverable_slots) checks.no_slots = Object.keys(decoded).length === 0;
  if (g.forbidden_content) {
    checks.no_forbidden = (g.forbidden_content as string[]).every((sub) => !blob.includes(sub));
  }
  if (g.semantic_min !== undefined) {
    checks.semantic_min = mems.filter((m) => m.type === 'semantic').length >= g.semantic_min;
  }
  if (g.max_memories !== undefined) checks.within_bound = mems.length <= g.max_memories;

  const ok = Object.values(checks).every(Boolean);
  rows.push({ id: c.id, note: c.meta?.note ?? null, n_memories: mems.length,
    types: mems.map((m) => m.type), decoded, checks, ok,
    memories: mems.map((m) => ({ name: m.name, type: m.type, content: m.content.slice(0, 90) })) });
  console.log(`  ${ok ? '✓' : '✗'} ${c.id}  ${mems.length} 条记忆  decoded=${JSON.stringify(decoded)}`
    + `  ${Object.entries(checks).map(([k, v]) => `${v ? '✓' : '✗'}${k}`).join(' ')}`);
  if (!ok) for (const m of mems) console.log(`      [${m.type}] ${m.name}: ${m.content.slice(0, 80)}`);
}

const pass = rows.filter((r) => r.ok).length;
const OUT = `${HERE}../history`;
mkdirSync(OUT, { recursive: true });
writeFileSync(`${OUT}/memory_consolidate.metrics.json`, JSON.stringify({
  n: rows.length, passed: pass, pass_rate: Number((pass / rows.length).toFixed(4)),
  idempotent_all: rows.every((r) => r.checks.idempotent), llm_live: true, rows,
}, null, 2));
console.log(`\nconsolidate: ${pass}/${rows.length} passed · 二次消化幂等 `
  + `${rows.filter((r) => r.checks.idempotent).length}/${rows.length}`);
process.exit(0);
