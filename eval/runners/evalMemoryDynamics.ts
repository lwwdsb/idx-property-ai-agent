/**
 * Memory DYNAMICS eval — merge semantics, conflict resolution, compaction correctness.
 * Fully deterministic, 0 LLM.
 *
 * The unit tests already cover the happy paths (create, merge, touch, evict). This set exists
 * for the properties that are easy to break and were NOT covered:
 *
 *   CONFLICT (same name)  a changed content is a REVERSED preference, so salience must follow the
 *                         NEW content rather than take the max — a max would leave a stale high
 *                         score sitting on new content.
 *   IDEMPOTENCY           compaction must be a fixed point: running it twice equals running it
 *                         once. It runs in a `finally`, i.e. on every exit path, so a
 *                         non-idempotent compaction would keep gnawing at the store.
 *   SIGNAL LIVENESS       three probes, each varying ONE of compScore's inputs (salience /
 *                         recency / frequency) with everything else equal. If a signal is dead the
 *                         ordering ties and the probe fails. This is not hypothetical: `confidence`
 *                         used to be a field nothing consumed, and only a code read found it.
 *
 *   CROSS-NAME CONTRADICTION  md-009 used to be a KNOWN GAP: two memories with opposite content
 *                         under different names, which no deterministic rule caught. It is now
 *                         CLOSED — not by reading the prose (parsing it produced an inversion on
 *                         this very pair) but by comparing the slots each memory DECLARES, which
 *                         is name-independent by construction. The assertions are two-part: the
 *                         conflict is detected, AND the conflicted field stops seeding searches.
 *                         Both memories still survive; deleting one on a heuristic is not the
 *                         deterministic layer's call.
 *   EPISODIC DYNAMICS     merge/conflict are keyed on NAME alone, so they should be type-agnostic;
 *                         md-013/014 assert that for episodics, which would otherwise rest on the
 *                         assumption that nobody special-cases semantic later. md-012/015 pin the
 *                         type SCOPING of conflict detection in both directions.
 *
 * KNOWN GAPS are reported but excluded from the pass rate (same discipline as the completion
 * self-judge not gating). What remains a gap is now narrower and stated as such: a contradiction
 * that exists ONLY in prose, between memories that declared no slots (md-011), and a same-name
 * re-add that asks for a different type (md-017). If someone later closes either, that case fails
 * and the eval says so — the gap is recorded rather than forgotten.
 *
 *   npx tsx eval/runners/evalMemoryDynamics.ts
 */
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { freshProfile, addMemory, touchMemory, compactMemories, slotConflicts, memoryDerivedFilter,
  type UserProfile } from '../../src/memory/profile.js';

const HERE = new URL('.', import.meta.url).pathname;
const cases = readFileSync(`${HERE}../datasets/memory_dynamics.jsonl`, 'utf8')
  .trim().split('\n').filter(Boolean).map((l) => JSON.parse(l));

/** compScore is private to profile.ts, so it is reproduced here — and md-006..008 exist to catch
 *  the two implementations drifting: if the real one changes shape, the ordering probes break. */
const daysSince = (d: string) => Math.max(0, Math.round((Date.now() - Date.parse(d)) / 86_400_000));
function score(m: { salience: number; useCount: number; lastUsed: string; updatedAt: string }) {
  const fresh = Math.min(daysSince(m.lastUsed), daysSince(m.updatedAt));
  return m.salience * (1 / (1 + fresh / 30)) * (1 + Math.log1p(m.useCount) / 5);
}
function rankOnly(m: { salience: number; lastUsed: string; updatedAt: string }) {
  const fresh = Math.min(daysSince(m.lastUsed), daysSince(m.updatedAt));
  return m.salience * (1 / (1 + fresh / 30));
}
const shiftDays = (iso: string, days: number) =>
  new Date(Date.parse(iso) - days * 86_400_000).toISOString().slice(0, 10);

function run(ops: any[], p: UserProfile) {
  let last: { removed: string[] } | null = null;
  for (const o of ops) {
    if (o.op === 'add') {
      p = addMemory(p, { name: o.name, description: o.description ?? 'd', type: o.type,
        content: o.content, salience: o.salience, sourceRuns: o.sourceRuns, mergedFrom: o.mergedFrom,
        slots: o.slots });
    } else if (o.op === 'touch') {
      // touchMemory mutates in place and does NOT return the profile — reassigning its
      // result wiped the profile on the first call.
      for (let i = 0; i < (o.times ?? 1); i++) touchMemory(p, o.name);
    } else if (o.op === 'age') {
      const m = p.memories.find((x) => x.name === o.name)!;
      m.lastUsed = shiftDays(m.lastUsed, o.days);
      m.updatedAt = shiftDays(m.updatedAt, o.days);
      m.createdAt = shiftDays(m.createdAt, o.days);
    } else if (o.op === 'compact') {
      last = compactMemories(p, { maxSemantic: o.maxSemantic, maxEpisodic: o.maxEpisodic, minScore: o.minScore });
    }
  }
  return { p, last };
}

let pass = 0, total = 0, gapsOk = 0, gapsTotal = 0;
const rows = [];

for (const c of cases) {
  const { p, last } = run(c.ops, freshProfile('eval'));
  const names = p.memories.map((m) => m.name).sort();
  const byName = new Map(p.memories.map((m) => [m.name, m]));
  const e = c.expect;
  const checks: Record<string, boolean> = {};

  if (e.survivors !== undefined) checks.survivors = JSON.stringify(names) === JSON.stringify([...e.survivors].sort());
  if (e.removed !== undefined) checks.removed = (e.removed as string[]).every((n) => !byName.has(n));
  if (e.salience !== undefined) {
    checks.salience = Object.entries(e.salience).every(([n, v]) => byName.get(n)?.salience === v);
  }
  if (e.content_contains !== undefined) {
    checks.content = Object.entries(e.content_contains).every(([n, sub]) => byName.get(n)?.content.includes(sub as string));
  }
  if (e.fields !== undefined) {
    checks.fields = Object.entries(e.fields as Record<string, any>).every(([n, f]) => {
      const m = byName.get(n); if (!m) return false;
      return Object.entries(f).every(([k, v]) => JSON.stringify((m as any)[k]) === JSON.stringify(v));
    });
  }
  if (e.score_order !== undefined) {
    const ordered = [...(e.score_order as string[])];
    const got = ordered.map((n) => byName.get(n)).filter(Boolean) as any[];
    // strict >: a TIE means the varied signal is dead, which is exactly what these probe
    checks.score_order = got.length === ordered.length
      && got.every((m, i) => i === 0 || score(got[i - 1]!) > score(m));
  }
  if (e.rank_order_differs === true) {
    const [a, b] = (e.score_order as string[]).map((n) => byName.get(n)!);
    checks.rank_differs = score(a) > score(b) && !(rankOnly(a) > rankOnly(b));
  }
  // CROSS-NAME CONTRADICTION, detected off DECLARED slots (never off prose — see slotConflicts).
  // Expected as [{type, field}] pairs; order-insensitive.
  if (e.conflicts !== undefined) {
    const got = slotConflicts(p.memories).map((c) => `${c.type}.${c.field}`).sort();
    const want = (e.conflicts as Array<{ type: string; field: string }>)
      .map((c) => `${c.type}.${c.field}`).sort();
    checks.conflicts = JSON.stringify(got) === JSON.stringify(want);
  }
  // What the deterministic channel actually seeds. The point of most conflict cases is not that
  // a conflict was NOTICED but that the conflicted field stops reaching a search at all.
  if (e.derived_filter !== undefined) {
    const got = memoryDerivedFilter(p.memories) as Record<string, unknown>;
    const want = e.derived_filter as Record<string, unknown>;
    const keys = [...new Set([...Object.keys(got), ...Object.keys(want)])];
    checks.derived_filter = keys.every((k) => JSON.stringify(got[k]) === JSON.stringify(want[k]));
  }
  if (e.types !== undefined) {
    checks.types = Object.entries(e.types as Record<string, string>)
      .every(([n, t]) => byName.get(n)?.type === t);
  }
  if (e.idempotent === true) {
    const before = p.memories.map((m) => m.name).sort().join(',');
    const second = compactMemories(p, {});
    checks.idempotent = second.removed.length === 0 && p.memories.map((m) => m.name).sort().join(',') === before;
  }

  const ok = Object.values(checks).every(Boolean);
  const isGap = !!c.meta?.known_gap;
  if (isGap) { gapsTotal++; if (ok) gapsOk++; } else { total++; if (ok) pass++; }
  rows.push({ id: c.id, note: c.meta?.note ?? null, known_gap: isGap, checks, ok,
              survivors: names, removed: last?.removed ?? [] });
  const marks = Object.entries(checks).map(([k, v]) => `${v ? '✓' : '✗'}${k}`).join(' ');
  console.log(`  ${ok ? '✓' : '✗'} ${c.id}${isGap ? ' [known gap]' : ''}  ${marks}`);
  if (!ok) console.log(`      survivors=${JSON.stringify(names)} expect=${JSON.stringify(e)}`);
}

const OUT = `${HERE}../history`;
mkdirSync(OUT, { recursive: true });
writeFileSync(`${OUT}/memory_dynamics.metrics.json`, JSON.stringify({
  n: total, passed: pass, pass_rate: Number((pass / total).toFixed(4)),
  known_gaps: { n: gapsTotal, behaving_as_documented: gapsOk }, rows,
}, null, 2));
console.log(`\ndynamics: ${pass}/${total} passed · 已知缺口 ${gapsOk}/${gapsTotal} 仍按文档记录的行为运行`);
process.exit(pass === total ? 0 : 1);
