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
import { evalDynamicsCase } from './lib/memoryDynamicsCase.js';

const HERE = new URL('.', import.meta.url).pathname;
const cases = readFileSync(`${HERE}../datasets/memory_dynamics.jsonl`, 'utf8')
  .trim().split('\n').filter(Boolean).map((l) => JSON.parse(l));

let pass = 0, total = 0, gapsOk = 0, gapsTotal = 0;
const rows = [];

for (const c of cases) {
  const { checks, ok, survivors: names, removed } = evalDynamicsCase(c);
  const isGap = !!c.meta?.known_gap;
  if (isGap) { gapsTotal++; if (ok) gapsOk++; } else { total++; if (ok) pass++; }
  rows.push({ id: c.id, note: c.meta?.note ?? null, known_gap: isGap, checks, ok,
              survivors: names, removed });
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
