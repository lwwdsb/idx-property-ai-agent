/**
 * Memory FACTS eval — the structured-preference layer (0 LLM, fully deterministic).
 *
 * Facts are the layer BOTH modes share, and the only one that touches the deterministic path,
 * so its correctness matters more than the semantic/episodic layers. Two things are measured:
 *
 *   LEARNING   after a sequence of turns, are the right fields enabled? A field needs the same
 *              value roughly three times to pass the 0.5 confidence gate — a single mention is
 *              not a preference, and a contradicting value ERODES rather than replaces (so the
 *              store does not blindly trust the newest turn).
 *   APPLICATION a remembered value may only fill a BLANK. If the user states a field this turn,
 *              the stated value must win. That is a HARD INVARIANT, in the same class as
 *              "emails actually delivered = 0": one violation is a failure regardless of every
 *              other number, because a preference that overrides an explicit request is not a
 *              default, it is a hijack.
 *
 *   npx tsx eval/runners/evalMemoryFacts.ts
 */
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { freshProfile, learnFromFilter, preferredFilter } from '../../src/memory/profile.js';
import { mergeFilter } from '../../src/search/filters.js';
import type { SearchFilter } from '../../src/search/filters.js';

const HERE = new URL('.', import.meta.url).pathname;
const cases = readFileSync(`${HERE}../datasets/memory_facts.jsonl`, 'utf8')
  .trim().split('\n').filter(Boolean).map((l) => JSON.parse(l));

let pass = 0, invariantViolations = 0;
const rows = [];

for (const c of cases) {
  let p = freshProfile('eval');
  for (const t of c.turns) p = learnFromFilter(p, t.stated as SearchFilter);

  const enabled = Object.keys(preferredFilter(p)).sort();
  // Production order: remembered values are the LAST resort, so the stated filter is merged ON TOP.
  const applied = mergeFilter(preferredFilter(p) as SearchFilter, c.final.stated) as Record<string, unknown>;

  const checks: Record<string, boolean> = {};
  if (c.gold.enabled !== undefined) {
    checks.enabled = JSON.stringify(enabled) === JSON.stringify([...c.gold.enabled].sort());
  }
  if (c.gold.applied !== undefined) {
    checks.applied = Object.entries(c.gold.applied)
      .every(([k, v]) => applied[k] === v) && Object.keys(applied).length === Object.keys(c.gold.applied).length;
  }
  if (c.gold.applied_has !== undefined) {
    checks.applied_has = (c.gold.applied_has as string[]).every((k) => applied[k] !== undefined);
  }
  // THE INVARIANT, checked on every case regardless of what else it asserts.
  const stated = c.final.stated as Record<string, unknown>;
  const hijacked = Object.keys(stated).filter((k) => applied[k] !== stated[k]);
  checks.no_hijack = hijacked.length === 0;
  if (hijacked.length) invariantViolations += 1;

  const ok = Object.values(checks).every(Boolean);
  if (ok) pass += 1;
  rows.push({ id: c.id, note: c.meta?.note ?? null, enabled, applied, checks, ok, hijacked });
  const marks = Object.entries(checks).map(([k, v]) => `${v ? '✓' : '✗'}${k}`).join(' ');
  console.log(`  ${ok ? '✓' : '✗'} ${c.id}  enabled=${JSON.stringify(enabled)}  ${marks}`);
  if (!ok) console.log(`      applied=${JSON.stringify(applied)}  gold=${JSON.stringify(c.gold)}`);
}

const OUT = `${HERE}../history`;
mkdirSync(OUT, { recursive: true });
writeFileSync(`${OUT}/memory_facts.metrics.json`, JSON.stringify({
  n: cases.length, passed: pass, pass_rate: Number((pass / cases.length).toFixed(4)),
  hijack_violations: invariantViolations,
  invariant_ok: invariantViolations === 0, rows,
}, null, 2));
console.log(`\nfacts: ${pass}/${cases.length} passed · 记忆覆盖当前指令的违例 = ${invariantViolations}`
  + `${invariantViolations === 0 ? ' (硬不变量成立)' : '  ✗ 硬不变量被破坏'}`);
process.exit(invariantViolations === 0 ? 0 : 1);
