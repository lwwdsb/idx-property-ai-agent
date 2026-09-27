/**
 * L4 — memory UTILITY. Does injecting memory actually change the outcome?
 *
 * Every other memory metric measures whether the machinery works. This one measures whether the
 * machinery is WORTH HAVING, and it is the only one that can justify the size limits: with no
 * utility measurement, moving selectSemantic/selectEpisodic from 5/3 to 8/5 moves no number, so
 * those knobs are untunable rather than tuned.
 *
 * PAIRED BY CONSTRUCTION: each case runs twice on identical input, once with memory and once
 * without. The decisive assertion is NOT "succeeds with memory" — it is that the two arms
 * DIFFER. If a task succeeds either way, that memory carried no weight and the limit question
 * around it is fake.
 *
 * Two layers, because they are injected through different doors:
 *   facts  deterministic, 0 LLM. preferredFilter feeds mergeFilter as a LAST resort, so utility
 *          is visible directly in the merged filter.
 *   agent  the real ReAct loop, twice. The assertion is on TOOL ARGUMENTS rather than on reply
 *          text: whether the remembered constraint reached the call is objective, whereas
 *          grading prose would need a judge.
 *
 * Three cases are NEGATIVE CONTROLS and matter as much as the positive ones — a memory system
 * that changes everything is as broken as one that changes nothing:
 *   mu-003  below the confidence gate  -> memory must have NO effect
 *   mu-004  the turn states the field  -> memory must not override it
 *   mu-007  an irrelevant memory       -> must not add constraints to the tool call
 *
 *   npx tsx eval/runners/evalMemoryUtility.ts
 */
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { freshProfile, learnFromFilter, preferredFilter, addMemory, profileHint, seedFilterFor, selectMemories } from '../../src/memory/profile.js';
import { mergeFilter, type SearchFilter } from '../../src/search/filters.js';
import { runAgent } from '../../src/agent/auto/loop.js';
import { InMemoryAgentRunStore } from '../../src/agent/auto/runStore.js';
import { InMemoryDraftStore } from '../../src/email/drafts.js';
import { buildRegistry } from '../../src/orchestrator/skills.js';
import { pythonBridge } from '../../src/orchestrator/bridge.js';
import { getLLMClient } from '../../src/llm/client.js';

const HERE = new URL('.', import.meta.url).pathname;
const cases = readFileSync(`${HERE}../datasets/memory_utility.jsonl`, 'utf8')
  .trim().split('\n').filter(Boolean).map((l) => JSON.parse(l));
const llm = getLLMClient();
const TODAY = new Date().toISOString().slice(0, 10);

const has = (o: any, want: Record<string, unknown>) => Object.entries(want).every(([k, v]) => o?.[k] === v);
const lacks = (o: any, keys: string[]) => keys.every((k) => o?.[k] === undefined);

/**
 * The EFFECTIVE filter across the trace, not the model's requested args.
 *
 * Asserting on `args` was wrong and hid the very bug this eval found: `args` is only what the
 * model asked for, while a constraint seeded from memory is merged in inside executeTool and
 * never appears there. So a memory could be working perfectly and still look like it changed
 * nothing. `effectiveFilter` is what actually ran.
 */
function argsOf(trace: Array<{ args?: Record<string, unknown>; effectiveFilter?: Record<string, unknown> }>) {
  const out: Record<string, unknown> = {};
  for (const s of trace) {
    for (const [k, v] of Object.entries(s.effectiveFilter ?? s.args ?? {})) {
      if (out[k] === undefined) out[k] = v;
    }
  }
  return out;
}

const rows = [];
for (const c of cases) {
  let withObs: any, withoutObs: any;

  if (c.layer === 'facts') {
    let p = freshProfile('eval');
    for (const t of c.history) p = learnFromFilter(p, t.stated as SearchFilter);
    // Production order: remembered values are the last resort, the turn's own filter goes on top.
    const stated = (c.task.match(/San Diego/) ? { city: 'San Diego' } : {}) as SearchFilter;
    withObs = mergeFilter(preferredFilter(p) as SearchFilter, stated);
    withoutObs = mergeFilter({} as SearchFilter, stated);
  } else {
    let p = freshProfile('eval');
    for (const m of c.memories) {
      p = addMemory(p, { name: m.name, description: m.description, type: m.type, content: m.content,
        salience: m.salience, slots: m.slots });
    }
    const picked = await selectMemories(p.memories, c.task, llm);
    const registry = buildRegistry(pythonBridge, new InMemoryDraftStore());
    const common = { userId: 'eval-mem', registry, llm, store: new InMemoryAgentRunStore(), progressive: false, maxSteps: 4 };
    // seedFilterFor is the SAME function production uses — rebuilding the expression here is how
    // this runner kept measuring the old wiring after entry.ts changed.
    const a = await runAgent(c.task, { ...common, profileHint: profileHint(p, picked), seedFilter: seedFilterFor(p, picked) });
    const b = await runAgent(c.task, { ...common });
    withObs = argsOf(a.trace);
    withoutObs = argsOf(b.trace);
  }

  const checks: Record<string, boolean> = {};
  for (const [arm, obs] of [['with', withObs], ['without', withoutObs]] as const) {
    const exp = c.expect[arm];
    if (!exp) continue;
    if (exp.filter_has) checks[`${arm}_has`] = has(obs, exp.filter_has);
    if (exp.filter_lacks) checks[`${arm}_lacks`] = lacks(obs, exp.filter_lacks);
    if (exp.tool_args_contain) checks[`${arm}_args`] = has(obs, exp.tool_args_contain);
    if (exp.tool_args_lack) checks[`${arm}_no_args`] = lacks(obs, exp.tool_args_lack);
  }
  const differs = JSON.stringify(withObs) !== JSON.stringify(withoutObs);
  const isControl = !!c.meta?.note?.startsWith('negative control');
  // For a positive case, "no difference" means the memory did nothing — the null result this
  // eval exists to expose. For a negative control, no difference is the CORRECT answer.
  checks.differential = isControl ? !differs : differs;

  const ok = Object.values(checks).every(Boolean);
  rows.push({ id: c.id, layer: c.layer, control: isControl, differs, with: withObs, without: withoutObs, checks, ok });
  console.log(`  ${ok ? '✓' : '✗'} ${c.id} [${c.layer}${isControl ? '/control' : ''}] differs=${differs}`
    + `  ${Object.entries(checks).map(([k, v]) => `${v ? '✓' : '✗'}${k}`).join(' ')}`);
  if (!ok) { console.log(`      with   =${JSON.stringify(withObs)}`); console.log(`      without=${JSON.stringify(withoutObs)}`); }
}

const pos = rows.filter((r) => !r.control);
const ctl = rows.filter((r) => r.control);
const OUT = `${HERE}../history`;
mkdirSync(OUT, { recursive: true });
writeFileSync(`${OUT}/memory_utility.metrics.json`, JSON.stringify({
  n: rows.length, passed: rows.filter((r) => r.ok).length,
  pass_rate: Number((rows.filter((r) => r.ok).length / rows.length).toFixed(4)),
  // The headline: on how many positive cases did memory actually change the outcome?
  utility_rate: Number((pos.filter((r) => r.differs).length / pos.length).toFixed(4)),
  controls_held: ctl.every((r) => r.ok), llm_live: llm.available, rows,
}, null, 2));
console.log(`\nutility: ${rows.filter((r) => r.ok).length}/${rows.length} passed`
  + ` · 记忆真正改变了结果的正例比例 = ${pos.filter((r) => r.differs).length}/${pos.length}`
  + ` · 负对照 ${ctl.filter((r) => r.ok).length}/${ctl.length}`);
process.exit(0);
