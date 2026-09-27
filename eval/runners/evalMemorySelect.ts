/**
 * Memory SELECTION eval — 注入准确率 / 召回准确率.
 *
 * `selectMemories` is a retrieval problem with objective gold: given a profile's memories and
 * the current task, a human can say which ones are relevant. So this needs no LLM judge —
 * the labels are human, the metrics are IR metrics, and the numbers can gate.
 *
 * TWO ARMS, scored on the same cases, which is the point:
 *   llm       — the real selector (one LLM call over name+description)
 *   fallback  — what runs when no LLM is available: salience x recency x frequency
 * The fallback IGNORES THE TASK ENTIRELY, so it is the honest floor. If the LLM arm does not
 * clearly beat it, the extra call is not earning its cost — nobody had asked that question.
 *
 * Three cases are probes rather than ordinary examples:
 *   ms-005 / ms-009  gold is EMPTY (greeting, out-of-domain). Any selection is a mis-injection;
 *                    these only move precision, and a selector that always returns something
 *                    cannot score well on them.
 *   ms-010           the current turn overrides a remembered budget, so the stale memory must
 *                    NOT be selected — the "soft default never hijacks the current request"
 *                    property, measured instead of asserted in a comment.
 *
 * Writes eval/history/memory_select.preds.jsonl for report_memory_select.py.
 *   npx tsx eval/runners/evalMemorySelect.ts
 */
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { selectMemories, profileHint, freshProfile, type MemoryEntry } from '../../src/memory/profile.js';
import { getLLMClient } from '../../src/llm/client.js';

const HERE = new URL('.', import.meta.url).pathname;
const DATA = `${HERE}../datasets`;
const OUT = `${HERE}../history`;
const TODAY = new Date().toISOString().slice(0, 10);

interface Case {
  id: string; task: string;
  profile: { memories: Array<Partial<MemoryEntry> & { name: string; description: string; type: 'semantic' | 'episodic'; content: string }> };
  gold: { relevant: string[]; reason?: string };
  meta?: { note?: string };
}

/** Fill the fields the store would have set, so the ranker sees a complete entry. */
function hydrate(m: Case['profile']['memories'][number]): MemoryEntry {
  return {
    name: m.name, description: m.description, type: m.type, content: m.content,
    createdAt: m.createdAt ?? TODAY, updatedAt: m.updatedAt ?? TODAY, lastUsed: m.lastUsed ?? TODAY,
    useCount: m.useCount ?? 0, salience: m.salience ?? 0.5,
    sourceRuns: m.sourceRuns ?? [], mergedFrom: m.mergedFrom ?? [],
  };
}

const cases: Case[] = readFileSync(`${DATA}/memory_select.jsonl`, 'utf8')
  .trim().split('\n').filter(Boolean).map((l) => JSON.parse(l));

const llm = getLLMClient();
if (!llm.available) console.log('WARNING: no LLM key — the llm arm will fall back, making the two arms identical');

const preds = [];
for (const c of cases) {
  const mems = c.profile.memories.map(hydrate);
  // The llm arm is the production path; the fallback arm is what runs with no key.
  const picked = await selectMemories(mems, c.task, llm);
  const fallback = await selectMemories(mems, c.task, undefined);
  // profileHint is what actually reaches the prompt — recorded so a selection that is right but
  // renders wrong cannot hide behind the selection metric.
  const prof = { ...freshProfile('eval'), memories: mems };
  preds.push({
    id: c.id, task: c.task, note: c.meta?.note ?? null,
    pool: mems.map((m) => m.name),
    gold: c.gold.relevant,
    llm: picked.map((m) => m.name),
    fallback: fallback.map((m) => m.name),
    hint_len: profileHint(prof, picked).length,
  });
  const p = preds[preds.length - 1]!;
  console.log(`  ${c.id} [${mems.length} in pool] gold=${JSON.stringify(p.gold)}`);
  console.log(`      llm      -> ${JSON.stringify(p.llm)}`);
  console.log(`      fallback -> ${JSON.stringify(p.fallback)}`);
}

mkdirSync(OUT, { recursive: true });
writeFileSync(`${OUT}/memory_select.preds.jsonl`, preds.map((p) => JSON.stringify(p)).join('\n') + '\n');
writeFileSync(`${OUT}/memory_select.meta.json`, JSON.stringify({
  n: cases.length, llm_live: llm.available, at: new Date().toISOString(),
}));
console.log(`\n${cases.length} cases -> eval/history/memory_select.preds.jsonl  [llm=${llm.available ? 'live' : 'off'}]`);
process.exit(0);
