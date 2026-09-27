/**
 * Precompute the PRODUCTION semantic text for the generated known-item set.
 *
 * loop_retrieval.py approximated it by deleting the gold city from the raw input. That was
 * wrong in the same way sending the raw query to /search was wrong: production runs
 * extractSemanticText, which normalizes first (so 尔湾 -> Irvine, 一百二十万 -> 120万, 三居 -> 3居)
 * and then strips the city, the numbers with their units, property-type and pool words, and
 * filler. On a Chinese query the difference is most of the string — and the generated set is
 * 56% Chinese, so a retrieval decision made on the approximation is a decision about input
 * production never sends.
 *
 * No LLM: the gold filter is ground truth, so this is pure deterministic stripping.
 *
 *   npx tsx eval/runners/genSemanticPreds.ts
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { extractSemanticText } from '../../src/orchestrator/skills.js';
import type { SearchFilter } from '../../src/search/filters.js';

const SRC = 'eval/datasets/mode_retrieval_gen.jsonl';
const OUT = 'eval/history/mode_retrieval_gen.preds.jsonl';

const rows = readFileSync(SRC, 'utf8').trim().split('\n').filter(Boolean).map((l) => JSON.parse(l));
let empty = 0;
const preds = rows.map((r) => {
  const filter = (r.gold?.filter ?? {}) as SearchFilter;
  const semantic = extractSemanticText(r.input, filter);
  if (!semantic) empty += 1;
  return { id: r.id, lang: r.lang, gold_filter: filter, semantic };
});
writeFileSync(OUT, preds.map((p) => JSON.stringify(p)).join('\n') + '\n');
console.log(`${preds.length} rows -> ${OUT}`);
console.log(`  empty semantic (never reaches Qdrant in production): ${empty}`);
for (const p of preds.slice(0, 4)) console.log(`  ${p.id} [${p.lang}] ${JSON.stringify(p.semantic)}`);
