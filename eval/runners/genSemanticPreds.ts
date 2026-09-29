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
 * 2026-09-28: 现在会翻译中文残余,因为生产会。之前这里写的是"No LLM:纯确定性剥离",而生产在残余
 * 含中文时会调一次 LLM 翻成英文(语料是英文的)。少了这一步,这个预计算文件测的就又是一条生产不走的
 * 路 —— 和"把原始 query 直接送 /search"是同一个错误,只是换了个位置。
 *
 * 影响面不小:生成集 56% 是中文,而 known-item 的分层召回是 英文 0.80 / 中文 0.21。
 * 两个字段都记下来(raw 与 semantic),这样翻译到底改了什么是可见的,而不是一个黑盒差异。
 *
 *   npx tsx eval/runners/genSemanticPreds.ts
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { extractSemanticText } from '../../src/orchestrator/skills.js';
import { translateSemantic, hasCJK } from '../../src/search/multiQuery.js';
import { getLLMClient } from '../../src/llm/client.js';
import type { SearchFilter } from '../../src/search/filters.js';

// 两个集都算:人工 39 条与生成 171 条。known-item 评测两个都跑,只给生成集算等于把人工那一半
// 留在近似路径上。
const SETS: Array<[string, string]> = [
  ['eval/datasets/mode_retrieval.jsonl', 'eval/history/mode_retrieval.semantic.jsonl'],
  ['eval/datasets/mode_retrieval_gen.jsonl', 'eval/history/mode_retrieval_gen.preds.jsonl'],
];

const llm = getLLMClient();
if (!llm.available) console.log('⚠️  没有 LLM_API_KEY —— 中文残余会被丢空(生产的降级行为),不是翻译后的结果');
for (const [SRC, OUT] of SETS) {
const rows = readFileSync(SRC, 'utf8').trim().split('\n').filter(Boolean).map((l) => JSON.parse(l));
let empty = 0, translated = 0, dropped = 0;
const preds = [];
for (const r of rows) {
  const filter = (r.gold?.filter ?? {}) as SearchFilter;
  const raw = extractSemanticText(r.input, filter);
  let semantic = raw;
  if (hasCJK(raw)) {
    semantic = await translateSemantic(raw, llm);
    if (semantic) translated += 1; else dropped += 1;
  }
  if (!semantic) empty += 1;
  preds.push({ id: r.id, lang: r.lang, gold_filter: filter, raw, semantic });
}
writeFileSync(OUT, preds.map((p) => JSON.stringify(p)).join('\n') + '\n');
console.log(`${preds.length} rows -> ${OUT}`);
console.log(`  中文残余翻译成功 ${translated} 条 · 翻不了被丢空 ${dropped} 条`);
console.log(`  最终空残余(生产里不会进 Qdrant,走结构化): ${empty}`);
for (const p of preds.filter((x) => x.raw !== x.semantic).slice(0, 4)) {
  console.log(`  ${p.id} [${p.lang}] ${JSON.stringify(p.raw)} -> ${JSON.stringify(p.semantic)}`);
}
}
