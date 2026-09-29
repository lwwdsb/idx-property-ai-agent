/**
 * 元数判定:这条请求是【一件事】还是【好几件事】。
 *
 * 为什么单独成一个 runner:新架构里这是一个【路由决策】而不是一个分类标签 ——
 *   一件事   -> 直接调单个工具(整句就是 query,不需要改写)
 *   好几件   -> 升级 LLM 一次调用,同时拆子 query 并抽参(planSkills)
 * 所以它的质量直接决定"会不会用户问了三件事只答了一件"。而两类错误的代价不对称:
 *   假单(漏判) 用户问了两件事只答一件,而且【无声】—— 不可恢复
 *   假多(误判) 多花一次 LLM 调用,而 planSkills 若认定其实是单意图会返回 null 回落 —— 自愈
 * 所以阈值应当偏向召回,而不是偏向精度。
 *
 * 现任是 detectMultiIntent(关键词计数):precision 0.500 / recall 0.545 —— 整个意图层最差的数。
 * 它的失效方式是结构性的:关键词出现在【从属位置】("email the market report" 里的 market 是
 * 宾语不是第二个诉求)会假多,第二个诉求换了说法("顺便看看贵不贵"不含行情词)会假单。
 *
 *   npx tsx eval/runners/evalJevArity.ts eval/datasets/mode_retrieval.jsonl
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { systemOne, jevAvailable } from '../../src/llm/jev.js';

const HERE = new URL('.', import.meta.url).pathname;
const src = process.argv[2] || 'eval/datasets/mode_retrieval.jsonl';
const path = src.startsWith('/') ? src : `${HERE}../../${src}`;
const tag = src.split('/').pop()!.replace('.jsonl', '');

const NOUL = { multi: { instructions: 'Does this request ask the assistant to do more than ONE '
  + 'distinct thing (e.g. find listings AND report market stats), as opposed to one request with '
  + 'several constraints?' } };

const cases = readFileSync(path, 'utf8').trim().split('\n').filter(Boolean).map((l) => JSON.parse(l));
if (!jevAvailable()) { console.error('TYPESAFE_API_KEY 未设置'); process.exit(1); }

const isMulti = (g: string[]) => g.length > 1 || g.includes('compound');
const rows: Array<Record<string, unknown>> = [];
let inTok = 0;
for (const c of cases) {
  const intents: string[] = c.gold?.intents ?? c.label?.intents ?? [];
  if (!intents.length) continue;
  const r = await systemOne(c.input, { nouls: NOUL });
  inTok += r.usage.inputTokens;
  const v = r.nouls.multi ?? 0;
  rows.push({ id: c.id, input: c.input, gold: intents, multi_gold: isMulti(intents), noul: v });
}
writeFileSync(`${HERE}../history/jev_arity.${tag}.preds.jsonl`, rows.map((r) => JSON.stringify(r)).join('\n') + '\n');

const pos = rows.filter((r) => r.multi_gold);
console.log(`\n元数判定 — ${tag} · ${rows.length} 条(多意图 ${pos.length} / 单意图 ${rows.length - pos.length})`);
console.log(`  ${'T'.padStart(5)}${'precision'.padStart(11)}${'recall'.padStart(9)}${'F1'.padStart(8)}${'漏'.padStart(5)}${'误'.padStart(5)}`);
for (const T of [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]) {
  const tp = rows.filter((r) => r.multi_gold && (r.noul as number) >= T).length;
  const fp = rows.filter((r) => !r.multi_gold && (r.noul as number) >= T).length;
  const fn = pos.length - tp;
  const p = tp + fp ? tp / (tp + fp) : 0, rc = pos.length ? tp / pos.length : 0;
  const f1 = p + rc ? (2 * p * rc) / (p + rc) : 0;
  console.log(`  ${T.toFixed(1).padStart(5)}${p.toFixed(3).padStart(11)}${rc.toFixed(3).padStart(9)}${f1.toFixed(3).padStart(8)}${String(fn).padStart(5)}${String(fp).padStart(5)}`);
}
console.log(`\n  输入 token ${inTok} · 成本 ≈ $${(inTok / 1e6 * 0.042).toFixed(5)}`);
