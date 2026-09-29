/**
 * 第三条臂:Jev 在 auto 的"选哪个工具"这一步上,对比 LLM 的 function calling。
 *
 * 为什么能直接对比:`mode_retrieval.jsonl` 44 条同时标了 `gold.intents`,而 auto 臂
 * (`auto_tools`)与确定性臂(`regex_intent`)已经在同一批 query 上测过。这里只往同一个预测文件
 * 里补一列 `jev_tools`,由同一个报告脚本用同一套指标算分 —— 换臂不换尺子。
 *
 * 公平性上刻意做的两件事:
 *   同样的工具描述  Choice 的选项描述直接取注册表里的 skill description,也就是 LLM 作为
 *                   function tool 描述看到的同一段文字。否则比的是 prompt 措辞,不是模型。
 *   同样的域外表达  auto 用"不调任何工具"表示域外,所以 Choice 多给一个显式的 none 选项,
 *                   让两条臂有同一种说"都不适用"的方式。
 *
 * 另外免费搭一个多标签变体:`gold.intents` 是列表,而 Choice 严格单选,所以 compound 用例它
 * 结构上最多只能答对一个。同一个请求里可以并行问多个问题、不额外增加往返,所以顺带对每个工具
 * 各问一个 noul("这个任务需要它吗"),按阈值得到一个集合 —— 报告里作为 `jev_multi` 单独看。
 * 这一列的存在本身就是个诚实性检查:如果单选臂的失分几乎都在 compound 上,那不是能力差距。
 *
 *   npx tsx eval/runners/evalJevIntent.ts            # 需要 TYPESAFE_API_KEY
 *   npx tsx eval/runners/evalJevIntent.ts --dry       # 不调用,只打印会发出去的请求
 */
import '../../src/testEnv.js';
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { systemOne, toolChoiceSpec, jevAvailable, JEV_NONE } from '../../src/llm/jev.js';
import { buildRegistry } from '../../src/orchestrator/skills.js';
import type { PythonBridge } from '../../src/search/pythonBridge.js';
import type { DraftStore } from '../../src/email/draftStore.js';

const HERE = new URL('.', import.meta.url).pathname;
const DATASET = `${HERE}../datasets/mode_retrieval.jsonl`;
const PREDS = `${HERE}../history/mode_retrieval.preds.jsonl`;
const DRY = process.argv.includes('--dry');
/** 多标签阈值:noul 是 0-1 的信念值,文档没有校准保证,所以 0.5 只是起点,后面按数据调。 */
const NOUL_T = Number(process.env.JEV_NOUL_THRESHOLD || 0.5);

const cases = readFileSync(DATASET, 'utf8').trim().split('\n').filter(Boolean).map((l) => JSON.parse(l));
const registry = buildRegistry({} as PythonBridge, {} as DraftStore);
const tools = registry.list().map((s) => ({ name: s.name, description: s.description }));
const choiceSpec = toolChoiceSpec(tools);
const nouls = Object.fromEntries(tools.map((t) => [
  `need_${t.name}`,
  { instructions: `Does this request require the "${t.name}" tool? ${t.description}` },
]));

if (DRY) {
  console.log(`工具选项(${tools.length + 1} 个,含显式 none):`);
  for (const [k, v] of Object.entries(choiceSpec.criteria)) console.log(`  ${k}: ${String(v).slice(0, 78)}`);
  console.log(`\n并行的 noul 问题:${Object.keys(nouls).length} 个`);
  console.log(`用例:${cases.length} 条 · 每条 1 个请求(1 个 choice + ${Object.keys(nouls).length} 个 noul)`);
  console.log(`\n示例 state:${JSON.stringify(cases[0].input)}`);
  process.exit(0);
}
if (!jevAvailable()) {
  console.error('TYPESAFE_API_KEY 未设置 —— 先到 console.typesafe.ai 拿 key 并写进 .env。');
  console.error('想先看会发什么请求:npx tsx eval/runners/evalJevIntent.ts --dry');
  process.exit(1);
}

// 预测文件按 id 合并而不是覆盖:另外两条臂的结果必须原样留着,否则就没得比了。
const existing = new Map<string, Record<string, unknown>>();
if (existsSync(PREDS)) {
  for (const l of readFileSync(PREDS, 'utf8').trim().split('\n').filter(Boolean)) {
    const r = JSON.parse(l) as { id: string };
    existing.set(r.id, r as Record<string, unknown>);
  }
}

let inTok = 0, outTok = 0, failed = 0;
const t0 = Date.now();
const lat: number[] = [];

for (const c of cases) {
  const row = existing.get(c.id) ?? { id: c.id };
  try {
    const s = Date.now();
    const r = await systemOne(c.input, { choices: { tool: choiceSpec }, nouls });
    lat.push(Date.now() - s);
    inTok += r.usage.inputTokens; outTok += r.usage.outputTokens;
    const a = r.choices.tool!;
    // none => 空数组,和 auto 臂"不调任何工具"表示域外的方式一致。
    row.jev_tools = a.choice === JEV_NONE ? [] : [a.choice];
    row.jev_confidence = a.confidence;
    row.jev_margin = a.margin;
    row.jev_probabilities = a.probabilities;
    row.jev_multi = tools.map((t) => t.name).filter((n) => (r.nouls[`need_${n}`] ?? 0) >= NOUL_T);
    row.jev_nouls = r.nouls;
    console.log(`  ${c.id}  choice=${a.choice.padEnd(10)} conf=${a.confidence.toFixed(3)} `
      + `margin=${a.margin.toFixed(3)}  multi=[${(row.jev_multi as string[]).join(',')}]  gold=[${c.gold.intents.join(',')}]`);
  } catch (e) {
    failed += 1;
    row.jev_error = String(e).slice(0, 160);
    console.log(`  ${c.id}  ✗ ${row.jev_error}`);
  }
  existing.set(c.id, row);
}

writeFileSync(PREDS, [...existing.values()].map((r) => JSON.stringify(r)).join('\n') + '\n');
const p = (q: number) => (lat.length ? lat.sort((a, b) => a - b)[Math.min(lat.length - 1, Math.round(q / 100 * (lat.length - 1)))] : 0);
const cost = inTok / 1e6 * 0.042;   // 输出免费
console.log(`\njev 臂写入 ${cases.length} 条(失败 ${failed})`);
console.log(`  延迟 p50 ${p(50)}ms · p95 ${p(95)}ms · max ${p(100)}ms · 总耗时 ${((Date.now() - t0) / 1000).toFixed(1)}s`);
console.log(`  token 输入 ${inTok} / 输出 ${outTok}(免费) · 本次成本 ≈ $${cost.toFixed(5)}`);
console.log(`\n跑报告看三臂对比:python3 eval/runners/report_mode_intent.py`);
