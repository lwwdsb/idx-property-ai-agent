/**
 * Jev 做意图识别 —— 在确定性路径的 118 条意图集上,对比现状。
 *
 * 为什么是这一面而不是 auto:auto 的工具选择已经命中 1.00、域外 5/5,上限只剩约 5%,而接进去要
 * 付"多一次往返"或"把一回合多工具逼成多回合"的代价(progressive 就是这么输的)。确定性这一面
 * 相反,有一个已测出来的空洞:
 *
 *   放开正则单测 embedding,118 条上 accuracy 0.576 / macro-F1 0.527,而且没有可用工作点 ——
 *   生产阈值下域内误拒 46%,把域内误拒压到 5% 则域外拒识掉到 0.457。原因在分布:score<0.58 只
 *   拦 2/35 域外,27/35 的域外分数落在域内取值范围内;全部工作是 margin 在做,而它同时拦掉
 *   38/83 域内。它的标签空间还只有 5 个技能,没有 unknown 也没有 compound。
 *
 * 也就是说:正则不是两层里的一层,它就是全部系统(118 条里 114 条由规则决定),后面没有能用的
 * 兜底。所以这里要验的不是"能不能赢过 0.90",而是【能不能补上那个不存在的兜底层】。
 *
 * Jev 在这件事上有个结构性优势:unknown 可以作为【显式选项】交给它选,把"弃权"从阈值问题变成
 * 分类问题 —— 正好绕开 embedding 栽在上面的那道坎(厂商也明说 confidence 没有校准保证)。
 *
 * 三种配置:
 *   all       118 条全交给 Jev。看它单独的天花板,和 embedding-only 那条臂直接可比
 *   residue   只接【规则没定案】的那些(现在是走到 embedding 门或落到最后 unknown 的用例)。
 *             这是我建议的生产形状:常见请求仍走 4ms 的正则,只有难例付网络代价
 *   无锚点     先用 IDX_NO_DOMAIN_ANCHOR=1 重跑 evalIntentParse,残余集会变大(实测域外拒识
 *             掉回 0.657),再跑 residue —— 验证"能不能停止再加正则规则"
 *
 *   npx tsx eval/runners/evalJevIntent.ts --dry
 *   npx tsx eval/runners/evalJevIntent.ts --config all
 *   npx tsx eval/runners/evalJevIntent.ts --config residue
 *   IDX_NO_DOMAIN_ANCHOR=1 npx tsx eval/runners/evalIntentParse.ts   # 先重算残余
 *   npx tsx eval/runners/evalJevIntent.ts --config residue --tag no-anchors
 */
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { systemOne, jevAvailable, type ChoiceSpec } from '../../src/llm/jev.js';

const HERE = new URL('.', import.meta.url).pathname;
const DATASET = `${HERE}../datasets/intent.jsonl`;
const PIPELINE_PREDS = `${HERE}../history/intent.preds.jsonl`;
const OUT = `${HERE}../history`;

const arg = (k: string, d?: string) => {
  const i = process.argv.indexOf(`--${k}`);
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1]! : d;
};
const CONFIG = arg('config', 'all')!;
const TAG = arg('tag', CONFIG)!;
const DRY = process.argv.includes('--dry');

/**
 * 选项描述。刻意写成"这个意图在本系统里是什么",而不是照着测试集的用例反推 —— 后者会把测试集
 * 的表达方式泄漏进 prompt,量出来的就不是泛化能力。compound 和 unknown 也给成显式选项,因为
 * 它们都是 gold 里的类,而 embedding 那一层结构上答不出它们。
 */
const INTENT_CRITERIA: Record<string, string> = {
  search: 'Find property listings matching criteria (city, bedrooms, budget, type, features).',
  market: 'City-level market statistics: median price, price per sqft, days on market, trend.',
  recommend: 'Given a listing the user already likes or referred to, find similar homes.',
  knowledge: 'Explain a real-estate term, metric or concept (what does DOM mean, how are comps computed).',
  email: 'Draft an outbound email to a recipient, e.g. send a report to a client address.',
  compound: 'BOTH a listing search AND a judgement about whether the price is fair — two things at once.',
  unknown: 'None of the above: small talk, another domain entirely, or something this '
    + 'real-estate assistant cannot do (mortgage math, buying a house for the user, jokes).',
};

const choiceSpec: ChoiceSpec = {
  instructions: 'What is the user asking this real-estate assistant to do? '
    + 'Pick "unknown" if the request is outside what it handles.',
  criteria: INTENT_CRITERIA,
};

const cases = readFileSync(DATASET, 'utf8').trim().split('\n').filter(Boolean).map((l) => JSON.parse(l));

/** 规则没定案的那些:走到 embedding 门被采纳(via='embedding'),或落到最后的 unknown。
 *  从流水线的预测文件里读,而不是在这里重新实现一遍判定 —— 两份实现会漂移。 */
function residueIds(): Set<string> {
  if (!existsSync(PIPELINE_PREDS)) {
    throw new Error(`没有 ${PIPELINE_PREDS} —— 先跑 npx tsx eval/runners/evalIntentParse.ts`);
  }
  const ids = new Set<string>();
  for (const l of readFileSync(PIPELINE_PREDS, 'utf8').trim().split('\n').filter(Boolean)) {
    const r = JSON.parse(l) as { id: string; via?: string; pred: string };
    if (r.via === 'embedding' || r.pred === 'unknown') ids.add(r.id);
  }
  return ids;
}

const selected = CONFIG === 'all' ? cases : (() => {
  const ids = residueIds();
  return cases.filter((c) => ids.has(c.id));
})();

if (DRY) {
  console.log(`配置 ${CONFIG} · tag ${TAG}`);
  console.log(`选项(${Object.keys(INTENT_CRITERIA).length} 个,含显式 compound 与 unknown):`);
  for (const [k, v] of Object.entries(INTENT_CRITERIA)) console.log(`  ${k.padEnd(10)} ${v.slice(0, 76)}`);
  console.log(`\n用例:${selected.length} / ${cases.length} 条`);
  if (CONFIG !== 'all') {
    const g = selected.filter((c) => c.label.intents.includes('unknown')).length;
    console.log(`  其中域外 ${g} 条 / 域内 ${selected.length - g} 条`);
  }
  console.log(`预估输入 token ≈ ${selected.length * 300} → 成本 ≈ $${(selected.length * 300 / 1e6 * 0.042).toFixed(5)}`);
  process.exit(0);
}
if (!jevAvailable()) {
  console.error('TYPESAFE_API_KEY 未设置 —— 到 console.typesafe.ai 拿 key 并写进 .env。');
  console.error('先看会发什么:npx tsx eval/runners/evalJevIntent.ts --dry --config ' + CONFIG);
  process.exit(1);
}

let inTok = 0, failed = 0;
const lat: number[] = [];
const rows: Array<Record<string, unknown>> = [];

for (const c of selected) {
  try {
    const s = Date.now();
    const r = await systemOne(c.input, { choices: { intent: choiceSpec } });
    lat.push(Date.now() - s);
    inTok += r.usage.inputTokens;
    const a = r.choices.intent!;
    const ok = c.label.intents.includes(a.choice);
    rows.push({ id: c.id, input: c.input, gold: c.label.intents, pred: a.choice,
      confidence: a.confidence, margin: a.margin, probabilities: a.probabilities, hit: ok });
    console.log(`  ${ok ? '✓' : '✗'} ${c.id}  ${a.choice.padEnd(10)} conf=${a.confidence.toFixed(3)} `
      + `margin=${a.margin.toFixed(3)}  gold=[${c.label.intents.join(',')}]`);
  } catch (e) {
    failed += 1;
    rows.push({ id: c.id, input: c.input, gold: c.label.intents, pred: 'error', error: String(e).slice(0, 160) });
    console.log(`  ✗ ${c.id}  ${String(e).slice(0, 100)}`);
  }
}

writeFileSync(`${OUT}/jev_intent.${TAG}.preds.jsonl`, rows.map((r) => JSON.stringify(r)).join('\n') + '\n');
const p = (q: number) => (lat.length ? [...lat].sort((a, b) => a - b)[Math.min(lat.length - 1, Math.round(q / 100 * (lat.length - 1)))] : 0);
console.log(`\n配置 ${CONFIG} · ${selected.length} 条(失败 ${failed})`);
console.log(`  延迟 p50 ${p(50)}ms · p95 ${p(95)}ms · max ${p(100)}ms`);
console.log(`  输入 token ${inTok} · 成本 ≈ $${(inTok / 1e6 * 0.042).toFixed(5)}(输出免费)`);
console.log(`\n算分:python3 eval/runners/report_jev_intent.py ${TAG}`);
