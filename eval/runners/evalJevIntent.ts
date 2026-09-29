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
 * 六个选项:五个技能 + 显式 unknown。
 *
 * 为什么没有 compound、也没有多标签。多意图在这个系统里【不是分类器的职责】:detectMultiIntent
 * 先判是否 >=2 个意图,然后 planSkills 用【一次】LLM 调用同时给出技能列表和每个技能的子 query。
 * 而拆子 query 是文本生成,Jev 结构上做不了 —— 所以把 Jev 插在多意图路径前面纯属多余:它给出
 * 集合,但生成子 query 的那次调用顺带就把技能说了,等于多一次往返、零收益。
 *
 * 反过来,这也界定了 Jev 该待的地方:【输出是标签而不是 query 的那些决定】。残余集 34 条里 32 条
 * 是域外探针,而域外的输出是一句拒答,永远不需要改写 —— 那正是现在这一层坏掉的地方
 * (embedding 单独跑:域内误拒 46%)。
 *
 * gold 里有 3 条标成 ['compound'],它们表达的其实是"搜索 + 估价"这个组合,和另外 8 条标成
 * {search, market} 的是同一类现象、两种编码 —— 因为标签跟着代码里那条 searchable && value 的
 * 特殊规则走了。六选项的 Choice 结构上答不出 compound,所以这 3 条必然算错。这是【gold 的
 * 不一致】而不是模型能力问题,报告里单独标出来,不擅自改标签。
 *
 * 描述刻意写成"这个意图在本系统里是什么",不照测试集用例反推 —— 否则会把测试集的表达方式泄漏
 * 进 prompt,量出来的就不是泛化能力。
 */
const INTENT_CRITERIA: Record<string, string> = {
  search: 'Find property listings matching criteria (city, bedrooms, budget, type, features).',
  market: 'City-level market statistics: median price, price per sqft, days on market, trend.',
  recommend: 'Given a listing the user already likes or referred to, find similar homes.',
  knowledge: 'Explain a real-estate term, metric or concept (what does DOM mean, how are comps computed).',
  email: 'Draft an outbound email to a recipient, e.g. send a report to a client address.',
  // compound 补回来【只为公平对比】:现状系统有一条规则能答它,给 Jev 更小的标签空间再比 macro-F1
  // 就不是同一把尺子 —— 首轮实测 8 条错里 3 条是 compound,单这一类为 0 就在 7 类里拖掉约 0.14。
  // 架构上我仍然认为它不该是一个类(它是"搜索+估价"这一个特定组合的伪类,三个意图就表达不了),
  // 但那是该不该改 gold 的问题,不该用"换一把更小的尺子"来回避。
  compound: 'BOTH a listing search AND a judgement about whether the price is fair — two things at once.',
  unknown: 'None of the above: small talk, another domain entirely, or something this '
    + 'real-estate assistant cannot do (mortgage math, buying a house for the user, jokes).',
};

/** 同一次请求里附带的两个是非题 —— 不额外增加往返,每个约 30 token。
 *  multi 对标现任元数判定(关键词计数,precision 0.500 / recall 0.545,整个意图层最差的数);
 *  ref 对标 REF_RE。两者都有现成 gold:gold.intents 是列表,指代类用例在域内集里。 */
const EXTRA_NOULS = {
  multi: { instructions: 'Does this request ask the assistant to do more than ONE distinct thing '
    + '(e.g. find listings AND report market stats), as opposed to one request with several constraints?' },
  ref: { instructions: 'Does this message refer back to a specific listing or result from earlier in '
    + 'the conversation (e.g. "that one", "the first", "这套", "上次看的")?' },
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
  const comp = selected.filter((c) => c.label.intents.includes('compound')).length;
  const multi = selected.filter((c) => c.label.intents.length > 1).length;
  if (comp) console.log(`  其中 ${comp} 条 gold 是 ['compound'] —— 六选项结构上答不出,必然算错(gold 不一致,见文件头)`);
  if (multi) console.log(`  其中 ${multi} 条是多意图(gold 为集合)—— Choice 答对其中任一个即计命中,与现有 accuracy_in_set 同口径`);
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
    const r = await systemOne(c.input, { choices: { intent: choiceSpec }, nouls: EXTRA_NOULS });
    lat.push(Date.now() - s);
    inTok += r.usage.inputTokens;
    const a = r.choices.intent!;
    const ok = c.label.intents.includes(a.choice);
    rows.push({ id: c.id, input: c.input, gold: c.label.intents, pred: a.choice,
      confidence: a.confidence, margin: a.margin, probabilities: a.probabilities, hit: ok,
      noul_multi: r.nouls.multi ?? null, noul_ref: r.nouls.ref ?? null });
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
