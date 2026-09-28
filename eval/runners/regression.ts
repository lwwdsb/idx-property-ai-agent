/**
 * 回归池 —— 一票否决闸。
 *
 * 和主集(基准集)分开,因为这两件事结构上互斥:主集要【冻结】才有可比性,副池要【永远增长】
 * 才能装下不断发现的 bad case。本会话撞了两次证据:agent 集 14→21 作废了整条噪声带,
 * memory_select 10→13 让 recall 0.792→0.697 看起来像退化而其实什么都没退化。
 *
 * 所以职责严格切开:
 *   主集  产生【可比的数字】-> 回答"该不该改"(配对检验 + 噪声带)
 *   副池  产生【布尔】,永不求平均 -> 回答"能不能改"(坏了任何一条即否决)
 * 布尔没有分母,所以往副池加用例不可能让任何数字移动 —— 这就是它能随便长的原因。
 *
 * 两种 kind,因为"已经失败的东西没法再被弄坏":
 *   must_hold       系统【现在通过】。弄坏 -> 失败退出,候选不可采纳。
 *   documented_gap  系统【现在失败】(能力缺失),钉住防遗忘。它不能当否决闸;
 *                   但如果它【开始通过】了,那是好消息 -> 报告为 NEWS,不判失败。
 *
 * 入池门槛:先证明 gold 是对的,再钉。钉错的断言比不钉更糟 —— 它会永久否决正确的改动。
 * 本会话有三次"我以为是缺陷、其实是我的 gold 写错了"(mf-004 / a-019 / L4 散文解析),
 * 所以这一步不是形式主义。
 *
 *   npx tsx eval/runners/regression.ts
 */
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { evalDynamicsCase } from './lib/memoryDynamicsCase.js';

const HERE = new URL('.', import.meta.url).pathname;
const POOL = `${HERE}../datasets/regression/pool.jsonl`;
const OUT = `${HERE}../history`;
mkdirSync(OUT, { recursive: true });

const cases = readFileSync(POOL, 'utf8').trim().split('\n').filter(Boolean).map((l) => JSON.parse(l));

const rows: Array<Record<string, unknown>> = [];
let broken = 0, held = 0, gapsAsDocumented = 0, news = 0, unsupported = 0;

for (const c of cases) {
  let ok: boolean | null = null;
  let checks: Record<string, boolean> = {};
  if (c.domain === 'memory_dynamics') {
    const r = evalDynamicsCase(c);
    ok = r.ok; checks = r.checks;
  } else {
    unsupported += 1;
    rows.push({ id: c.id, kind: c.kind, domain: c.domain, status: 'unsupported' });
    console.log(`  ? ${c.id}  domain "${c.domain}" 还没有执行器 —— 计为未支持,不当通过`);
    continue;
  }

  let status: string;
  if (c.kind === 'must_hold') {
    status = ok ? 'held' : 'BROKEN';
    if (ok) held += 1; else broken += 1;
  } else {
    // documented_gap 断言的是【当前(有缺陷的)行为】。它成立 = 缺口还在,按文档运行;
    // 它不成立 = 行为变了,极可能是有人把缺口修好了 -> 这是新闻而不是失败。
    status = ok ? 'gap_as_documented' : 'NEWS(行为变了,可能已修好)';
    if (ok) gapsAsDocumented += 1; else news += 1;
  }
  rows.push({ id: c.id, kind: c.kind, domain: c.domain, status, checks,
              note: c.meta?.note ?? null, fixed_at: c.meta?.fixed_at ?? null });
  const mark = c.kind === 'must_hold' ? (ok ? '✓' : '✗') : (ok ? '·' : '!');
  console.log(`  ${mark} ${c.id} [${c.kind}]  ${status}`
    + `  ${Object.entries(checks).map(([k, v]) => `${v ? '✓' : '✗'}${k}`).join(' ')}`);
  if (c.kind === 'must_hold' && !ok) console.log(`      钉住的是: ${c.meta?.note ?? c.why ?? ''}`);
  if (c.kind === 'documented_gap' && !ok) console.log(`      缺口行为变了: ${c.meta?.note ?? ''}`);
}

const metrics = {
  n: cases.length,
  must_hold: { n: held + broken, held, broken },
  documented_gap: { n: gapsAsDocumented + news, as_documented: gapsAsDocumented, changed: news },
  unsupported,
  // 故意没有 pass_rate:副池永不求平均。一旦有了比率,加用例就会移动数字,
  // 而"加用例不移动数字"正是副池能自由增长的全部前提。
  veto: broken > 0 || unsupported > 0,
};
writeFileSync(`${OUT}/regression.metrics.json`, JSON.stringify(metrics, null, 1));

console.log(`\nregression pool: must_hold ${held}/${held + broken} 保持`
  + ` · documented_gap ${gapsAsDocumented}/${gapsAsDocumented + news} 按文档`
  + (news ? ` · ${news} 条缺口行为变了(查是不是修好了)` : '')
  + (unsupported ? ` · ${unsupported} 条未支持` : ''));
if (metrics.veto) {
  console.log('\nVETO:候选不可采纳 —— 弄坏了钉住的行为(或有未支持的用例没被真正检验)');
  process.exit(1);
}
console.log('\nVETO: none —— 可以进入主集比较"该不该改"');
