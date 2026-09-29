/**
 * 字段审计:这句话【说了】哪些字段 —— 而不是【抽到了】哪些。
 *
 * 为什么需要它。系统分不清字段为什么是空的:
 *   用户没提           -> 该用记忆/默认填,或直接不填
 *   说了但正则没抽到    -> 【不许】用记忆填(那是覆盖用户刚说的话),应当升级解析或追问
 * 两种情况在代码里都是 undefined。这是删掉事实记忆之后唯一还悬着的洞。
 *
 * 现在的升级规则只有一个条件:【城市缺失】。所以"预算说法奇怪但有城市"永远不会升级 —— 实测
 * "在 Irvine 找个三居,预算三百出头" 正则只抽到 {city, beds},预算静默丢失;而
 * "我能接受到三百出头" 更糟,被抽成 proximity:{to:"三百出头"},会把预算当地名送去 geocode。
 *
 * gold 是现成的:parse.jsonl 的 label.filter 就是"这句话说了哪些字段"。
 *
 *   npx tsx eval/runners/evalJevFields.ts
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { systemOne, jevAvailable } from '../../src/llm/jev.js';
import { parseQuery } from '../../src/search/parseQuery.js';

const HERE = new URL('.', import.meta.url).pathname;
const T = Number(process.env.JEV_FIELD_THRESHOLD || 0.5);
const KNOWN = new Set(['irvine', 'san diego', 'los angeles', 'tustin', 'pasadena', 'anaheim',
  'san jose', 'fullerton', 'diamond bar', 'san gabriel', 'beverly hills', 'long beach',
  'arcadia', 'cupertino'].map((s) => s.toLowerCase()));

const FIELDS: Record<string, { key: string; q: string }> = {
  f_maxPrice: { key: 'maxPrice', q: 'Does this message state a budget or an upper price limit?' },
  f_minPrice: { key: 'minPrice', q: 'Does this message state a MINIMUM price or a lower bound?' },
  f_beds: { key: 'beds', q: 'Does this message state a number of bedrooms?' },
  f_baths: { key: 'baths', q: 'Does this message state a number of bathrooms?' },
  f_propertyType: { key: 'propertyType', q: 'Does this message state a property type (condo, townhouse, single-family)?' },
  f_pool: { key: 'pool', q: 'Does this message state a pool requirement (wants one, or does not want one)?' },
  f_city: { key: 'city', q: 'Does this message name a city or area to search in?' },
};

const cases = readFileSync(`${HERE}../datasets/parse.jsonl`, 'utf8').trim().split('\n')
  .filter(Boolean).map((l) => JSON.parse(l));
if (!jevAvailable()) { console.error('TYPESAFE_API_KEY 未设置'); process.exit(1); }

const rows = [];
let inTok = 0;
for (const c of cases) {
  const gf = (c.label?.filter ?? {}) as Record<string, unknown>;
  const parsed = await parseQuery(c.input, { isKnownCity: (x: string) => KNOWN.has(x.trim().toLowerCase()) });
  const r = await systemOne(c.input, {
    nouls: Object.fromEntries(Object.entries(FIELDS).map(([id, f]) => [id, { instructions: f.q }])),
  });
  inTok += r.usage.inputTokens;
  rows.push({ id: c.id, input: c.input, gold: gf,
    regex: parsed.filter as Record<string, unknown>, nouls: r.nouls });
}
writeFileSync(`${HERE}../history/jev_fields.preds.jsonl`, rows.map((r) => JSON.stringify(r)).join('\n') + '\n');

console.log(`\n字段审计 — parse.jsonl ${rows.length} 条 · 阈值 ${T}\n`);
console.log(`  ${'字段'.padEnd(14)}${'说了(gold)'.padStart(11)}${'审计P'.padStart(8)}${'审计R'.padStart(8)}`
  + `${'正则漏抽'.padStart(10)}${'审计抓到'.padStart(10)}`);
let missTotal = 0, caughtTotal = 0;
for (const [id, f] of Object.entries(FIELDS)) {
  const stated = rows.filter((r) => r.gold[f.key] != null);
  if (!stated.length) continue;
  const tp = stated.filter((r) => (r.nouls[id] ?? 0) >= T).length;
  const fp = rows.filter((r) => r.gold[f.key] == null && (r.nouls[id] ?? 0) >= T).length;
  // 真正可行动的那个数:gold 说了、而正则没抽到 —— 这些就该升级,而审计能不能发现它们
  const missed = stated.filter((r) => r.regex[f.key] == null);
  const caught = missed.filter((r) => (r.nouls[id] ?? 0) >= T);
  missTotal += missed.length; caughtTotal += caught.length;
  const p = tp + fp ? tp / (tp + fp) : 0;
  console.log(`  ${f.key.padEnd(14)}${String(stated.length).padStart(11)}${p.toFixed(3).padStart(8)}`
    + `${(tp / stated.length).toFixed(3).padStart(8)}${String(missed.length).padStart(10)}${String(caught.length).padStart(10)}`);
}
console.log(`\n  【可行动】正则漏抽合计 ${missTotal} 处,审计抓到 ${caughtTotal} 处`
  + ` (${missTotal ? (caughtTotal / missTotal * 100).toFixed(0) : 0}%)`);
console.log(`  输入 token ${inTok} · 成本 ≈ $${(inTok / 1e6 * 0.042).toFixed(5)}`);
