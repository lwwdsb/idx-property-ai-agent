/**
 * Tunable algorithm parameters — the SINGLE source for every knob a sweep may change.
 *
 * Deliberately separate from `config.ts`: that one holds environment/deployment values
 * (DB creds, ports, keys) written by a human once; this one holds algorithm parameters
 * that the tuning loop rewrites programmatically. Different writers, different lifecycles
 * — the same reason the user profile is split per writer.
 *
 * Keys are grouped by BLAST RADIUS, so a sweep knows which eval suites to run from the
 * config structure alone rather than from someone remembering:
 *   shared        — both paths (auto reuses the same SkillRegistry, so skill-level knobs
 *                   are shared by construction). Changing these needs BOTH eval suites.
 *   deterministic — only the deterministic router (auto decides tools via function
 *                   calling and never touches intent classification).
 *   auto          — only the ReAct loop. NOT swept automatically: 14-task eval set and
 *                   p99 15.4s means variance swamps the effect size.
 *
 * Override the file for one run (how a sweep tests a candidate without touching the
 * checked-in defaults):  IDX_TUNING=/abs/path/candidate.json npm run ...
 */
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

export interface Tuning {
  version: number;
  shared: {
    retrieval: { rerankCoarse: number; prefetch: number; rerankEnabled: boolean; topK: number };
    search: { maxResults: number; tooMany: number };
    rag: { chunkSize: number; chunkOverlap: number; topK: number };
  };
  deterministic: { intent: { embedThreshold: number; embedMargin: number;
    arity: 'regex' | 'jev'; arityThreshold: number;
    fieldAudit: boolean; fieldAuditThreshold: number } };
  auto: {
    loop: { maxSteps: number; maxPerTool: number; progressive: boolean; intentSelector: 'llm' | 'jev' };
    memory: { selectSemantic: number; selectEpisodic: number; fallbackSemantic: number; fallbackEpisodic: number };
  };
}

/** Resolved from THIS file's location, not cwd — services start from different dirs. */
export const TUNING_PATH = process.env.IDX_TUNING?.trim()
  || join(dirname(fileURLToPath(import.meta.url)), '..', 'config', 'tuning.json');

function load(): Tuning {
  // No try/catch on purpose: a missing or malformed tuning file must fail loudly at
  // startup. Silently falling back to hardcoded defaults would make a sweep compare
  // two runs that secretly used the same parameters.
  return JSON.parse(readFileSync(TUNING_PATH, 'utf8')) as Tuning;
}

export const tuning: Tuning = load();

// Named exports so call sites read the same as the constants they replaced.
export const RERANK_COARSE = tuning.shared.retrieval.rerankCoarse;
export const PREFETCH = tuning.shared.retrieval.prefetch;
export const SEARCH_TOP_K = tuning.shared.retrieval.topK;
export const MAX_RESULTS = tuning.shared.search.maxResults;
export const TOO_MANY = tuning.shared.search.tooMany;
export const EMBED_THRESHOLD = tuning.deterministic.intent.embedThreshold;
export const EMBED_MARGIN = tuning.deterministic.intent.embedMargin;
/**
 * 谁来判断"这条请求是一件事还是好几件事"。
 *
 *   regex  detectMultiIntent —— 数有几个意图关键词命中。实测 precision 0.500 / recall 0.545
 *          (intent.jsonl) 与 1.000 / 0.667 (mode_retrieval),是整个意图层最差的数。它的失效方式
 *          是结构性的:关键词落在【从属位置】("email the market report" 里的 market 是宾语而不是
 *          第二个诉求)会假多;第二个诉求换了说法("顺便看看贵不贵"不含行情词)会假单。
 *   jev    一个类型化的是非题。实测 0.900/0.818 与 1.000/1.000,而且在后者上多意图全部落在
 *          0.88~0.95、单意图全部落在 0.04~0.18,间隙 0.70 —— 阈值放 0.2~0.8 结果完全相同。
 *
 * 阈值偏低而不是偏高,因为两类错误代价不对称:漏判(多意图判成单)意味着用户问了两件事只答一件
 * 而且【无声】,不可恢复;误判只是多花一次 LLM 调用,而 planSkills 若认定其实是单意图会返回 null
 * 回落单路由,能自愈。
 */
// 枚举值在加载时校验。曾经被写成 '"jev"'(多一层引号),于是 ARITY !== 'jev' 成立、静默回落正则,
// 而评测跑出来的数字和正则一模一样 —— 那个"完全一样"是唯一的破绽。配置里的非法枚举必须炸,
// 不能悄悄变成"用默认行为",否则一次扫描会比较两个其实相同的配置。
const ARITY_VALUES = ['regex', 'jev'] as const;
if (!(ARITY_VALUES as readonly string[]).includes(tuning.deterministic.intent.arity)) {
  throw new Error(`config/tuning.json: deterministic.intent.arity must be one of `
    + `${ARITY_VALUES.join(' | ')}, got ${JSON.stringify(tuning.deterministic.intent.arity)}`);
}
export const ARITY = tuning.deterministic.intent.arity;
export const ARITY_THRESHOLD = tuning.deterministic.intent.arityThreshold;
/**
 * 字段审计:问"这句话【说了】哪些字段",而不是"抽到了哪些"。
 *
 * 它修的是系统分不清的两种"空":用户没提 vs 说了但正则没抽到 —— 两者在代码里都是 undefined,
 * 于是第二种情况下默认值/记忆值看起来是软默认、实际在覆盖用户刚说的话。实测证据:
 * "在 Irvine 找个三居,预算三百出头" 正则只抽到 {city, beds},预算静默丢失。
 *
 * 现在的升级规则只有一个条件(城市缺失),所以"预算说法奇怪但有城市"永远不会升级。
 *
 * 实测(parse.jsonl 40 条):审计召回在每个字段上都是 1.000,正则漏抽 11 处全部抓到;而且它触发的
 * 升级【比现在的规则更少】(阈值 0.7 时 10/40,现规则 12/40),零漏判。阈值取 0.7 而不是更严的
 * 0.9,是因为 40 条太小,不该贴着边走 —— 代价只是多 1 次白升级,而白升级是便宜的错误(LLM 解析
 * 只会确认正则已有的结果)。
 */
export const FIELD_AUDIT = tuning.deterministic.intent.fieldAudit;
export const FIELD_AUDIT_THRESHOLD = tuning.deterministic.intent.fieldAuditThreshold;
export const MAX_STEPS = tuning.auto.loop.maxSteps;
export const MAX_PER_TOOL = tuning.auto.loop.maxPerTool;
/**
 * Progressive tool loading: expose only `find_tools` up front and let the agent enable what it
 * needs, instead of preloading every tool schema into every turn.
 *
 * It lives HERE, and not as a caller-supplied flag, for two reasons found by measuring:
 *
 * ONE DEFAULT, NOT THREE. `entry.ts` defaulted it to true, `loop.ts` to false, and the agent eval
 * hardcoded false — so the eval had never measured the configuration production actually runs.
 * A knob with a different default at each layer is not a knob, it is three.
 *
 * SWEEPABLE. The objective's A/A guard keys on the sha of this file: two arms that differ only in
 * a flag passed at the call site would hash identically, and a real difference between them would
 * be reported as INVALID. Putting the flag in the config is what makes the arms distinguishable.
 */
export const PROGRESSIVE = tuning.auto.loop.progressive;
/**
 * Who decides WHICH tool an auto-mode task needs.
 *
 *   llm   the model's own function calling — it picks the tool AND writes the arguments in one call
 *   jev   an enumerated typed decision (TypeSafe's Choice) picks the tool; the arguments still have
 *         to come from somewhere, because that model gives up string generation entirely
 *
 * It lives in the config rather than at a call site for the same reason progressive does: the
 * objective's A/A guard keys on this file's hash, so two arms that differ only in a flag passed by
 * the caller would hash identically and a real difference between them would be reported as
 * INVALID. It is also why the eval must read it from here instead of hardcoding an arm.
 *
 * Currently 'llm'. Flipping it is not a one-line change — see eval/runners/evalJevIntent.ts for
 * what has to be measured first, and note that the decision is only half of function calling.
 */
export const INTENT_SELECTOR = tuning.auto.loop.intentSelector;
/**
 * How many memories may be INJECTED, capped per type. Two caps, not one, because the cap does a
 * completely different job on each of the two selection paths — measured 2026-09-28 on the 13-case
 * memory_select set at (3,1) / (5,3) / (8,5), three runs each:
 *
 *   LLM path       precision 1.000 and over-selections 0 at every cap, so the cap never removes a
 *                  wrong pick — it only ever clips a CORRECT one. Recall rose 0.671 -> 0.697 and
 *                  exact 0.538 -> 0.615 going from (5,3) to (8,5), at a cost of ~14 tokens per
 *                  admitted memory, i.e. about 2 tokens per request. So this cap should be LOOSE:
 *                  the selector's own precision is the real control here.
 *   FALLBACK path  no LLM, ranks by salience x recency x frequency and fills the cap regardless of
 *                  the task, so the cap is the ONLY thing bounding misinjection. Over-selections
 *                  went 26 -> 35 -> 39 across the same three settings, with probe precision 0.000
 *                  throughout (it injects on greetings and out-of-domain requests too). So this cap
 *                  should be TIGHT.
 *
 * One number cannot satisfy both, and it was previously loose where it needed to be tight.
 */
export const SELECT_SEMANTIC = tuning.auto.memory.selectSemantic;
export const SELECT_EPISODIC = tuning.auto.memory.selectEpisodic;
export const FALLBACK_SEMANTIC = tuning.auto.memory.fallbackSemantic;
export const FALLBACK_EPISODIC = tuning.auto.memory.fallbackEpisodic;
