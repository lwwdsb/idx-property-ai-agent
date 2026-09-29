/**
 * Deterministic intent classification (Week 9) — rules + reserved LLM slot.
 *
 * Rules cover the clear cases cheaply; when they're ambiguous an LLM would decide
 * (reserved via parseQuery's llm option), but with no key the safe move is to ask
 * (Q6: don't guess). Param extraction is merged in by reusing parseQuery.
 */
import { parseQuery } from '../search/parseQuery.js';
import { isKnownCity } from '../search/cityDictionary.js';
import { structuralCount, type SearchFilter } from '../search/filters.js';
import type { LLMClient } from '../llm/client.js';
import type { IntentGuess } from './bridge.js';
// config/tuning.json -> deterministic.intent (auto mode never routes through here).
import { EMBED_THRESHOLD, EMBED_MARGIN, FIELD_AUDIT, FIELD_AUDIT_THRESHOLD } from '../tuning.js';
import { systemOne, jevAvailable } from '../llm/jev.js';
import { logger } from '../logger.js';

/**
 * 路由目标。
 *
 * 'compound' 曾经在这里,而它是错的:它把【搜索+估价】这一个特定组合硬编码成一个类,所以三个意图
 * 或别的组合都表达不了。已拆成两样:'validate' 是一个真实的、独立的诉求("这套值不值这个价"),
 * 'multi' 不是一个意图而是【计划器跑了多个技能】这个事实的标签 —— 具体是哪几个由 skill 字段说。
 */
export type Intent = 'search' | 'market' | 'recommend' | 'knowledge' | 'validate' | 'email' | 'unknown'
  | 'multi';

export interface Classification {
  intent: Intent;
  confidence: 'high' | 'low';
  filter: SearchFilter;
  clarification?: string;
  /** A city was named but we don't serve it. Surfaced so callers know the city slot is an
   * explicit (rejected) choice rather than a blank — a stored preference must not fill it. */
  rejectedCity?: string;
  /** How the intent was decided (for logging/debugging). */
  via?: 'rule' | 'embedding';
}

export interface ClassifyOptions {
  llm?: LLMClient;
  /** Embedding-based intent classifier (warm service), used when rules are unsure. */
  classify?: (message: string) => Promise<IntentGuess>;
}

/** Embedding-classifier acceptance gate (both swept on the intent eval set, not pitched):
 * accept its guess only if top1 score >= EMBED_THRESHOLD AND the top1-top2 margin is
 * decisive (>= EMBED_MARGIN). The margin catches out-of-domain inputs that score
 * moderately high on some intent but are "half-like" several — a single score threshold
 * can't separate those because in/out scores overlap. Below either -> unknown/clarify. */
const ROUTABLE = new Set<Intent>(['search', 'market', 'recommend', 'knowledge', 'email']);

export const MARKET_RE = /\b(market|median|average price|avg price|price per|per sq\.?\s?ft|per square foot|trend|appreciat|going up|going down|good time to buy|worth buying)\b|行情|均价|中位|每平尺|每平方|走势|趋势|房价|涨|跌|升值|贬值|涨幅|跌幅|成交怎么样|最近成交/i;
export const RECOMMEND_RE = /\b(similar|recommend|comparable|like this|more like|anything like|like the (first|second|third|\d+))\b|类似|相似|推荐|像这套|差不多的/i;
export const KNOWLEDGE_RE = /\b(what is|what's|what does|how (is|are|do)|explain|define|definition|meaning|stand for)\b|什么是|怎么算|怎么计算|如何计算|什么意思|定义|表示什么|哪个字段|哪个列/i;
export const EMAIL_RE = /\be-?mail\b|发邮件|发送邮件|邮件发给|[^@\s]+@[^@\s]+\.[^@\s]+/i;
export const VALUE_RE = /\b(priced? (fair|right|well)|worth it|good deal|overpriced|underpriced|fair price|is it worth)\b|贵不贵|值不值|合理吗|价格合理|划算/i;

/**
 * Domain anchor for the two rules that match on a bare INTENT VERB rather than on
 * real-estate vocabulary. KNOWLEDGE_RE fires on any "what is / how do" question and
 * RECOMMEND_RE on a bare "recommend / 推荐", so without an anchor they swallowed
 * "what's the capital of France", "how are you doing today", "recommend a good restaurant
 * near me", "给我推荐一部电影" — 7 of the 8 out-of-domain queries that the rule layer was
 * routing before the embedding OOD gate ever saw them, which is what capped OOD rejection
 * at 0.771 no matter how the gate was tuned.
 *
 * REF counts as an anchor on purpose: "more like #2", "跟第一个类似的", "similar to the
 * first" carry no property noun, but a reference to previously shown listings IS a
 * real-estate signal.
 *
 * Deliberately NOT applied to `cityAgnostic` below: that flag only decides whether to spend
 * an LLM parse, and making out-of-domain messages non-city-agnostic would buy an LLM call
 * plus the risk of a hallucinated filter for no benefit — they fall to the gate either way.
 */
const DOMAIN_RE = /\b(?:homes?|houses?|propert(?:y|ies)|listings?|condos?|townhouses?|apartments?|real ?estate|mls|dom|days on market|sold[- ]to[- ]list|comps?|contingen(?:t|cy|cies)|pending|escrow|price per|per sq\.?\s?ft|square (?:foot|feet)|sqft|bed(?:room)?s?|bath(?:room)?s?|yard|pool|garage|zip|neighborhood|school district|hoa)\b|房|套|户|居室|卧|卫|平米|平尺|学区|房源|房产|成交|挂牌|字段|列名/i;
const REF_RE = /#\s*\d+|\b(?:first|second|third|fourth|fifth|1st|2nd|3rd)\b|\bthat one\b|\bthis one\b|第\s*[一二两三四五六七八九十\d]|这套|那套|这个|刚才|\b\d{6,}\b/i;
/** True when the message mentions the domain at all (property vocabulary or a listing reference).
 *
 * MEASUREMENT-ONLY ESCAPE HATCH. Setting IDX_NO_DOMAIN_ANCHOR=1 makes this always true, which
 * un-does the anchor tightening that took out-of-domain rejection from 0.63 to 0.80. It exists
 * because that tightening is the thing a typed classifier would replace: the question "can we STOP
 * adding regex rules and let a classifier own the tail" cannot be measured while the rules are
 * still catching the tail. Never set it in production — it deliberately makes the router greedier.
 */
export function hasDomainAnchor(message: string): boolean {
  if (process.env.IDX_NO_DOMAIN_ANCHOR === '1') return true;
  return DOMAIN_RE.test(message) || REF_RE.test(message);
}

/** 审计的字段 -> 问句。只问结构化槽位;city 也在内,因为"说了城市但没抽到"和别的字段同性质。 */
const AUDIT_FIELDS: Array<[keyof SearchFilter, string]> = [
  ['maxPrice', 'Does this message state a budget or an upper price limit?'],
  ['minPrice', 'Does this message state a MINIMUM price or a lower bound?'],
  ['beds', 'Does this message state a number of bedrooms?'],
  ['baths', 'Does this message state a number of bathrooms?'],
  ['propertyType', 'Does this message state a property type (condo, townhouse, single-family)?'],
  ['pool', 'Does this message state a pool requirement (wants one, or does not want one)?'],
  ['city', 'Does this message name a city or area to search in?'],
];

/**
 * 有没有【说了但没抽到】的字段 —— 这是升级 LLM 解析的新触发条件。
 *
 * 旧规则只有一个条件:城市缺失。于是"预算说法奇怪但有城市"永远不会升级,预算静默丢失,而下游
 * 无法区分"用户没提"和"说了没抽到" —— 后者用默认值填就是在覆盖用户刚说的话。
 *
 * 实测:审计召回在每个字段上都是 1.000,正则漏抽 11 处全部抓到;触发的升级比旧规则还少
 * (10/40 vs 12/40),零漏判。失败时返回空数组 = 退回旧规则,不会更糟。
 */
async function missedFields(message: string, got: SearchFilter): Promise<Array<keyof SearchFilter>> {
  if (!FIELD_AUDIT || !jevAvailable()) return [];
  const blank = AUDIT_FIELDS.filter(([k]) => got[k] == null);
  if (!blank.length) return [];                       // 全抽到了 -> 没什么可审计的
  try {
    const r = await systemOne(message, {
      nouls: Object.fromEntries(blank.map(([k, q]) => [String(k), { instructions: q }])),
    });
    return blank.filter(([k]) => (r.nouls[String(k)] ?? 0) >= FIELD_AUDIT_THRESHOLD).map(([k]) => k);
  } catch {
    return [];                                         // 乙: 审计失败 -> 退回旧规则
  }
}

export async function classifyIntent(message: string, opts: ClassifyOptions = {}): Promise<Classification> {
  // Cheap regex-only parse first (no LLM). City-agnostic intents (email / knowledge /
  // recommend) are decided from this alone — they don't need a city, so we never pay for
  // an LLM parse just to "recover a missing city" for them. Only a search/market-type
  // message that still lacks a city escalates to the LLM.
  const regexParsed = await parseQuery(message, { isKnownCity });
  const cityAgnostic = EMAIL_RE.test(message)
    || (KNOWLEDGE_RE.test(message) && !regexParsed.filter.city)
    || RECOMMEND_RE.test(message);
  // 升级到 LLM 解析的触发条件有两个(第二个是 2026-09-29 加的):
  //   ① 城市缺失且意图不是城市无关的 —— 旧规则
  //   ② 审计说某个字段【说了】而正则没抽到 —— 新规则,覆盖"有城市但别的字段漏了"这一整类
  const missed = await missedFields(message, regexParsed.filter);
  const needLlm = (!cityAgnostic && !regexParsed.filter.city) || missed.length > 0;
  const parsed = (needLlm && opts.llm?.available)
    ? await parseQuery(message, { llm: opts.llm, isKnownCity })
    : regexParsed;
  if (missed.length) {
    logger.debug('field audit: stated but not extracted', { missed, escalated: Boolean(opts.llm?.available) });
  }
  const searchable = Boolean(parsed.filter.city);
  const value = VALUE_RE.test(message);

  // email: drafting an outbound email (keyword or a recipient address present).
  // Checked early so "email the Irvine report to x@y.com" isn't taken as a search.
  if (EMAIL_RE.test(message)) {
    return { intent: 'email', confidence: 'high', filter: parsed.filter, via: 'rule' };
  }
  // knowledge (definitional, no city) — checked before market so "what is days on
  // MARKET" isn't misread as a market-stats query by the substring "market".
  if (KNOWLEDGE_RE.test(message) && !searchable && hasDomainAnchor(message)) {
    return { intent: 'knowledge', confidence: 'high', filter: parsed.filter, via: 'rule' };
  }
  // explicit market ask
  if (MARKET_RE.test(message)) {
    if (parsed.filter.city) return { intent: 'market', confidence: 'high', filter: parsed.filter, via: 'rule' };
    return { intent: 'market', confidence: 'low', filter: parsed.filter, via: 'rule',
             clarification: 'Which city do you want market stats for?' };
  }
  // recommendation
  if (RECOMMEND_RE.test(message) && hasDomainAnchor(message)) {
    return { intent: 'recommend', confidence: 'high', filter: parsed.filter, via: 'rule' };
  }

  // 估价 —— 放在 recommend 【之后】。"similar to the first and are they overpriced" 里
  // "overpriced" 会命中估价词,但它主要是个推荐诉求;推荐先判就不会被抢走。
  // 估价:"这套值不值这个价"。以前这里是 `searchable && value -> compound`,把【搜索+估价】
  // 这一个特定组合硬编码成一个伪类 —— 三个意图或别的组合都表达不了。现在估价是一个独立技能,
  // 而"搜索 + 估价"由元数闸门判成多意图、交给计划器拆,所以这里只需要认出【纯估价】的情形。
  if (value && !searchable) {
    return { intent: 'validate', confidence: 'high', filter: parsed.filter, via: 'rule' };
  }
  // plain search
  if (searchable) {
    return { intent: 'search', confidence: 'high', filter: parsed.filter };
  }
  // some real (structural) constraints, OR a named-but-unserveable city -> it's a search
  // that needs a valid city. Ask (with the specific reason) instead of falling to a guess.
  // Only STRUCTURAL fields count here: a bare keyword residue (e.g. an LLM parse hallucinating
  // "joke" on an out-of-domain message) must NOT masquerade as a search — it falls through to
  // the embedding OOD gate below and is rejected as unknown.
  if (structuralCount(parsed.filter) > 0 || parsed.rejectedCity) {
    return { intent: 'search', confidence: 'low', filter: parsed.filter, via: 'rule',
             rejectedCity: parsed.rejectedCity,
             clarification: parsed.clarification ?? 'Which city are you looking in?' };
  }

  // rules unsure -> embedding classifier (warm service), then clarify as the floor
  if (opts.classify) {
    try {
      const guess = await opts.classify(message);
      if (guess.score >= EMBED_THRESHOLD && guess.margin >= EMBED_MARGIN && ROUTABLE.has(guess.skill as Intent)) {
        return { intent: guess.skill as Intent, confidence: 'high', filter: parsed.filter, via: 'embedding' };
      }
    } catch { /* 乙: classifier down -> fall through to clarify */ }
  }
  return { intent: 'unknown', confidence: 'low', filter: parsed.filter, via: 'rule',
           clarification: "I can search listings, give market stats, recommend similar homes, or answer real-estate questions — what would you like?" };
}
