/**
 * Deterministic intent classification (Week 9) — rules + reserved LLM slot.
 *
 * Rules cover the clear cases cheaply; when they're ambiguous an LLM would decide
 * (reserved via parseQuery's llm option), but with no key the safe move is to ask
 * (Q6: don't guess). Param extraction is merged in by reusing parseQuery.
 */
import { parseQuery } from '../search/parseQuery.js';
import { isKnownCity } from '../search/cityDictionary.js';
import { filledCount, type SearchFilter } from '../search/filters.js';
import type { LLMClient } from '../llm/client.js';
import type { IntentGuess } from './bridge.js';

export type Intent = 'search' | 'market' | 'recommend' | 'knowledge' | 'compound' | 'email' | 'unknown';

export interface Classification {
  intent: Intent;
  confidence: 'high' | 'low';
  filter: SearchFilter;
  clarification?: string;
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
const EMBED_THRESHOLD = 0.58;
const EMBED_MARGIN = 0.05;
const ROUTABLE = new Set<Intent>(['search', 'market', 'recommend', 'knowledge', 'email']);

export const MARKET_RE = /\b(market|median|average price|avg price|price per|per sq\.?\s?ft|per square foot|trend|appreciat|going up|going down|good time to buy|worth buying)\b|行情|均价|中位|每平尺|每平方|走势|趋势|房价|涨|跌|升值|贬值|涨幅|跌幅|成交怎么样|最近成交/i;
export const RECOMMEND_RE = /\b(similar|recommend|comparable|like this|more like|anything like|like the (first|second|third|\d+))\b|类似|相似|推荐|像这套|差不多的/i;
export const KNOWLEDGE_RE = /\b(what is|what's|what does|how (is|are|do)|explain|define|definition|meaning|stand for)\b|什么是|怎么算|怎么计算|如何计算|什么意思|定义|表示什么|哪个字段|哪个列/i;
export const EMAIL_RE = /\be-?mail\b|发邮件|发送邮件|邮件发给|[^@\s]+@[^@\s]+\.[^@\s]+/i;
const VALUE_RE = /\b(priced? (fair|right|well)|worth it|good deal|overpriced|underpriced|fair price|is it worth)\b|贵不贵|值不值|合理吗|价格合理|划算/i;

export async function classifyIntent(message: string, opts: ClassifyOptions = {}): Promise<Classification> {
  // Cheap regex-only parse first (no LLM). City-agnostic intents (email / knowledge /
  // recommend) are decided from this alone — they don't need a city, so we never pay for
  // an LLM parse just to "recover a missing city" for them. Only a search/market-type
  // message that still lacks a city escalates to the LLM.
  const regexParsed = await parseQuery(message, { isKnownCity });
  const cityAgnostic = EMAIL_RE.test(message)
    || (KNOWLEDGE_RE.test(message) && !regexParsed.filter.city)
    || RECOMMEND_RE.test(message);
  const parsed = (!cityAgnostic && !regexParsed.filter.city && opts.llm?.available)
    ? await parseQuery(message, { llm: opts.llm, isKnownCity })
    : regexParsed;
  const searchable = Boolean(parsed.filter.city);
  const value = VALUE_RE.test(message);

  // email: drafting an outbound email (keyword or a recipient address present).
  // Checked early so "email the Irvine report to x@y.com" isn't taken as a search.
  if (EMAIL_RE.test(message)) {
    return { intent: 'email', confidence: 'high', filter: parsed.filter, via: 'rule' };
  }
  // compound: a real search + a "is it priced fairly" question -> recipe
  if (searchable && value) {
    return { intent: 'compound', confidence: 'high', filter: parsed.filter };
  }
  // knowledge (definitional, no city) — checked before market so "what is days on
  // MARKET" isn't misread as a market-stats query by the substring "market".
  if (KNOWLEDGE_RE.test(message) && !searchable) {
    return { intent: 'knowledge', confidence: 'high', filter: parsed.filter, via: 'rule' };
  }
  // explicit market ask
  if (MARKET_RE.test(message)) {
    if (parsed.filter.city) return { intent: 'market', confidence: 'high', filter: parsed.filter, via: 'rule' };
    return { intent: 'market', confidence: 'low', filter: parsed.filter, via: 'rule',
             clarification: 'Which city do you want market stats for?' };
  }
  // recommendation
  if (RECOMMEND_RE.test(message)) {
    return { intent: 'recommend', confidence: 'high', filter: parsed.filter, via: 'rule' };
  }
  // plain search
  if (searchable) {
    return { intent: 'search', confidence: 'high', filter: parsed.filter };
  }
  // some constraints, OR a named-but-unserveable city -> it's a search that needs a
  // valid city. Ask (with the specific reason) instead of falling to a guess.
  if (filledCount(parsed.filter) > 0 || parsed.rejectedCity) {
    return { intent: 'search', confidence: 'low', filter: parsed.filter, via: 'rule',
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
