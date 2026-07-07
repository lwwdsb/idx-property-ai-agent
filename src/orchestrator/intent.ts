/**
 * Deterministic intent classification (Week 9) — rules + reserved LLM slot.
 *
 * Rules cover the clear cases cheaply; when they're ambiguous an LLM would decide
 * (reserved via parseQuery's llm option), but with no key the safe move is to ask
 * (Q6: don't guess). Param extraction is merged in by reusing parseQuery.
 */
import { parseQuery } from '../search/parseQuery.js';
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

/** Min cosine for the embedding classifier to be trusted; below -> clarify. */
const EMBED_THRESHOLD = 0.55;
const ROUTABLE = new Set<Intent>(['search', 'market', 'recommend', 'knowledge', 'email']);

const MARKET_RE = /\b(market|median|average price|avg price|price per|per sq\.?\s?ft|per square foot|trend|appreciat)\b|行情|均价|中位|每平尺|每平方|走势|趋势/i;
const RECOMMEND_RE = /\b(similar|recommend|comparable|like this|more like)\b|类似|相似|推荐|像这套|差不多的/i;
const KNOWLEDGE_RE = /\b(what is|what's|what does|how (is|are|do)|explain|define|definition|meaning|stand for)\b|什么是|怎么算|怎么计算|如何计算|什么意思|定义|表示什么/i;
const EMAIL_RE = /\be-?mail\b|发邮件|发送邮件|邮件发给|[^@\s]+@[^@\s]+\.[^@\s]+/i;
const VALUE_RE = /\b(priced? (fair|right|well)|worth it|good deal|overpriced|underpriced|fair price|is it worth)\b|贵不贵|值不值|合理吗|价格合理|划算/i;

export async function classifyIntent(message: string, opts: ClassifyOptions = {}): Promise<Classification> {
  const parsed = await parseQuery(message, { llm: opts.llm });
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
  // some constraints but no city -> it's a search that needs a city
  if (filledCount(parsed.filter) > 0) {
    return { intent: 'search', confidence: 'low', filter: parsed.filter, via: 'rule',
             clarification: parsed.clarification ?? 'Which city are you looking in?' };
  }

  // rules unsure -> embedding classifier (warm service), then clarify as the floor
  if (opts.classify) {
    try {
      const guess = await opts.classify(message);
      if (guess.score >= EMBED_THRESHOLD && ROUTABLE.has(guess.skill as Intent)) {
        return { intent: guess.skill as Intent, confidence: 'high', filter: parsed.filter, via: 'embedding' };
      }
    } catch { /* 乙: classifier down -> fall through to clarify */ }
  }
  return { intent: 'unknown', confidence: 'low', filter: parsed.filter, via: 'rule',
           clarification: "I can search listings, give market stats, recommend similar homes, or answer real-estate questions — what would you like?" };
}
