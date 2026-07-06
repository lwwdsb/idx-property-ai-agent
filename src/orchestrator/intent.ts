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

export type Intent = 'search' | 'market' | 'recommend' | 'knowledge' | 'compound' | 'unknown';

export interface Classification {
  intent: Intent;
  confidence: 'high' | 'low';
  filter: SearchFilter;
  clarification?: string;
}

const MARKET_RE = /\b(market|median|average price|avg price|price per|per sq\.?\s?ft|per square foot|trend|appreciat)\b|行情|均价|中位|每平尺|每平方|走势|趋势/i;
const RECOMMEND_RE = /\b(similar|recommend|comparable|like this|more like)\b|类似|相似|推荐|像这套|差不多的/i;
const KNOWLEDGE_RE = /\b(what is|what's|what does|how (is|are|do)|explain|define|definition|meaning|stand for)\b|什么是|怎么算|怎么计算|如何计算|什么意思|定义|表示什么/i;
const VALUE_RE = /\b(priced? (fair|right|well)|worth it|good deal|overpriced|underpriced|fair price|is it worth)\b|贵不贵|值不值|合理吗|价格合理|划算/i;

export async function classifyIntent(message: string, llm?: LLMClient): Promise<Classification> {
  const parsed = await parseQuery(message, { llm });
  const searchable = Boolean(parsed.filter.city);
  const value = VALUE_RE.test(message);

  // compound: a real search + a "is it priced fairly" question -> recipe
  if (searchable && value) {
    return { intent: 'compound', confidence: 'high', filter: parsed.filter };
  }
  // explicit market ask
  if (MARKET_RE.test(message)) {
    if (parsed.filter.city) return { intent: 'market', confidence: 'high', filter: parsed.filter };
    return { intent: 'market', confidence: 'low', filter: parsed.filter,
             clarification: 'Which city do you want market stats for?' };
  }
  // recommendation
  if (RECOMMEND_RE.test(message)) {
    return { intent: 'recommend', confidence: 'high', filter: parsed.filter };
  }
  // knowledge (definitional) — but only when it isn't actually a search
  if (KNOWLEDGE_RE.test(message) && !searchable) {
    return { intent: 'knowledge', confidence: 'high', filter: parsed.filter };
  }
  // plain search
  if (searchable) {
    return { intent: 'search', confidence: 'high', filter: parsed.filter };
  }
  // some constraints but no city -> it's a search that needs a city
  if (filledCount(parsed.filter) > 0) {
    return { intent: 'search', confidence: 'low', filter: parsed.filter,
             clarification: parsed.clarification ?? 'Which city are you looking in?' };
  }
  // nothing matched (LLM would disambiguate here; no key -> ask)
  return { intent: 'unknown', confidence: 'low', filter: parsed.filter,
           clarification: "I can search listings, give market stats, recommend similar homes, or answer real-estate questions — what would you like?" };
}
