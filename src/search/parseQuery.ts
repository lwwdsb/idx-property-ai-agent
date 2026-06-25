/**
 * NL query -> structured SearchFilter (Week 2).
 *
 * Pipeline (per design decisions):
 *  1. Regex fast-path — cheap, no LLM (Q13 budget; 乙 fallback).
 *  2. If the required slot (city) is still missing AND an LLM is available,
 *     escalate to the LLM and merge.
 *  3. Confidence/clarification come from STRUCTURAL signals (missing city,
 *     nothing parsed) — never from a model's self-reported confidence (Q6).
 *
 * Multi-turn (Q12): pass the session's current filter as `base`; each turn's
 * parse is applied as a patch (set/replace), so "actually 4 beds" overrides
 * "3 beds" instead of accumulating a contradictory AND.
 */
import { regexParse } from './regexParse.js';
import { mergeFilter, filledCount, type SearchFilter } from './filters.js';
import type { LLMClient } from '../llm/client.js';

export interface ParseResult {
  filter: SearchFilter;
  source: 'regex' | 'llm' | 'base';
  confidence: 'high' | 'low';
  /** Set when a structural signal says we should ask before searching. */
  clarification?: string;
}

export interface ParseOptions {
  /** Current session filter for multi-turn refinement. */
  base?: SearchFilter;
  /** Injected LLM client (omit/unavailable => regex-only). */
  llm?: LLMClient;
}

const ASK_CITY = 'Which city are you looking in?';
const ASK_ANYTHING = "Tell me a city and what you're looking for — e.g. beds, budget, or property type.";

export async function parseQuery(query: string, opts: ParseOptions = {}): Promise<ParseResult> {
  const base = opts.base ?? {};

  // 1. regex fast-path
  const regex = regexParse(query);
  let filter = mergeFilter(base, regex);
  let source: ParseResult['source'] = filledCount(regex) > 0 ? 'regex' : 'base';

  // 2. escalate to LLM only if we still lack the required slot (city) and it's available
  if (!filter.city && opts.llm?.available) {
    try {
      const llmPatch = await opts.llm.parseFilters(query);
      filter = mergeFilter(filter, llmPatch);
      if (Object.keys(llmPatch).length > 0) source = 'llm';
    } catch {
      // 乙: LLM failure must not break parsing — keep the regex/base result.
    }
  }

  // 3. structural confidence + clarification
  if (filter.city) {
    return { filter, source, confidence: 'high' };
  }
  const clarification = filledCount(filter) > 0 ? ASK_CITY : ASK_ANYTHING;
  return { filter, source, confidence: 'low', clarification };
}
