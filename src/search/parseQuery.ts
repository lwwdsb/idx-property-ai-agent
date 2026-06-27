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
import { normalizeQuery } from './normalize.js';
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

/** Rough content-word count: latin words + CJK characters. */
function contentLength(s: string): number {
  return (s.match(/[A-Za-z]+/g)?.length ?? 0) + (s.match(/[一-鿿]/g)?.length ?? 0);
}

/**
 * Heuristic: the user wrote something substantial but regex barely captured it,
 * so structured constraints were likely missed (phrasing the regex can't match).
 * This broadens LLM escalation beyond just "missing city".
 */
function looksUnderParsed(query: string, regex: SearchFilter): boolean {
  return contentLength(query) >= 6 && filledCount(regex) <= 1;
}

export async function parseQuery(query: string, opts: ParseOptions = {}): Promise<ParseResult> {
  const base = opts.base ?? {};

  // 0. deterministic normalization (full-width, spelled numbers, synonyms)
  const normalized = normalizeQuery(query);

  // 1. regex fast-path
  const regex = regexParse(normalized);
  let filter = mergeFilter(base, regex);
  let source: ParseResult['source'] = filledCount(regex) > 0 ? 'regex' : 'base';

  // 2. escalate to LLM when available AND we likely under-parsed:
  //    missing the required slot (city) OR a substantial query barely captured.
  const shouldEscalate = !filter.city || looksUnderParsed(normalized, regex);
  if (shouldEscalate && opts.llm?.available) {
    try {
      const llmPatch = await opts.llm.parseFilters(normalized);
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
