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
  /** A city was named but isn't one we can serve — a strong "this is a search" signal. */
  rejectedCity?: string;
}

export interface ParseOptions {
  /** Current session filter for multi-turn refinement. */
  base?: SearchFilter;
  /** Injected LLM client (omit/unavailable => regex-only). */
  llm?: LLMClient;
  /** Reject a parsed "city" that isn't a real, serveable city (injected; omit to skip). */
  isKnownCity?: (city: string) => boolean | Promise<boolean>;
}

const ASK_CITY = 'Which city are you looking in?';
const ASK_ANYTHING = "Tell me a city and what you're looking for — e.g. beds, budget, or property type.";

// Negation / exclusion markers the regex handles poorly — a risk signal that we may
// have MIS-parsed (false positive), so we escalate to the LLM even if regex "succeeded".
const RISK_WORDS = /\b(no|not|without|except|excluding)\b|不要|不带|不含|不需要|没有|除了|不是/i;

async function cityValid(city: string | undefined, check?: ParseOptions['isKnownCity']): Promise<boolean> {
  if (!city) return false;
  if (!check) return true;          // no validator injected -> accept (backward compatible)
  return Boolean(await check(city));
}

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

  // 1b. city sanity check: a regex-extracted city that isn't real is a false positive
  //     our under-parse signal can't catch. Drop it so we escalate / clarify honestly.
  let rejectedCity: string | undefined;
  if (filter.city && !(await cityValid(filter.city, opts.isKnownCity))) {
    rejectedCity = filter.city;
    delete filter.city;
  }

  // 2. escalate to LLM when available AND we may have under- OR mis-parsed:
  //    missing/invalid city, a substantial query barely captured, OR a risk word
  //    (negation) the regex handles poorly.
  const shouldEscalate = !filter.city || looksUnderParsed(normalized, regex) || RISK_WORDS.test(normalized);
  if (shouldEscalate && opts.llm?.available) {
    try {
      const llmPatch = await opts.llm.parseFilters(normalized);
      filter = mergeFilter(filter, llmPatch);
      if (Object.keys(llmPatch).length > 0) source = 'llm';
      // validate the LLM's city too — don't blindly trust either parser.
      if (filter.city && !(await cityValid(filter.city, opts.isKnownCity))) {
        rejectedCity = filter.city;
        delete filter.city;
      } else if (filter.city) {
        rejectedCity = undefined;
      }
    } catch {
      // 乙: LLM failure must not break parsing — keep the regex/base result.
    }
  }

  // 3. structural confidence + clarification
  if (filter.city) {
    return { filter, source, confidence: 'high' };
  }
  const clarification = rejectedCity
    ? `I don't have listings in "${rejectedCity}". Which city should I search?`
    : filledCount(filter) > 0 ? ASK_CITY : ASK_ANYTHING;
  return { filter, source, confidence: 'low', clarification, rejectedCity };
}
