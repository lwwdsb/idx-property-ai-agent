/**
 * Provider-agnostic chat LLM client (OpenAI-compatible /chat/completions).
 * Works with OpenAI, MiniMax, or any compatible endpoint by changing
 * LLM_BASE_URL / LLM_API_KEY / LLM_MODEL in .env — no code change.
 *
 * If no API key is configured, `available` is false and callers fall back to
 * the regex fast-path (Week-11 乙 degradation: every LLM step has a non-LLM path).
 */
import { config } from '../config.js';
import { logger } from '../logger.js';
import { FILTER_KEYS, type FilterPatch, type SearchFilter } from '../search/filters.js';

export interface LLMClient {
  readonly available: boolean;
  parseFilters(query: string): Promise<FilterPatch>;
}

const SYSTEM_PROMPT = [
  'You extract real-estate search filters from a user message (English or Chinese).',
  'Return ONLY a JSON object using any of these keys, omitting ones you are unsure about:',
  '- city: string',
  '- beds: number (minimum bedrooms)',
  '- baths: number (minimum bathrooms)',
  '- maxPrice: number (USD)',
  '- minPrice: number (USD)',
  '- propertyType: one of "condo" | "townhouse" | "single-family"',
  '- pool: boolean',
  '- minSqft: number',
  'Note: Chinese 万 = 10,000 (e.g. 150万 = 1500000). No prose, JSON only.',
].join('\n');

const TYPES = new Set(['condo', 'townhouse', 'single-family']);

/** Keep only known keys with valid types; drop everything else. */
export function sanitizeFilter(raw: unknown): FilterPatch {
  const out: FilterPatch = {};
  if (!raw || typeof raw !== 'object') return out;
  const o = raw as Record<string, unknown>;
  for (const k of FILTER_KEYS) {
    const v = o[k];
    if (v == null) continue;
    if (k === 'city' && typeof v === 'string' && v.trim()) out.city = v.trim();
    else if (k === 'propertyType' && typeof v === 'string' && TYPES.has(v)) out.propertyType = v;
    else if (k === 'pool' && typeof v === 'boolean') out.pool = v;
    else if (['beds', 'baths', 'maxPrice', 'minPrice', 'minSqft'].includes(k)) {
      const n = typeof v === 'number' ? v : Number(v);
      if (Number.isFinite(n) && n >= 0) (out as Record<string, unknown>)[k] = n;
    }
  }
  return out;
}

/** Tolerant JSON extraction (some models wrap JSON in prose/fences). */
function extractJson(text: string): unknown {
  try { return JSON.parse(text); } catch { /* fall through */ }
  const start = text.indexOf('{');
  const end = text.lastIndexOf('}');
  if (start >= 0 && end > start) {
    try { return JSON.parse(text.slice(start, end + 1)); } catch { /* ignore */ }
  }
  return {};
}

export function getLLMClient(): LLMClient {
  const { baseUrl, apiKey, model } = config.llm;
  if (!apiKey) {
    return {
      available: false,
      async parseFilters(): Promise<FilterPatch> {
        throw new Error('LLM not configured (set LLM_API_KEY)');
      },
    };
  }
  return {
    available: true,
    async parseFilters(query: string): Promise<FilterPatch> {
      const res = await fetch(`${baseUrl.replace(/\/$/, '')}/chat/completions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${apiKey}` },
        body: JSON.stringify({
          model,
          temperature: 0,
          messages: [
            { role: 'system', content: SYSTEM_PROMPT },
            { role: 'user', content: query },
          ],
          response_format: { type: 'json_object' },
        }),
      });
      if (!res.ok) {
        throw new Error(`LLM HTTP ${res.status}: ${(await res.text()).slice(0, 200)}`);
      }
      const data = (await res.json()) as { choices?: Array<{ message?: { content?: string } }> };
      const content = data.choices?.[0]?.message?.content ?? '{}';
      const filter = sanitizeFilter(extractJson(content));
      logger.debug('llm parsed filters', { query, filter });
      return filter;
    },
  };
}

export type { SearchFilter };
