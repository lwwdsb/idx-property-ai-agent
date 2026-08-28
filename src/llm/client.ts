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
import { withResilience, CircuitBreaker } from '../resilience/resilience.js';

// one breaker for the LLM provider (shared across all client instances/methods)
const llmBreaker = new CircuitBreaker(5, 20_000);

/** One step of a multi-skill plan: which skill + the self-contained sub-query for it. */
export interface PlanStep { skill: string; query: string; }

// ── Tool-calling (the auto/agent mode's ReAct loop) ────────────────────────────
/** An OpenAI-style function tool exposed to the agent loop. `parameters` is a JSON Schema. */
export interface ToolSpec { name: string; description: string; parameters: Record<string, unknown>; }
/** A message in a tool-calling conversation. `raw` preserves the provider's assistant
 * message verbatim so an assistant-with-tool_calls turn can be echoed back unchanged. */
export interface ChatMessage {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: string;
  tool_call_id?: string;      // set on a tool-result message
  tool_calls?: unknown[];     // set on an assistant message that requested tools
}
/** One tool the model wants called (arguments already JSON-parsed). */
export interface ToolCall { id: string; name: string; arguments: Record<string, unknown>; }
/** One assistant turn: a final answer (`content`) and/or a batch of tool calls.
 * `raw` is the provider message to push back verbatim before appending tool results. */
export interface ChatTurn { content: string; toolCalls: ToolCall[]; raw: ChatMessage; }

export interface LLMClient {
  readonly available: boolean;
  parseFilters(query: string): Promise<FilterPatch>;
  /** Plan an ordered set of skills + a per-skill sub-query (the LLM decomposes the
   * message so each skill only sees its own part). Optional — callers must fall back
   * deterministically when absent (乙). */
  planSkills?(message: string, skills: Array<{ name: string; description: string }>): Promise<PlanStep[]>;
  /** One turn of a tool-calling loop: send the running transcript + available tools,
   * get back either a final answer or tool calls to execute. Optional (auto mode only). */
  chatWithTools?(messages: ChatMessage[], tools: ToolSpec[]): Promise<ChatTurn>;
}

const PLAN_PROMPT = [
  'You are a task planner for a real-estate assistant.',
  'Given a user message and available skills, output the MINIMAL ordered set of skills',
  'needed to fully answer. Prefer ONE skill; use several ONLY when the message clearly',
  'asks for multiple things (e.g. find homes AND show the market).',
  'For EACH chosen skill, also give "query": the self-contained part of the message',
  'relevant to that skill only (keep its constraints; drop the other skills\' parts).',
  'Use only skill names from the list. Return JSON only:',
  '{"plan": [{"skill": "name", "query": "sub-query for this skill"}, ...]}',
].join('\n');

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
  '- proximity: object {to: string, withinMinutes?: number, mode?: "driving"|"transit"|"walking"}',
  '    — set ONLY if the user wants to be near/within a commute of a place',
  '      (e.g. "within 30 min of downtown LA", "距XX公司30分钟车程").',
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
    else if (k === 'proximity' && typeof v === 'object' && v !== null) {
      const p = v as Record<string, unknown>;
      if (typeof p.to === 'string' && p.to.trim()) {
        const prox: NonNullable<FilterPatch['proximity']> = { to: p.to.trim() };
        const mins = typeof p.withinMinutes === 'number' ? p.withinMinutes : Number(p.withinMinutes);
        if (Number.isFinite(mins) && mins > 0) prox.withinMinutes = mins;
        if (p.mode === 'driving' || p.mode === 'transit' || p.mode === 'walking') prox.mode = p.mode;
        out.proximity = prox;
      }
    }
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
  async function chatJSON(system: string, user: string): Promise<unknown> {
    return withResilience(async () => {
      const res = await fetch(`${baseUrl.replace(/\/$/, '')}/chat/completions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${apiKey}` },
        body: JSON.stringify({
          model,
          temperature: 0,
          messages: [
            { role: 'system', content: system },
            { role: 'user', content: user },
          ],
          response_format: { type: 'json_object' },
        }),
      });
      if (!res.ok) {
        throw new Error(`LLM HTTP ${res.status}: ${(await res.text()).slice(0, 200)}`);
      }
      const data = (await res.json()) as { choices?: Array<{ message?: { content?: string } }> };
      return extractJson(data.choices?.[0]?.message?.content ?? '{}');
    }, { name: 'llm/chatJSON', timeoutMs: 30_000, retries: 2, breaker: llmBreaker });
  }

  return {
    available: true,
    async parseFilters(query: string): Promise<FilterPatch> {
      const filter = sanitizeFilter(await chatJSON(SYSTEM_PROMPT, query));
      logger.debug('llm parsed filters', { query, filter });
      return filter;
    },
    async planSkills(message, skills): Promise<PlanStep[]> {
      const user = `Available skills:\n${skills.map((s) => `- ${s.name}: ${s.description}`).join('\n')}`
        + `\n\nUser message: ${message}`;
      const raw = await chatJSON(PLAN_PROMPT, user) as { plan?: unknown };
      const known = new Set(skills.map((s) => s.name));
      const items = Array.isArray(raw?.plan) ? raw.plan : [];
      // validate: only known skill names, de-duped, capped at 3 (no runaway plans);
      // a missing/blank sub-query falls back to the full message.
      const seen = new Set<string>();
      const out: PlanStep[] = [];
      for (const it of items) {
        const skill = (it as { skill?: unknown })?.skill;
        const query = (it as { query?: unknown })?.query;
        if (typeof skill === 'string' && known.has(skill) && !seen.has(skill)) {
          seen.add(skill);
          out.push({ skill, query: typeof query === 'string' && query.trim() ? query.trim() : message });
          if (out.length >= 3) break;
        }
      }
      logger.debug('llm plan', { message, plan: out });
      return out;
    },
    async chatWithTools(messages: ChatMessage[], tools: ToolSpec[]): Promise<ChatTurn> {
      return withResilience(async () => {
        const res = await fetch(`${baseUrl.replace(/\/$/, '')}/chat/completions`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${apiKey}` },
          body: JSON.stringify({
            model,
            temperature: 0,
            messages,
            tools: tools.map((t) => ({ type: 'function', function: t })),
            tool_choice: 'auto',
          }),
        });
        if (!res.ok) {
          throw new Error(`LLM HTTP ${res.status}: ${(await res.text()).slice(0, 200)}`);
        }
        const data = (await res.json()) as {
          choices?: Array<{ message?: { content?: string | null; tool_calls?: unknown[] } }>;
        };
        const msg = data.choices?.[0]?.message ?? { content: '' };
        const rawCalls = Array.isArray(msg.tool_calls) ? msg.tool_calls : [];
        const toolCalls: ToolCall[] = rawCalls.map((tc) => {
          const c = tc as { id?: string; function?: { name?: string; arguments?: string } };
          return {
            id: c.id ?? '',
            name: c.function?.name ?? '',
            arguments: extractJson(c.function?.arguments ?? '{}') as Record<string, unknown>,
          };
        });
        return {
          content: msg.content ?? '',
          toolCalls,
          raw: { role: 'assistant', content: msg.content ?? '', tool_calls: msg.tool_calls },
        };
      }, { name: 'llm/chatWithTools', timeoutMs: 45_000, retries: 2, breaker: llmBreaker });
    },
  };
}

export type { SearchFilter };
