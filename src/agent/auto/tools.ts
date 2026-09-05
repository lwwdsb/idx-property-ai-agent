/**
 * Auto/agent mode — tool layer.
 *
 * Exposes the existing SkillRegistry to the ReAct loop as OpenAI function tools.
 * Deliberately thin: every skill shares ONE parameter schema (a natural-language
 * `query` + optional structured filter slots). `executeTool` reuses `sanitizeFilter`
 * to pull the slots and hands `query` to the existing `skill.run` — we do NOT rewrite
 * skills. A hallucinated tool name or a thrown error becomes an observation string
 * (fed back so the model can recover), never a crash.
 */
import { sanitizeFilter, type ToolSpec, type LLMClient, type SearchFilter } from '../../llm/client.js';
import { mergeFilter, type FilterPatch } from '../../search/filters.js';
import type { SkillRegistry } from '../../orchestrator/skill.js';

/** Shared JSON Schema. `query` is the self-contained sub-task; slots sharpen precision. */
export const SHARED_TOOL_SCHEMA: Record<string, unknown> = {
  type: 'object',
  properties: {
    query: {
      type: 'string',
      description: 'The self-contained sub-task in natural language for THIS tool only, '
        + 'e.g. "3-bed homes in Irvine under 2M", or "email the Irvine report to a@b.com".',
    },
    city: { type: 'string' },
    beds: { type: 'number', description: 'minimum bedrooms' },
    baths: { type: 'number', description: 'minimum bathrooms' },
    maxPrice: { type: 'number', description: 'USD upper budget. Use for "under/below/around/about/up to/no more than X".' },
    minPrice: { type: 'number', description: 'USD lower bound. ONLY when the user explicitly wants AT LEAST / more than / starting from X. Do NOT set it for "around/about X" — that is maxPrice only.' },
    propertyType: { type: 'string', enum: ['condo', 'townhouse', 'single-family'] },
    pool: { type: 'boolean' },
    minSqft: { type: 'number' },
    semantic: { type: 'string', description: 'free-text style/features NOT covered by the structured fields, e.g. "ocean view craftsman", "bright open floor plan". ALWAYS write this in ENGLISH (the listing corpus is English), even if the user wrote in another language — translate the style/feature terms.' },
  },
  required: ['query'],
};

/** The catalog: one function tool per registered skill. A skill may declare its own
 * `paramSchema` (e.g. email's subject/body); otherwise it gets the shared slot schema. */
export function toolSpecs(registry: SkillRegistry): ToolSpec[] {
  return registry.list().map((s) => ({
    name: s.name,
    description: s.description,
    parameters: s.paramSchema ?? SHARED_TOOL_SCHEMA,
  }));
}

// ── Progressive tool loading (deferred-tool style) ─────────────────────────────
/** The meta-tool: in progressive mode it's the ONLY tool exposed up front. The agent
 * calls it to discover + ENABLE the real tools it needs, so the prompt isn't preloaded
 * with every schema (same idea as this environment's ToolSearch over deferred tools). */
export const FIND_TOOLS_SPEC: ToolSpec = {
  name: 'find_tools',
  description: 'Discover available tools by keyword and ENABLE them for you to call. Returns '
    + 'matching tool names + descriptions. Call this first whenever you need a capability you '
    + 'have not enabled yet (e.g. "market stats", "send email", "similar listings").',
  parameters: {
    type: 'object',
    properties: { query: { type: 'string', description: 'keywords for the capability you need' } },
    required: ['query'],
  },
};

/** Keyword-match registry skills; returns matching ToolSpecs (or all, so the agent is never stranded). */
export function findTools(registry: SkillRegistry, query: string): ToolSpec[] {
  const terms = query.toLowerCase().split(/\s+/).filter(Boolean);
  const specs = toolSpecs(registry);
  const scored = specs.map((s) => {
    const hay = `${s.name} ${s.description}`.toLowerCase();
    return { s, score: terms.filter((t) => hay.includes(t)).length };
  });
  const hits = scored.filter((x) => x.score > 0).sort((a, b) => b.score - a.score).map((x) => x.s);
  return hits.length ? hits : specs;
}

export interface ToolRunCtx { userId: string; llm?: LLMClient; memConstraints?: SearchFilter; }
/** ALWAYS resolves; `draftId` is set when an email tool produced a pending draft
 * (the loop's HITL interrupt point). Errors come back as an observation, never a throw. */
export interface ToolResult { observation: string; draftId?: number; }

export async function executeTool(
  registry: SkillRegistry,
  name: string,
  args: Record<string, unknown>,
  ctx: ToolRunCtx,
): Promise<ToolResult> {
  const skill = registry.get(name);
  if (!skill) {
    const known = registry.list().map((s) => s.name).join(', ');
    return { observation: `error: unknown tool "${name}". Available tools: ${known}.` };
  }
  const query = typeof args.query === 'string' ? args.query : '';
  try {
    // filter = LLM args merged OVER accumulated memory constraints (args take priority);
    // sanitizeFilter only sets valid values (never null), so the casts are safe.
    const argFilter = sanitizeFilter(args) as FilterPatch;
    const filter = ctx.memConstraints ? mergeFilter(ctx.memConstraints, argFilter) : (argFilter as SearchFilter);
    const r = await skill.run({ userId: ctx.userId, message: query, filter, llm: ctx.llm, args });
    // email returns the created draft as `data`; surface its id so the loop can suspend for HITL.
    const d = r.data as { id?: unknown } | undefined;
    const draftId = name === 'email' && d && typeof d.id === 'number' ? d.id : undefined;
    return { observation: r.reply, draftId };
  } catch (e) {
    return { observation: `error running ${name}: ${String(e)}` };
  }
}
