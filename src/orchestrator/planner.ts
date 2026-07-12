/**
 * Constrained multi-skill planner (an escalation, NOT an autonomous loop).
 *
 * Gate: only engages when a cheap deterministic detector sees >=2 distinct intents
 * (so single-intent queries stay on the free rule path). Plan: the LLM picks an
 * ordered subset of REGISTERED skills (validated against the registry, capped at 3);
 * if the LLM is unavailable/invalid it falls back to the deterministically detected
 * set (乙). Execute: plan-then-execute ONCE (no re-plan loop). Skills keep their own
 * guardrails (email still draft-only, verifier still deterministic) — the planner
 * only decides WHICH skills run, never bypasses their locks.
 */
import type { SearchFilter } from '../search/filters.js';
import type { SkillContext, SkillRegistry } from './skill.js';
import type { LLMClient, PlanStep } from '../llm/client.js';
import { MARKET_RE, RECOMMEND_RE, KNOWLEDGE_RE, EMAIL_RE } from './intent.js';

// A real "search" needs a constraint beyond city (else "Irvine 行情" would look like
// search+market). city alone is shared by market/recommend and isn't a search signal.
const SEARCH_CONSTRAINTS: Array<keyof SearchFilter> = ['beds', 'baths', 'maxPrice', 'minPrice', 'propertyType', 'pool', 'minSqft'];
const ORDER = ['search', 'market', 'recommend', 'knowledge', 'email'];
const MAX_PLAN = 3;

/** Distinct intent types present in the message (canonical order). */
export function detectMultiIntent(message: string, filter: SearchFilter): string[] {
  const set = new Set<string>();
  if (SEARCH_CONSTRAINTS.some((k) => filter[k] != null)) set.add('search');
  // strip the "days on market" phrase so its "market" substring isn't a false signal
  const marketText = message.replace(/\bdays on market\b|在市天数|市场天数/gi, ' ');
  if (MARKET_RE.test(marketText)) set.add('market');
  if (RECOMMEND_RE.test(message)) set.add('recommend');
  if (KNOWLEDGE_RE.test(message)) set.add('knowledge');
  if (EMAIL_RE.test(message)) set.add('email');
  return ORDER.filter((s) => set.has(s));
}

/** Returns an ordered plan of >=2 steps, or null to fall back to single-skill routing.
 * The LLM decomposes the message into per-skill sub-queries; the deterministic fallback
 * hands each skill the full message (乙). */
export async function maybePlan(
  message: string,
  filter: SearchFilter,
  registry: SkillRegistry,
  llm?: LLMClient,
): Promise<PlanStep[] | null> {
  const detected = detectMultiIntent(message, filter);
  if (detected.length < 2) return null;                 // gate: single intent -> normal routing

  let plan: PlanStep[] | null = null;
  if (llm?.available && llm.planSkills) {
    try {
      const p = await llm.planSkills(message, registry.list().map((s) => ({ name: s.name, description: s.description })));
      if (p.length) plan = p;                            // client validated (known names, cap 3)
    } catch { /* 乙: planner failure -> deterministic fallback */ }
  }
  // fallback: each detected skill gets the full message (no LLM to split it)
  if (!plan) plan = detected.map((s) => ({ skill: s, query: message }));
  plan = plan.filter((st) => registry.has(st.skill)).slice(0, MAX_PLAN);
  return plan.length >= 2 ? plan : null;                 // LLM may decide it's really single -> null
}

/** Run the planned steps in order, each skill on ITS OWN sub-query, compose replies
 * (plan-then-execute, once). The LLM keeps each sub-query self-contained (its own
 * constraints); the semantic-search path also reuses ctx.filter. A lossy sub-query at
 * worst broadens that skill's results (still valid + user-visible), never fabricates. */
export async function executePlan(
  plan: PlanStep[],
  ctx: SkillContext,
  registry: SkillRegistry,
): Promise<{ reply: string; skills: string[] }> {
  const parts: string[] = [];
  const skills: string[] = [];
  for (const step of plan) {
    const skill = registry.get(step.skill);
    if (!skill) continue;
    const res = await skill.run({ ...ctx, message: step.query });
    parts.push(res.reply);
    skills.push(res.skill);
  }
  return { reply: parts.join('\n\n────────\n\n'), skills };
}
