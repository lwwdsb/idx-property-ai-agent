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
import type { SkillContext, SkillRegistry, SkillResult } from './skill.js';
import type { LLMClient, PlanStep } from '../llm/client.js';
import { MARKET_RE, RECOMMEND_RE, KNOWLEDGE_RE, EMAIL_RE, VALUE_RE } from './intent.js';
import { logger } from '../logger.js';
import { ARITY, ARITY_THRESHOLD } from '../tuning.js';
import { systemOne, jevAvailable } from '../llm/jev.js';

// A real "search" needs a constraint beyond city (else "Irvine 行情" would look like
// search+market). city alone is shared by market/recommend and isn't a search signal.
const SEARCH_CONSTRAINTS: Array<keyof SearchFilter> = ['beds', 'baths', 'maxPrice', 'minPrice', 'propertyType', 'pool', 'minSqft'];
// validate 紧跟在 search 之后:它要看前一步搜出来的东西
const ORDER = ['search', 'validate', 'market', 'recommend', 'knowledge', 'email'];
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
  if (VALUE_RE.test(message)) set.add('validate');   // 兜底集合也要含估价,否则计划拆不出它
  return ORDER.filter((s) => set.has(s));
}

/**
 * 这条请求是【一件事】还是【好几件事】。
 *
 * 这是一个【路由决策】而不是一个分类标签:好几件 -> 升级 LLM 一次调用拆子 query 并抽参;
 * 一件 -> 走单工具流程。所以它的质量直接决定"会不会用户问了三件事只答了一件"。
 *
 * 两类错误代价不对称,这决定了整个设计的偏向:
 *   漏判(多判成单) 用户问了两件事只答一件,而且【无声】—— 不可恢复
 *   误判(单判成多) 多花一次 LLM 调用,而 planSkills 若认定其实是单意图会返回 null 回落 —— 自愈
 * 所以偏向召回,阈值取低不取高。
 *
 * Jev 不可用时回落正则计数 —— 那是今天的行为,已知 recall 0.545,不理想但不会更糟。
 */
async function isMultiIntent(message: string, regexDetected: string[], llm?: LLMClient,
                             known?: boolean): Promise<boolean> {
  const byRegex = regexDetected.length >= 2;
  // 上游那一次请求已经顺带问过了 -> 直接用,别再发一次。每条消息两个往返是纯浪费。
  if (known !== undefined) return known;
  if (ARITY !== 'jev' || !jevAvailable()) return byRegex;
  try {
    const r = await systemOne(message, {
      nouls: { multi: { instructions: 'Does this request ask the assistant to do more than ONE '
        + 'distinct thing (e.g. find listings AND report market stats), as opposed to one request '
        + 'with several constraints?' } },
    });
    const v = r.nouls.multi ?? 0;
    logger.debug('arity via jev', { message: message.slice(0, 60), noul: v, byRegex });
    return v >= ARITY_THRESHOLD;
  } catch {
    return byRegex;                                     // 乙: 判元数失败 -> 今天的行为
  }
}

/** Returns an ordered plan of >=2 steps, or null to fall back to single-skill routing.
 * The LLM decomposes the message into per-skill sub-queries; the deterministic fallback
 * hands each skill the full message (乙). */
export async function maybePlan(
  message: string,
  filter: SearchFilter,
  registry: SkillRegistry,
  llm?: LLMClient,
  /** 上游 classifyIntent 那次 Jev 请求顺带答出的元数;给了就不再发第二次请求。 */
  knownMultiIntent?: boolean,
): Promise<PlanStep[] | null> {
  // 正则仍然跑,但职责变了:它不再是【闸门】,只提供"哪几个技能"这个兜底集合(当 LLM 拆不出计划时
  // 每个技能拿到整句)。闸门交给 isMultiIntent —— 因为正则数关键词在两个方向上都不可靠。
  const detected = detectMultiIntent(message, filter);
  if (!(await isMultiIntent(message, detected, llm, knownMultiIntent))) return null;   // 一件事 -> 单工具流程
  // 闸门说是多意图但正则一个技能都没数出来:没有兜底集合可用,只能靠 LLM 的计划。
  if (!detected.length && !(llm?.available && llm.planSkills)) return null;

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

/** Execute a plan (plan-then-execute, once), composing replies in plan order.
 *
 * OPT-IN parallel scheduling (safe by default): only skills explicitly marked
 * `parallelSafe` run together in a PARALLEL batch (Promise.allSettled — one failing
 * doesn't drop the others, 乙). Everything else runs SEQUENTIALLY afterwards — so an
 * unmarked skill (unsure, or dependent on others' output via ctx.priorResults) is never
 * wrongly parallelized. Each skill runs on ITS OWN sub-query; the LLM keeps it
 * self-contained. Replies are composed in the original plan order. */
export async function executePlan(
  plan: PlanStep[],
  ctx: SkillContext,
  registry: SkillRegistry,
): Promise<{ reply: string; skills: string[] }> {
  const steps = plan.map((s) => ({ step: s, skill: registry.get(s.skill) })).filter((x) => x.skill);
  const parallel = steps.filter((x) => x.skill!.parallelSafe);
  const serial = steps.filter((x) => !x.skill!.parallelSafe);   // default: serial (safe)
  const done = new Map<string, SkillResult>();

  // verified-independent skills -> parallel batch
  const settled = await Promise.allSettled(parallel.map((x) => x.skill!.run({ ...ctx, message: x.step.query })));
  parallel.forEach((x, i) => {
    const r = settled[i]!;
    if (r.status === 'fulfilled') done.set(x.step.skill, r.value);
    else logger.warn('plan skill failed', { skill: x.step.skill, error: String(r.reason) });
  });

  // everything else -> sequential, able to read prior outputs
  for (const x of serial) {
    try {
      const r = await x.skill!.run({ ...ctx, message: x.step.query, priorResults: [...done.values()] });
      done.set(x.step.skill, r);
    } catch (e) {
      logger.warn('plan skill failed', { skill: x.step.skill, error: String(e) });
    }
  }

  // compose in the original plan order
  const parts: string[] = [];
  const skills: string[] = [];
  for (const s of plan) {
    const r = done.get(s.skill);
    if (r) { parts.push(r.reply); skills.push(r.skill); }
  }
  return { reply: parts.join('\n\n────────\n\n'), skills };
}
