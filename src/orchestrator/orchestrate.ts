/**
 * Orchestrator (Week 9): single entry that routes a message to the right skill(s).
 * Deterministic router (not an autonomous loop, per Q3). Compound queries run a
 * fixed recipe (search -> price-validate the top result). Every route decision is
 * logged. Honest naming: router + skills + a verify step — no "agents".
 */
import { classifyIntent, type Intent } from './intent.js';
import { buildRegistry } from './skills.js';
import { maybePlan, executePlan } from './planner.js';
import { handleDraftCommand } from './draftCommands.js';
import { pythonBridge, type PythonBridge } from './bridge.js';
import type { SkillRegistry } from './skill.js';
import { getLLMClient, type LLMClient } from '../llm/client.js';
import type { ListingRow } from '../search/listingRow.js';
import { MySqlDraftStore, type DraftStore } from '../email/drafts.js';
import type { SendFn } from '../email/email.js';
import { logger } from '../logger.js';

export interface OrchestrateOptions {
  registry?: SkillRegistry;
  bridge?: PythonBridge;
  llm?: LLMClient;
  draftStore?: DraftStore;
  /** Injectable email sender (tests only; default = real SMTP). */
  send?: SendFn;
}

export interface OrchestrateResult {
  intent: Intent;
  skill?: string;
  reply: string;
}

export async function orchestrate(
  userId: string,
  message: string,
  opts: OrchestrateOptions = {},
): Promise<OrchestrateResult> {
  const bridge = opts.bridge ?? pythonBridge;
  const draftStore = opts.draftStore ?? new MySqlDraftStore();
  const registry = opts.registry ?? buildRegistry(bridge, draftStore);

  // Deterministic email approval commands FIRST — never via the LLM (丙).
  const cmd = await handleDraftCommand(userId, message, draftStore, opts.send);
  if (cmd !== null) {
    logger.info('draft command', { userId, message });
    return { intent: 'email', skill: 'email-approve', reply: cmd };
  }

  // default to the configured LLM (DeepSeek) so the parse-extraction fallback is
  // live, not just wired in tests. Empty key => an unavailable client (regex only).
  const llm = opts.llm ?? getLLMClient();
  const cls = await classifyIntent(message, {
    llm,
    classify: (m) => bridge.classify(m),
  });
  logger.info('orchestrate route', { userId, intent: cls.intent, confidence: cls.confidence, via: cls.via });

  // low-confidence / unknown -> clarify instead of guessing (Q6)
  if (cls.confidence === 'low' && cls.clarification) {
    return { intent: cls.intent, reply: cls.clarification };
  }

  const ctx = { userId, message, filter: cls.filter, llm };

  // compound recipe: search, then validate the top result's price (fixed chain)
  if (cls.intent === 'compound') {
    const search = await registry.get('search')!.run(ctx);
    const rows = (search.data as { rows?: ListingRow[] } | undefined)?.rows;
    if (!rows?.length) {
      return { intent: 'compound', skill: 'search', reply: search.reply };
    }
    const top = rows[0]!;
    let verdict = 'price check unavailable';
    try {
      verdict = (JSON.parse(await bridge.validate(top)) as { verdict?: string }).verdict ?? verdict;
    } catch { /* 乙: validation failure must not break the reply */ }
    return {
      intent: 'compound',
      skill: 'search+validate',
      reply: `${search.reply}\n\n💰 Top result price check: ${verdict}`,
    };
  }

  // multi-skill planner (gated) — when the query wants several registry skills at once.
  // A constrained plan-then-execute, NOT an autonomous loop; skills keep their own locks.
  const plan = await maybePlan(message, cls.filter, registry, llm);
  if (plan) {
    logger.info('multi-skill plan', { userId, plan });
    const { reply, skills } = await executePlan(plan, ctx, registry);
    return { intent: 'compound', skill: skills.join('+'), reply };
  }

  const skill = registry.get(cls.intent);
  if (!skill) {
    return { intent: cls.intent, reply: cls.clarification ?? "I'm not sure how to help with that yet." };
  }
  const res = await skill.run(ctx);
  return { intent: cls.intent, skill: res.skill, reply: res.reply };
}
