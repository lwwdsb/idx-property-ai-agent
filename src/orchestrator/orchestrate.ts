/**
 * Orchestrator (Week 9): single entry that routes a message to the right skill(s).
 * Deterministic router (not an autonomous loop, per Q3). Compound queries run a
 * fixed recipe (search -> price-validate the top result). Every route decision is
 * logged. Honest naming: router + skills + a verify step — no "agents".
 */
import { classifyIntent, type Intent } from './intent.js';
import { buildRegistry } from './skills.js';
import { pythonBridge, type PythonBridge } from './bridge.js';
import type { SkillRegistry } from './skill.js';
import type { LLMClient } from '../llm/client.js';
import type { ListingRow } from '../search/listingRow.js';
import { logger } from '../logger.js';

export interface OrchestrateOptions {
  registry?: SkillRegistry;
  bridge?: PythonBridge;
  llm?: LLMClient;
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
  const registry = opts.registry ?? buildRegistry(bridge);

  const cls = await classifyIntent(message, {
    llm: opts.llm,
    classify: (m) => bridge.classify(m),
  });
  logger.info('orchestrate route', { userId, intent: cls.intent, confidence: cls.confidence, via: cls.via });

  // low-confidence / unknown -> clarify instead of guessing (Q6)
  if (cls.confidence === 'low' && cls.clarification) {
    return { intent: cls.intent, reply: cls.clarification };
  }

  const ctx = { userId, message, filter: cls.filter };

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

  const skill = registry.get(cls.intent);
  if (!skill) {
    return { intent: cls.intent, reply: cls.clarification ?? "I'm not sure how to help with that yet." };
  }
  const res = await skill.run(ctx);
  return { intent: cls.intent, skill: res.skill, reply: res.reply };
}
