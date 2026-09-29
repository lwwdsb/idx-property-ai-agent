/**
 * Jev (TypeSafe AI "System One") — typed decisions instead of generated text.
 *
 * WHAT IT IS AND IS NOT. Jev answers enumerated questions: a Choice returns one option out of a
 * declared set plus a full probability distribution, a Noul returns a single 0-1 belief. It
 * DELIBERATELY GIVES UP STRING GENERATION, which fixes what it can and cannot replace here:
 *
 *   it CAN decide  "which tool does this request need" — that is an enumeration
 *   it CANNOT make "query: 3-bed homes in Irvine under 2M" — that is free text
 *
 * So in auto mode it is a candidate for the DECISION half of function calling, never the whole
 * thing: the arguments still have to come from somewhere (the LLM, or the deterministic parser).
 * This module deliberately stops at the decision, because that is the part that can be measured
 * against gold labels the project already has.
 *
 * "CANNOT HALLUCINATE" IS ABOUT TYPES, NOT ABOUT BEING RIGHT. An answer is always one of the
 * declared options — a class of failure the current code has to guard (an LLM naming a tool that
 * does not exist) simply cannot occur. It can still choose the WRONG option, which is exactly what
 * the eval is for.
 *
 * CONFIDENCE IS NOT CALIBRATED. The vendor documents `confidence` as a shape statistic over the
 * probability distribution, with no calibration guarantee and explicit advice to test on your own
 * data. The full distribution is therefore kept and returned, so the project's existing
 * threshold-and-margin rule (top1 - top2) can be applied to it rather than trusting one number.
 */
import { withResilience, CircuitBreaker } from '../resilience/resilience.js';
import { logger } from '../logger.js';

const BASE = (process.env.JEV_BASE_URL || 'https://api.typesafe.ai/v1').replace(/\/$/, '');
const MODEL = process.env.JEV_MODEL || 'jev-latest';
const KEY = (process.env.TYPESAFE_API_KEY || '').trim();

/** Same breaker discipline as the LLM path: a dead dependency must fail fast, not queue up. */
const jevBreaker = new CircuitBreaker(5, 15_000);

export const jevAvailable = (): boolean => KEY.length > 0;

export interface ChoiceAnswer {
  choice: string;
  /** full distribution over the declared options, sums to ~1 */
  probabilities: Record<string, number>;
  /** vendor's shape statistic — NOT a calibrated probability */
  confidence: number;
  /** top1 - top2, computed here because the project's router already thinks in margins */
  margin: number;
}

export interface JevUsage { inputTokens: number; outputTokens: number; }

export interface SystemOneResult {
  choices: Record<string, ChoiceAnswer>;
  nouls: Record<string, number>;
  usage: JevUsage;
  model: string;
}

export interface ChoiceSpec { instructions: string; criteria: Record<string, string | null>; }
export interface NoulSpec { instructions: string; }

/**
 * One request, many questions. The API answers every question in a single round trip, so asking
 * "which tool" and "is this compound" together costs one network hop rather than two — which
 * matters because the round trip, not the tokens, is the expensive part of this call.
 */
export async function systemOne(
  state: string | Record<string, unknown>,
  questions: { choices?: Record<string, ChoiceSpec>; nouls?: Record<string, NoulSpec> },
): Promise<SystemOneResult> {
  if (!jevAvailable()) throw new Error('Jev not configured (set TYPESAFE_API_KEY)');
  const body: Record<string, unknown> = { state, model: MODEL, questions: {} };
  const q = body.questions as Record<string, unknown>;
  for (const [id, c] of Object.entries(questions.choices ?? {})) {
    q[id] = { type: 'choice', instructions: c.instructions, criteria: c.criteria };
  }
  for (const [id, n] of Object.entries(questions.nouls ?? {})) {
    q[id] = { type: 'noul', instructions: n.instructions };
  }

  return withResilience(async () => {
    const res = await fetch(`${BASE}/systemone`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${KEY}` },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      // 429/529 are the documented retry cases; withResilience's backoff handles them, and a
      // 422 (bad question shape) must surface loudly rather than be retried into a rate limit.
      throw new Error(`Jev HTTP ${res.status}: ${(await res.text()).slice(0, 200)}`);
    }
    const data = (await res.json()) as {
      model?: string;
      answers?: Record<string, { type?: string; choice?: string; probabilities?: Record<string, number>;
        confidence?: number; noul?: number }>;
      usage?: { input_tokens?: number; output_tokens?: number };
    };
    const choices: Record<string, ChoiceAnswer> = {};
    const nouls: Record<string, number> = {};
    for (const [id, a] of Object.entries(data.answers ?? {})) {
      if (a.type === 'noul' || typeof a.noul === 'number') {
        nouls[id] = typeof a.noul === 'number' ? a.noul : 0;
        continue;
      }
      const probs = a.probabilities ?? {};
      const sorted = Object.values(probs).sort((x, y) => y - x);
      choices[id] = {
        choice: a.choice ?? '',
        probabilities: probs,
        confidence: typeof a.confidence === 'number' ? a.confidence : 0,
        margin: sorted.length > 1 ? Number((sorted[0]! - sorted[1]!).toFixed(6)) : (sorted[0] ?? 0),
      };
    }
    const usage = { inputTokens: data.usage?.input_tokens ?? 0, outputTokens: data.usage?.output_tokens ?? 0 };
    logger.debug('jev systemOne', { model: data.model, usage, nq: Object.keys(q).length });
    return { choices, nouls, usage, model: data.model ?? MODEL };
  }, { name: 'jev/systemOne', timeoutMs: 15_000, retries: 2, breaker: jevBreaker });
}

/** Marker for "none of the declared tools apply" — auto mode expresses OOD by calling no tool,
 *  and an enumerated selector needs an explicit option to say the same thing. */
export const JEV_NONE = 'none';

/** Ask which registered tool a task needs. Descriptions come from the SAME skill descriptions the
 *  LLM sees as function-tool descriptions, so the two arms are told the same thing about each
 *  tool — otherwise the comparison would be measuring prompt wording, not the model. */
export function toolChoiceSpec(tools: Array<{ name: string; description: string }>): ChoiceSpec {
  const criteria: Record<string, string> = {};
  for (const t of tools) criteria[t.name] = t.description;
  criteria[JEV_NONE] = 'None of the above — the request is outside what these tools can do '
    + '(chit-chat, another domain, or an action this system does not support).';
  return {
    instructions: 'Which ONE tool should handle this user request? Pick "none" if no tool applies.',
    criteria,
  };
}
