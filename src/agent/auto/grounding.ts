/**
 * Grounding: does the agent's reply only cite facts that came from tool results?
 *
 * ONE implementation, shared by the runtime gate (loop.ts) and the offline eval
 * (evalAgent.ts) — same reason the eval metrics are single-sourced: no drift.
 *
 * Scope (v1): hard facts = 6+ digit numbers (MLS# / listing ids) — the thing an agent
 * would invent to fake a listing, and the one it won't rewrite. Prices/percentages are
 * out of scope for now (commas + "$1.22M" shorthand make matching noisy).
 */
import type { ChatMessage, LLMClient } from '../../llm/client.js';
import { logger } from '../../logger.js';

const MAX_GROUND_RETRY = 2;   // LLM-rewrite attempts before deterministic strip

/** Concatenated text of every tool observation in a transcript. */
export function observationBlob(messages: ChatMessage[]): string {
  return messages.filter((m) => m.role === 'tool').map((m) => String(m.content)).join(' ');
}

/** Hard-fact ids in the reply and which of them are NOT present in the observations. */
export function checkGrounding(reply: string, obsBlob: string): { idCount: number; ungrounded: string[] } {
  const ids = [...new Set(reply.match(/\b\d{6,}\b/g) ?? [])];
  const blob = obsBlob.replace(/[\s,]/g, '');
  return { idCount: ids.length, ungrounded: ids.filter((id) => !blob.includes(id)) };
}

/** Deterministic fallback: mark unverifiable ids so nothing invented reaches the user. */
export function stripUngrounded(reply: string, ungrounded: string[]): string {
  let out = reply;
  for (const id of ungrounded) out = out.split(id).join('[unverified]');
  return out;
}

/**
 * Runtime grounding gate (option C). Before a final answer reaches the user: verify every
 * hard fact (MLS#/id) traces to a tool observation. If not — first ask the LLM to rewrite
 * using only tool facts (retry ≤ MAX_GROUND_RETRY); if it still can't, deterministically
 * mark the unverifiable ids [unverified] + disclaim. Zero cost on the common path.
 */
export interface GroundResult { reply: string; rewrites: number; stripped: number; calls: number; }
export async function groundFinal(reply: string, messages: ChatMessage[], llm: LLMClient): Promise<GroundResult> {
  if (!llm.chatWithTools) return { reply, rewrites: 0, stripped: 0, calls: 0 };
  const blob = observationBlob(messages);
  let out = reply;
  let { ungrounded } = checkGrounding(out, blob);
  let tries = 0, calls = 0;
  while (ungrounded.length && tries < MAX_GROUND_RETRY) {
    tries++; calls++;
    const fix = await llm.chatWithTools([
      ...messages,
      { role: 'assistant', content: out },
      { role: 'user', content: `Your answer cites listing ids not found in any tool result: ${ungrounded.join(', ')}. `
        + 'Rewrite it using ONLY facts from the tool results above; remove any listing or number you cannot cite. '
        + 'Return just the corrected answer.' },
    ], []);
    out = fix.content || out;
    ungrounded = checkGrounding(out, blob).ungrounded;
  }
  let stripped = 0;
  if (ungrounded.length) {
    logger.warn('grounding gate: stripping unverifiable ids', { ungrounded });
    stripped = ungrounded.length;
    out = stripUngrounded(out, ungrounded)
      + "\n\n_(Some details couldn't be verified against the search results and were marked [unverified].)_";
  }
  return { reply: out, rewrites: tries, stripped, calls };
}
