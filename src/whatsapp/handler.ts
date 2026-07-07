/**
 * WhatsApp communication layer (Week 10): turn an inbound message into a reply.
 *
 * Guardrails around the orchestrator:
 *  - Idempotency: the same message id is processed once (WhatsApp/webhooks can
 *    redeliver) — prevents a duplicate message triggering a duplicate query.
 *  - Rate limit: per-user cap so one user can't hammer the DB/LLM.
 *  - Typing indicator: sent before the (possibly slow) orchestrate call to mask latency.
 *  - Failure degrades to a friendly message, never a crash (乙).
 *
 * The Channel + orchestrate are injected, so this is fully testable offline.
 */
import { logger } from '../logger.js';

export interface InboundMessage {
  from: string;   // sender id (E.164)
  id: string;     // provider message id (for idempotency)
  text: string;
}

export interface Channel {
  sendTyping(to: string): Promise<void>;
  sendText(to: string, text: string): Promise<void>;
}

export type OrchestrateFn = (userId: string, text: string) => Promise<string>;

/** Idempotency: remember recently-seen message ids (with TTL). */
export class SeenStore {
  private readonly seen = new Map<string, number>();
  constructor(private readonly ttlMs = 10 * 60_000) {}
  check(id: string): boolean {
    const now = Date.now();
    for (const [k, t] of this.seen) if (now - t > this.ttlMs) this.seen.delete(k);
    if (this.seen.has(id)) return true;
    this.seen.set(id, now);
    return false;
  }
}

/** Fixed-window per-user rate limiter. */
export class RateLimiter {
  private readonly hits = new Map<string, number[]>();
  constructor(private readonly max = 10, private readonly windowMs = 60_000) {}
  allow(userId: string): boolean {
    const now = Date.now();
    const arr = (this.hits.get(userId) ?? []).filter((t) => now - t < this.windowMs);
    if (arr.length >= this.max) { this.hits.set(userId, arr); return false; }
    arr.push(now); this.hits.set(userId, arr);
    return true;
  }
}

export interface HandlerDeps {
  channel: Channel;
  orchestrate: OrchestrateFn;
  seen?: SeenStore;
  limiter?: RateLimiter;
}

export type HandleStatus = 'ok' | 'duplicate' | 'rate_limited' | 'error';

export async function handleInbound(
  msg: InboundMessage,
  deps: HandlerDeps,
): Promise<{ status: HandleStatus; reply?: string }> {
  const seen = deps.seen ?? defaultSeen;
  const limiter = deps.limiter ?? defaultLimiter;

  if (seen.check(msg.id)) {
    logger.info('inbound duplicate ignored', { id: msg.id });
    return { status: 'duplicate' };
  }
  if (!limiter.allow(msg.from)) {
    await deps.channel.sendText(msg.from, "You're sending messages very fast — give me a moment 🙏");
    return { status: 'rate_limited' };
  }

  try {
    await deps.channel.sendTyping(msg.from);
    const reply = await deps.orchestrate(msg.from, msg.text);
    await deps.channel.sendText(msg.from, reply);
    return { status: 'ok', reply };
  } catch (err) {
    logger.error('inbound handling failed', { id: msg.id, error: String(err) });
    await deps.channel.sendText(msg.from, "Sorry — I hit a problem handling that. Please try again.");
    return { status: 'error' };
  }
}

const defaultSeen = new SeenStore();
const defaultLimiter = new RateLimiter();
