/**
 * Unified resilience wrapper for every external call (LLM / retrieval / DB / Maps / SMTP).
 *
 * One place implements: TIMEOUT (don't hang) + RETRY (expo backoff + jitter, transient
 * errors only) + CIRCUIT BREAKER (stop hammering a down service) + FALLBACK (degrade, don't
 * crash) + METRICS. Wrap each dependency once instead of scattering try/catch everywhere.
 *
 * Degradation follows 丙>甲>乙: side-effecting calls (email) must pass an idempotency key so a
 * retry can't double-send; read-only calls (LLM/retrieval) retry freely; fallbacks return a
 * safe degraded value (regex parse, cached, "temporarily unavailable") rather than throwing up.
 */
import { logger } from '../logger.js';

export interface ResilienceMetric {
  name: string; ok: boolean; attempt: number; ms: number; error?: string; breaker?: string;
}

export interface ResilienceOptions<T> {
  name: string;
  timeoutMs?: number;                     // default 30s
  retries?: number;                       // default 2 (so up to 3 attempts)
  baseDelayMs?: number;                   // default 200ms
  retryOn?: (e: unknown) => boolean;      // default isTransient
  breaker?: CircuitBreaker;               // shared per-dependency
  fallback?: (e: unknown) => T | Promise<T>;
  onMetric?: (m: ResilienceMetric) => void;
}

/** Per-dependency circuit breaker: closed -> (failures≥threshold) open -> (after cooldown) half-open. */
export class CircuitBreaker {
  private failures = 0;
  private openedAt = 0;
  constructor(private readonly threshold = 5, private readonly cooldownMs = 15_000) {}
  get state(): 'closed' | 'open' | 'half-open' {
    if (this.openedAt === 0) return 'closed';
    return Date.now() - this.openedAt >= this.cooldownMs ? 'half-open' : 'open';
  }
  canPass(): boolean { return this.state !== 'open'; }   // half-open lets ONE probe through
  onSuccess(): void { this.failures = 0; this.openedAt = 0; }
  onFailure(): void { if (++this.failures >= this.threshold) this.openedAt = Date.now(); }
}

/** Transient = worth retrying: timeouts, network resets, 429/5xx. Not 4xx/logic/data errors. */
export function isTransient(e: unknown): boolean {
  const msg = String((e as Error)?.message ?? e).toLowerCase();
  if (/\b(4[013-9]\d|4[2-9]\d)\b/.test(msg) && !/\b429\b/.test(msg)) return false;   // most 4xx = permanent
  return /timeout|timed out|abort|econnrefused|econnreset|etimedout|enotfound|network|fetch failed/.test(msg)
    || /\b(429|500|502|503|504)\b/.test(msg);
}

function withTimeout<T>(p: Promise<T>, ms: number, name: string): Promise<T> {
  let timer: ReturnType<typeof setTimeout>;
  const timeout = new Promise<never>((_, rej) => {
    timer = setTimeout(() => rej(new Error(`${name}: timeout after ${ms}ms`)), ms);
  });
  return Promise.race([p, timeout]).finally(() => clearTimeout(timer));
}

const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

export async function withResilience<T>(fn: () => Promise<T>, opts: ResilienceOptions<T>): Promise<T> {
  const { name, timeoutMs = 30_000, retries = 2, baseDelayMs = 200,
    retryOn = isTransient, breaker, fallback, onMetric } = opts;

  // circuit OPEN -> fast-fail to fallback, don't hammer a service that's down
  if (breaker && !breaker.canPass()) {
    onMetric?.({ name, ok: false, attempt: -1, ms: 0, error: 'circuit_open', breaker: 'open' });
    logger.warn('resilience: circuit open, fast-fail', { name });
    if (fallback) return fallback(new Error(`${name}: circuit open`));
    throw new Error(`${name}: circuit open`);
  }

  let lastErr: unknown;
  for (let attempt = 0; attempt <= retries; attempt++) {
    const t0 = Date.now();
    try {
      const out = await withTimeout(fn(), timeoutMs, name);
      breaker?.onSuccess();
      onMetric?.({ name, ok: true, attempt, ms: Date.now() - t0, breaker: breaker?.state });
      return out;
    } catch (e) {
      lastErr = e;
      breaker?.onFailure();
      onMetric?.({ name, ok: false, attempt, ms: Date.now() - t0, error: String((e as Error)?.message ?? e), breaker: breaker?.state });
      if (attempt < retries && retryOn(e)) {
        await sleep(baseDelayMs * 2 ** attempt + Math.random() * baseDelayMs);   // expo backoff + jitter
        continue;
      }
      break;
    }
  }
  logger.warn('resilience: call failed after retries', { name, error: String((lastErr as Error)?.message ?? lastErr) });
  if (fallback) return fallback(lastErr);
  throw lastErr;
}
