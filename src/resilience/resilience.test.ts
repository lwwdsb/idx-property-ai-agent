/**
 * Resilience wrapper tests (offline): retry(transient only) + timeout + fallback +
 * circuit breaker (open/half-open) + transient classification.
 * Run: npx tsx src/resilience/resilience.test.ts
 */
import assert from 'node:assert/strict';
import { withResilience, CircuitBreaker, isTransient } from './resilience.js';

let pass = 0, fail = 0;
async function check(name: string, fn: () => Promise<void> | void) {
  try { await fn(); pass++; console.log('✓', name); }
  catch (e) { fail++; console.log('✗', name, (e as Error).message); }
}

await check('retries transient errors then succeeds', async () => {
  let n = 0;
  const r = await withResilience(async () => { if (++n < 3) throw new Error('ETIMEDOUT'); return 'ok'; },
    { name: 't', retries: 3, baseDelayMs: 1 });
  assert.equal(r, 'ok'); assert.equal(n, 3);
});
await check('does NOT retry non-transient (4xx/logic)', async () => {
  let n = 0;
  await assert.rejects(() => withResilience(async () => { n++; throw new Error('HTTP 400 bad request'); },
    { name: 't', retries: 3, baseDelayMs: 1 }));
  assert.equal(n, 1);   // no retry
});
await check('timeout aborts a hanging call', async () => {
  await assert.rejects(() => withResilience(() => new Promise<never>(() => {}),
    { name: 't', timeoutMs: 20, retries: 0 }), /timeout/);
});
await check('fallback returns degraded value on failure', async () => {
  const r = await withResilience<string>(async () => { throw new Error('ETIMEDOUT'); },
    { name: 't', retries: 1, baseDelayMs: 1, fallback: () => 'degraded' });
  assert.equal(r, 'degraded');
});
await check('breaker opens after threshold, fast-fails without calling fn', async () => {
  const b = new CircuitBreaker(2, 10_000);
  const failing = () => withResilience<string>(async () => { throw new Error('ETIMEDOUT'); },
    { name: 't', retries: 0, breaker: b, fallback: () => 'f' });
  await failing(); await failing();               // 2 failures -> open
  assert.equal(b.state, 'open');
  let executed = false;
  const r = await withResilience<string>(async () => { executed = true; return 'x'; },
    { name: 't', breaker: b, fallback: () => 'fast-fail' });
  assert.equal(r, 'fast-fail'); assert.equal(executed, false);   // fn NOT called while open
});
await check('half-open after cooldown lets a probe through, success closes it', async () => {
  const b = new CircuitBreaker(1, 20);
  await withResilience<string>(async () => { throw new Error('ETIMEDOUT'); },
    { name: 't', retries: 0, breaker: b, fallback: () => 'f' });
  assert.equal(b.state, 'open');
  await new Promise((r) => setTimeout(r, 30));
  assert.equal(b.state, 'half-open');
  const r = await withResilience(async () => 'recovered', { name: 't', breaker: b });
  assert.equal(r, 'recovered'); assert.equal(b.state, 'closed');
});
await check('isTransient: retry timeouts/429/5xx, not 4xx/auth', () => {
  assert.ok(isTransient(new Error('ETIMEDOUT')));
  assert.ok(isTransient(new Error('LLM HTTP 503')));
  assert.ok(isTransient(new Error('LLM HTTP 429: rate limit')));
  assert.ok(!isTransient(new Error('LLM HTTP 400 bad request')));
  assert.ok(!isTransient(new Error('unauthorized')));
});

console.log(`\n${pass}/${pass + fail} passed`);
if (fail) process.exit(1);
