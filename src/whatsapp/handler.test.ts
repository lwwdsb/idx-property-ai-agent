/**
 * Week 10 WhatsApp handler guardrail tests (offline: fake channel + orchestrate).
 * Run: npm run test:wa
 */
import assert from 'node:assert/strict';
import { handleInbound, SeenStore, RateLimiter, type Channel } from './handler.js';

type Case = { name: string; fn: () => Promise<void> };
const cases: Case[] = [];
const t = (name: string, fn: Case['fn']) => cases.push({ name, fn });

function fakeChannel() {
  const log: string[] = [];
  const channel: Channel = {
    async sendTyping(to) { log.push(`typing:${to}`); },
    async sendText(to, text) { log.push(`text:${to}:${text.slice(0, 30)}`); },
  };
  return { channel, log };
}

t('normal message: typing then reply, status ok', async () => {
  const { channel, log } = fakeChannel();
  let called = '';
  const r = await handleInbound({ from: '+1', id: 'm1', text: 'hi' },
    { channel, orchestrate: async (u, t) => { called = `${u}:${t}`; return 'ANSWER'; },
      seen: new SeenStore(), limiter: new RateLimiter() });
  assert.equal(r.status, 'ok');
  assert.equal(called, '+1:hi');
  assert.ok(log[0]!.startsWith('typing:'), 'typing sent first');
  assert.ok(log.some((l) => l.includes('ANSWER')), 'reply sent');
});

t('duplicate message id is ignored (idempotency)', async () => {
  const { channel, log } = fakeChannel();
  const seen = new SeenStore();
  const deps = { channel, orchestrate: async () => 'A', seen, limiter: new RateLimiter() };
  await handleInbound({ from: '+1', id: 'dup', text: 'x' }, deps);
  log.length = 0;
  const r = await handleInbound({ from: '+1', id: 'dup', text: 'x' }, deps);
  assert.equal(r.status, 'duplicate');
  assert.equal(log.length, 0, 'nothing sent on duplicate');
});

t('rate limit blocks after max, orchestrate not called', async () => {
  const { channel } = fakeChannel();
  const limiter = new RateLimiter(2, 60_000);
  let calls = 0;
  const deps = { channel, orchestrate: async () => { calls++; return 'A'; }, seen: new SeenStore(), limiter };
  await handleInbound({ from: '+u', id: 'a', text: '1' }, deps);
  await handleInbound({ from: '+u', id: 'b', text: '2' }, deps);
  const r = await handleInbound({ from: '+u', id: 'c', text: '3' }, deps);
  assert.equal(r.status, 'rate_limited');
  assert.equal(calls, 2, 'orchestrate ran only within the limit');
});

t('orchestrate failure -> friendly error, no crash (乙)', async () => {
  const { channel, log } = fakeChannel();
  const r = await handleInbound({ from: '+1', id: 'e1', text: 'boom' },
    { channel, orchestrate: async () => { throw new Error('db down'); },
      seen: new SeenStore(), limiter: new RateLimiter() });
  assert.equal(r.status, 'error');
  assert.ok(log.some((l) => /Sorry/i.test(l)), 'friendly error sent');
});

(async () => {
  let pass = 0, fail = 0;
  for (const c of cases) {
    try { await c.fn(); pass++; console.log('✓', c.name); }
    catch (e) { fail++; console.error('✗', c.name, '\n   ', (e as Error).message.split('\n')[0]); }
  }
  console.log(`\n${pass}/${pass + fail} passed`);
  if (fail) process.exitCode = 1;
})();
