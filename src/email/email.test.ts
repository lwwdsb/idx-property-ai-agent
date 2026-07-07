/**
 * Week 11 email red-line tests (offline, in-memory store, dry-run send).
 * The non-negotiables: no unapproved send, gated authorization, batch cap,
 * idempotent send. Run: npm run test:email
 */
import assert from 'node:assert/strict';
import { config } from '../config.js';
import { InMemoryDraftStore } from './drafts.js';
import { draftEmail, approveAndSend, cancelDraft, MAX_RECIPIENTS } from './email.js';

const OP = config.email.allowlist[0] ?? 'op';          // an authorized operator
const OUTSIDER = '+10000000001';                        // not on the allowlist
const okReq = (over = {}) => ({ createdBy: OP, recipients: ['client@example.com'], subject: 'Hi', body: 'Body', ...over });

let pass = 0, fail = 0;
async function check(name: string, fn: () => Promise<void>) {
  try { await fn(); pass++; console.log('✓', name); }
  catch (e) { fail++; console.error('✗', name, '\n   ', (e as Error).message.split('\n')[0]); }
}

await check('draftEmail creates a PENDING draft — never sent', async () => {
  const store = new InMemoryDraftStore();
  const r = await draftEmail(okReq(), store);
  assert.equal(r.ok, true);
  assert.equal(r.draft!.status, 'pending_approval');   // 丙: drafting never sends
});

await check('only approveAndSend can send; it marks sent (dry-run)', async () => {
  const store = new InMemoryDraftStore();
  const { draft } = await draftEmail(okReq(), store);
  const r = await approveAndSend(draft!.id, OP, store);
  assert.ok(r.status === 'sent' || r.status === 'sent_dryrun');
  assert.equal((await store.get(draft!.id))!.status, 'sent');
});

await check('idempotent: approving twice does not re-send', async () => {
  const store = new InMemoryDraftStore();
  const { draft } = await draftEmail(okReq(), store);
  await approveAndSend(draft!.id, OP, store);
  const second = await approveAndSend(draft!.id, OP, store);
  assert.equal(second.status, 'already_sent');
});

await check('non-allowlisted user cannot draft', async () => {
  if (config.email.allowlist.length === 0) return; // allowlist disabled in this env
  const store = new InMemoryDraftStore();
  const r = await draftEmail(okReq({ createdBy: OUTSIDER }), store);
  assert.equal(r.ok, false);
  assert.match(r.error!, /authoriz/i);
});

await check('non-allowlisted approver cannot send', async () => {
  if (config.email.allowlist.length === 0) return;
  const store = new InMemoryDraftStore();
  const { draft } = await draftEmail(okReq(), store);
  const r = await approveAndSend(draft!.id, OUTSIDER, store);
  assert.equal(r.status, 'unauthorized');
  assert.equal((await store.get(draft!.id))!.status, 'pending_approval'); // not sent
});

await check('batch cap enforced (no unbounded blast)', async () => {
  const store = new InMemoryDraftStore();
  const many = Array.from({ length: MAX_RECIPIENTS + 1 }, (_, i) => `c${i}@example.com`);
  const r = await draftEmail(okReq({ recipients: many }), store);
  assert.equal(r.ok, false);
  assert.match(r.error!, /too many/i);
});

await check('invalid recipient / empty fields rejected', async () => {
  const store = new InMemoryDraftStore();
  assert.equal((await draftEmail(okReq({ recipients: ['not-an-email'] }), store)).ok, false);
  assert.equal((await draftEmail(okReq({ recipients: [] }), store)).ok, false);
  assert.equal((await draftEmail(okReq({ subject: '' }), store)).ok, false);
});

await check('cancelled draft cannot be sent', async () => {
  const store = new InMemoryDraftStore();
  const { draft } = await draftEmail(okReq(), store);
  assert.equal(await cancelDraft(draft!.id, store), true);
  const r = await approveAndSend(draft!.id, OP, store);
  assert.equal(r.status, 'not_pending');
});

console.log(`\n${pass}/${pass + fail} passed`);
if (fail) process.exitCode = 1;
