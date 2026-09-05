/**
 * End-to-end eval — PREDICTION stage.
 *
 * Runs the REAL orchestrate() over the e2e dataset and checks the user-level outcome:
 * intent routing, skill composition, reply content, and safety behavior (email drafted
 * not sent). This is the "internal end-to-end" (from the orchestrate entry to the reply);
 * it does not include the OpenClaw front-door hop (external agent, not automatable).
 *
 * Real by design: real pythonBridge (needs :8099 for classify/recommend/rag), real DB
 * (search/market), real LLM if a key is set (parse/planner). SAFE by design: a FAKE
 * sender (never delivers email) + an in-memory draft store (never touches the real DB).
 *
 * Assertions per case (all optional): intent (exact), skill_has (substring), reply_has,
 * reply_not (substring must be ABSENT — key for the 丙 "not Sent" check). Metrics are
 * computed by report_e2e.py.
 *
 * Run: npx tsx eval/runners/evalE2E.ts
 */
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { orchestrate } from '../../src/orchestrator/orchestrate.js';
import { buildRegistry } from '../../src/orchestrator/skills.js';
import { pythonBridge } from '../../src/orchestrator/bridge.js';
import { getLLMClient } from '../../src/llm/client.js';
import { InMemoryDraftStore } from '../../src/email/drafts.js';
import { config } from '../../src/config.js';
import { closePool } from '../../src/db.js';

const HERE = new URL('.', import.meta.url).pathname;
const OUT = `${HERE}../history`;
mkdirSync(OUT, { recursive: true });

function readJsonl(path: string): any[] {
  return readFileSync(path, 'utf8').trim().split('\n').filter(Boolean).map((l) => JSON.parse(l));
}

/** Run one case, return prediction + whether each assertion passed. */
function evalCase(expect: any, res: { intent: string; skill?: string; reply: string }) {
  const checks: Record<string, boolean> = {};
  if (expect.intent !== undefined) checks.intent = res.intent === expect.intent;
  if (expect.skill_has !== undefined) checks.skill_has = (res.skill ?? '').includes(expect.skill_has);
  if (expect.reply_has !== undefined) checks.reply_has = res.reply.toLowerCase().includes(String(expect.reply_has).toLowerCase());
  if (expect.reply_not !== undefined) checks.reply_not = !res.reply.toLowerCase().includes(String(expect.reply_not).toLowerCase());
  const pass = Object.values(checks).every(Boolean);
  return { checks, pass };
}

(async () => {
  const cases = readJsonl(`${HERE}../datasets/e2e.jsonl`);
  const llm = getLLMClient();
  let classifyLive = false;
  try { await pythonBridge.classify('ping'); classifyLive = true; } catch { /* :8099 down */ }

  // SAFE injection: fake sender (never delivers), in-memory drafts (never touches real DB).
  const sentBox: string[] = [];
  const send = async (m: { to: string }) => { sentBox.push(m.to); };
  const draftStore = new InMemoryDraftStore();
  const registry = buildRegistry(pythonBridge, draftStore);
  const opts = { registry, bridge: pythonBridge, llm, draftStore, send };

  // an authorized operator (email allowlist); email cases must run as this user, else the
  // auth gate refuses to draft and we'd only ever test the refusal, not the draft->approval flow.
  const operator = config.email.allowlist[0] ?? 'e2e-operator';

  const preds = [];
  for (const c of cases) {
    // fresh userId per case so multi-turn session state doesn't leak between cases;
    // meta.user==="operator" -> run as the authorized sender so drafting is exercised.
    const userId = c.meta?.user === 'operator' ? operator : `e2e-${c.id}`;
    let res = { intent: 'error', skill: undefined as string | undefined, reply: '' };
    try {
      const r = await orchestrate(userId, c.input, opts);
      res = { intent: r.intent, skill: r.skill, reply: r.reply };
    } catch (e) {
      res.reply = `ERROR: ${String(e)}`;
    }
    const { checks, pass } = evalCase(c.expect, res);
    preds.push({ id: c.id, input: c.input, note: c.note, expect: c.expect,
      got: { intent: res.intent, skill: res.skill, reply: res.reply.slice(0, 200) },
      checks, pass });
  }

  // safety invariant: the fake sender must have received NOTHING (no case should send)
  const anySent = sentBox.length;

  writeFileSync(`${OUT}/e2e.preds.jsonl`, preds.map((p) => JSON.stringify(p)).join('\n') + '\n');
  writeFileSync(`${OUT}/e2e.meta.json`, JSON.stringify({
    llmLive: llm.available, classifyLive, anySent, at: new Date().toISOString(),
  }));
  await closePool();
  const passed = preds.filter((p) => p.pass).length;
  console.log(`e2e: ${passed}/${preds.length} passed  [llm=${llm.available ? 'live' : 'off'}, classify=${classifyLive ? 'live' : 'off'}, emailsSent=${anySent}]`);
})();
