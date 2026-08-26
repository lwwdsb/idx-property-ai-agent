/**
 * Intent + parse eval — PREDICTION stage (TS).
 *
 * Runs the real classifyIntent / parseQuery over the labeled datasets and writes
 * predictions to eval/history/*.preds.jsonl. Metrics are computed by the Python
 * runner (eval/runners/report_intent_parse.py) using the single tested metrics
 * library — so there is exactly ONE metrics implementation (no drift).
 *
 * Intent: full pipeline (real LLM if a key is set + embedding classifier if :8099 is up);
 *   records which subsystems were live.
 * Parse: two deterministic measurements (no real LLM, reproducible):
 *   - escalation decision via a spy LLM (did it decide to call the LLM?)
 *   - regex filter quality (regex/base output) vs gold.
 *
 * Run: npx tsx eval/runners/evalIntentParse.ts
 */
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { classifyIntent } from '../../src/orchestrator/intent.js';
import { parseQuery } from '../../src/search/parseQuery.js';
import { getLLMClient, type LLMClient } from '../../src/llm/client.js';
import { pythonBridge } from '../../src/orchestrator/bridge.js';
import type { SearchFilter } from '../../src/search/filters.js';
import { closePool } from '../../src/db.js';

const HERE = new URL('.', import.meta.url).pathname;
const DATA = `${HERE}../datasets`;
const OUT = `${HERE}../history`;
mkdirSync(OUT, { recursive: true });

function readJsonl(path: string): any[] {
  return readFileSync(path, 'utf8').trim().split('\n').filter(Boolean).map((l) => JSON.parse(l));
}

// Real CA cities that appear in the parse dataset — a self-contained known-city set so
// the parse eval reproduces the city-validation behavior without needing the DB.
const KNOWN = new Set(['irvine', 'san diego', 'los angeles', 'tustin', 'pasadena', 'anaheim',
  'san jose', 'fullerton', 'diamond bar', 'san gabriel', 'beverly hills', 'long beach',
  'arcadia', 'cupertino'].map((s) => s.toLowerCase()));
const isKnownCity = (c: string) => KNOWN.has(c.trim().toLowerCase());

async function runIntent() {
  const cases = readJsonl(`${DATA}/intent.jsonl`);
  const llm = getLLMClient();
  const llmLive = llm.available;
  let classifyLive = false;
  try { await pythonBridge.classify('ping'); classifyLive = true; } catch { /* service down */ }
  const classify = classifyLive ? (m: string) => pythonBridge.classify(m) : undefined;

  const preds = [];
  for (const c of cases) {
    let pred = 'error';
    let via: string | undefined;
    try {
      const r = await classifyIntent(c.input, { llm, classify });
      pred = r.intent;
      via = r.via;
    } catch { /* keep 'error' */ }
    // ALSO record the raw embedding-classifier top skill+score (for threshold sweeping).
    // This is what the fallback layer thresholds on; independent of what classifyIntent
    // finally returned (rules may have decided first).
    let topSkill: string | null = null;
    let topScore: number | null = null;
    let topMargin: number | null = null;
    if (classifyLive) {
      try {
        const g = await pythonBridge.classify(c.input);
        topSkill = g.skill; topScore = g.score; topMargin = g.margin;
      } catch { /* leave null */ }
    }
    preds.push({ id: c.id, input: c.input, gold: c.label.intents, pred, via, topSkill, topScore, topMargin });
  }
  writeFileSync(`${OUT}/intent.preds.jsonl`,
    preds.map((p) => JSON.stringify(p)).join('\n') + '\n');
  console.log(`intent: ${cases.length} cases  [llm=${llmLive ? 'live' : 'off'}, embedClassifier=${classifyLive ? 'live' : 'off'}]`);
  return { llmLive, classifyLive };
}

async function runParse() {
  const cases = readJsonl(`${DATA}/parse.jsonl`);
  const preds = [];
  for (const c of cases) {
    // spy LLM: records whether the pipeline decided to escalate; returns nothing so the
    // resulting filter is the regex/base output (deterministic).
    let called = false;
    const spy: LLMClient = { available: true, async parseFilters() { called = true; return {}; } };
    let filter: SearchFilter = {};
    try {
      const r = await parseQuery(c.input, { llm: spy, isKnownCity });
      filter = r.filter;
    } catch { /* empty */ }
    preds.push({
      id: c.id, input: c.input,
      gold_filter: c.label.filter, gold_escalate: c.label.escalate,
      pred_filter: filter, pred_escalate: called,
    });
  }
  writeFileSync(`${OUT}/parse.preds.jsonl`,
    preds.map((p) => JSON.stringify(p)).join('\n') + '\n');
  console.log(`parse: ${cases.length} cases  [spy-LLM, deterministic]`);
}

(async () => {
  const live = await runIntent();
  await runParse();
  writeFileSync(`${OUT}/intent.meta.json`, JSON.stringify({ ...live, at: new Date().toISOString() }));
  await closePool();
  console.log('predictions written to eval/history/');
})();
