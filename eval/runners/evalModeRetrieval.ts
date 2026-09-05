/**
 * Mode-retrieval predictor (TS side): for each query, extract the search FILTER two ways —
 *   - deterministic: parseQuery (regex, the 0-LLM primary path)
 *   - auto:          the real agent path — give the LLM the full tool set, capture the `search`
 *                    tool call's arguments, sanitizeFilter them (+ which tools it chose)
 * Writes predictions; the Python scorer (eval_mode_retrieval.py) runs hybrid_search with each
 * mode's filter and computes known-item recall + param-extraction F1 against the human gold.
 *
 * TS does the REAL extraction (no reimplementation → no drift); Python does retrieval + metrics.
 * Needs LLM_API_KEY (auto extraction). Run:  npx tsx eval/runners/evalModeRetrieval.ts
 */
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { parseQuery } from '../../src/search/parseQuery.js';
import { isKnownCity } from '../../src/search/cityDictionary.js';
import { getLLMClient, sanitizeFilter } from '../../src/llm/client.js';
import { buildRegistry } from '../../src/orchestrator/skills.js';
import { pythonBridge } from '../../src/orchestrator/bridge.js';
import { toolSpecs } from '../../src/agent/auto/tools.js';
import { InMemoryDraftStore } from '../../src/email/drafts.js';
import { closePool } from '../../src/db.js';

const HERE = new URL('.', import.meta.url).pathname;
const DATA = `${HERE}../datasets/mode_retrieval.jsonl`;
const OUT = `${HERE}../history/mode_retrieval.preds.jsonl`;

interface Case { id: string; input: string; style: string; lang: string; gold: unknown; }

const SYSTEM = 'You are a real-estate assistant. If the user is searching for homes, call the '
  + 'search tool. Fill ONLY the structured fields the user actually stated; NEVER invent a bound '
  + 'they did not say (e.g. for "around/under X" set maxPrice only, NOT minPrice). Put style/'
  + 'feature words in the free-text `semantic` field. If the request is NOT a property task '
  + '(chit-chat, off-topic), do NOT call any tool. You may call multiple tools for multi-part requests.';

async function main() {
  const llm = getLLMClient();
  if (!llm.chatWithTools) { console.error('LLM chatWithTools unavailable (need LLM_API_KEY).'); process.exit(1); }
  const registry = buildRegistry(pythonBridge, new InMemoryDraftStore());
  const tools = toolSpecs(registry);

  const cases = readFileSync(DATA, 'utf8').split('\n').filter(Boolean).map((l) => JSON.parse(l) as Case);
  const preds: unknown[] = [];

  for (const c of cases) {
    // deterministic: regex extraction (0-LLM primary)
    const regex = (await parseQuery(c.input, { isKnownCity })).filter;

    // auto: real agent extraction — LLM sees the full tool set, we capture its choices
    let autoFilter: Record<string, unknown> = {};
    let autoSemantic = '';
    let toolsCalled: string[] = [];
    try {
      const turn = await llm.chatWithTools([{ role: 'system', content: SYSTEM }, { role: 'user', content: c.input }], tools);
      toolsCalled = [...new Set(turn.toolCalls.map((t) => t.name))];
      const searchCall = turn.toolCalls.find((t) => t.name === 'search');
      if (searchCall) {
        autoFilter = sanitizeFilter(searchCall.arguments) as Record<string, unknown>;
        autoSemantic = typeof searchCall.arguments.semantic === 'string' ? searchCall.arguments.semantic : '';
      }
    } catch (e) {
      console.error(`  ! auto extraction failed for ${c.id}: ${String(e)}`);
    }

    preds.push({ id: c.id, regex_filter: regex, auto_filter: autoFilter, auto_semantic: autoSemantic, auto_tools: toolsCalled });
    console.log(`  ${c.id} [${c.style}/${c.lang}] regex=${JSON.stringify(regex)} auto=${JSON.stringify(autoFilter)} tools=${toolsCalled.join(',')||'-'}`);
  }

  mkdirSync(`${HERE}../history`, { recursive: true });
  writeFileSync(OUT, preds.map((p) => JSON.stringify(p)).join('\n') + '\n');
  console.log(`\n${preds.length} predictions -> ${OUT}`);
  await closePool();
}

main().catch((e) => { console.error(e); process.exit(1); });
