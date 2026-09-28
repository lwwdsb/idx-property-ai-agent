/**
 * Tunable algorithm parameters — the SINGLE source for every knob a sweep may change.
 *
 * Deliberately separate from `config.ts`: that one holds environment/deployment values
 * (DB creds, ports, keys) written by a human once; this one holds algorithm parameters
 * that the tuning loop rewrites programmatically. Different writers, different lifecycles
 * — the same reason the user profile is split per writer.
 *
 * Keys are grouped by BLAST RADIUS, so a sweep knows which eval suites to run from the
 * config structure alone rather than from someone remembering:
 *   shared        — both paths (auto reuses the same SkillRegistry, so skill-level knobs
 *                   are shared by construction). Changing these needs BOTH eval suites.
 *   deterministic — only the deterministic router (auto decides tools via function
 *                   calling and never touches intent classification).
 *   auto          — only the ReAct loop. NOT swept automatically: 14-task eval set and
 *                   p99 15.4s means variance swamps the effect size.
 *
 * Override the file for one run (how a sweep tests a candidate without touching the
 * checked-in defaults):  IDX_TUNING=/abs/path/candidate.json npm run ...
 */
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

export interface Tuning {
  version: number;
  shared: {
    retrieval: { rerankCoarse: number; prefetch: number; rerankEnabled: boolean; topK: number };
    search: { maxResults: number; tooMany: number };
    rag: { chunkSize: number; chunkOverlap: number; topK: number };
    facts: { confidenceThreshold: number };
  };
  deterministic: { intent: { embedThreshold: number; embedMargin: number } };
  auto: {
    loop: { maxSteps: number; maxPerTool: number; progressive: boolean };
    memory: { selectSemantic: number; selectEpisodic: number; fallbackSemantic: number; fallbackEpisodic: number };
  };
}

/** Resolved from THIS file's location, not cwd — services start from different dirs. */
export const TUNING_PATH = process.env.IDX_TUNING?.trim()
  || join(dirname(fileURLToPath(import.meta.url)), '..', 'config', 'tuning.json');

function load(): Tuning {
  // No try/catch on purpose: a missing or malformed tuning file must fail loudly at
  // startup. Silently falling back to hardcoded defaults would make a sweep compare
  // two runs that secretly used the same parameters.
  return JSON.parse(readFileSync(TUNING_PATH, 'utf8')) as Tuning;
}

export const tuning: Tuning = load();

// Named exports so call sites read the same as the constants they replaced.
export const RERANK_COARSE = tuning.shared.retrieval.rerankCoarse;
export const PREFETCH = tuning.shared.retrieval.prefetch;
export const SEARCH_TOP_K = tuning.shared.retrieval.topK;
export const MAX_RESULTS = tuning.shared.search.maxResults;
export const TOO_MANY = tuning.shared.search.tooMany;
export const FACT_CONFIDENCE_THRESHOLD = tuning.shared.facts.confidenceThreshold;
export const EMBED_THRESHOLD = tuning.deterministic.intent.embedThreshold;
export const EMBED_MARGIN = tuning.deterministic.intent.embedMargin;
export const MAX_STEPS = tuning.auto.loop.maxSteps;
export const MAX_PER_TOOL = tuning.auto.loop.maxPerTool;
/**
 * Progressive tool loading: expose only `find_tools` up front and let the agent enable what it
 * needs, instead of preloading every tool schema into every turn.
 *
 * It lives HERE, and not as a caller-supplied flag, for two reasons found by measuring:
 *
 * ONE DEFAULT, NOT THREE. `entry.ts` defaulted it to true, `loop.ts` to false, and the agent eval
 * hardcoded false — so the eval had never measured the configuration production actually runs.
 * A knob with a different default at each layer is not a knob, it is three.
 *
 * SWEEPABLE. The objective's A/A guard keys on the sha of this file: two arms that differ only in
 * a flag passed at the call site would hash identically, and a real difference between them would
 * be reported as INVALID. Putting the flag in the config is what makes the arms distinguishable.
 */
export const PROGRESSIVE = tuning.auto.loop.progressive;
/**
 * How many memories may be INJECTED, capped per type. Two caps, not one, because the cap does a
 * completely different job on each of the two selection paths — measured 2026-09-28 on the 13-case
 * memory_select set at (3,1) / (5,3) / (8,5), three runs each:
 *
 *   LLM path       precision 1.000 and over-selections 0 at every cap, so the cap never removes a
 *                  wrong pick — it only ever clips a CORRECT one. Recall rose 0.671 -> 0.697 and
 *                  exact 0.538 -> 0.615 going from (5,3) to (8,5), at a cost of ~14 tokens per
 *                  admitted memory, i.e. about 2 tokens per request. So this cap should be LOOSE:
 *                  the selector's own precision is the real control here.
 *   FALLBACK path  no LLM, ranks by salience x recency x frequency and fills the cap regardless of
 *                  the task, so the cap is the ONLY thing bounding misinjection. Over-selections
 *                  went 26 -> 35 -> 39 across the same three settings, with probe precision 0.000
 *                  throughout (it injects on greetings and out-of-domain requests too). So this cap
 *                  should be TIGHT.
 *
 * One number cannot satisfy both, and it was previously loose where it needed to be tight.
 */
export const SELECT_SEMANTIC = tuning.auto.memory.selectSemantic;
export const SELECT_EPISODIC = tuning.auto.memory.selectEpisodic;
export const FALLBACK_SEMANTIC = tuning.auto.memory.fallbackSemantic;
export const FALLBACK_EPISODIC = tuning.auto.memory.fallbackEpisodic;
