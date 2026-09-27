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
    loop: { maxSteps: number; maxPerTool: number };
    memory: { selectSemantic: number; selectEpisodic: number };
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
export const SELECT_SEMANTIC = tuning.auto.memory.selectSemantic;
export const SELECT_EPISODIC = tuning.auto.memory.selectEpisodic;
