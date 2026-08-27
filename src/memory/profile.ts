/**
 * Long-term user memory (cross-session) — shared by both modes. Classified memory:
 *   - facts:    structured slots (prefs). No name/desc; type is decided by schema, not
 *               content -> 0 LLM. Always full-injected as soft defaults.
 *   - semantic: generalized preferences ("likes bright old homes"). name/desc + metadata.
 *   - episodic: specific events. name/desc + metadata; SELECTIVELY loaded by description.
 *
 * Semantic/episodic entries carry compaction signals: recency (createdAt/lastUsed),
 * frequency (useCount), importance (salience/confidence) + provenance (sourceRuns/mergedFrom).
 * Consolidation/compaction itself is the periodic sub-agent's job; this file stores the
 * data + the cheap immediate level (facts) + load/select helpers.
 *
 * Stored as JSON (authoritative) + a rendered .md (human-visualizable) under data/profiles/.
 * Preferences are SOFT DEFAULTS: only fill fields the user didn't give.
 */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import type { SearchFilter } from '../search/filters.js';
import type { LLMClient, ChatMessage } from '../llm/client.js';

const PREF_KEYS = ['city', 'beds', 'baths', 'maxPrice', 'minPrice', 'propertyType', 'pool', 'minSqft'] as const;
type PrefKey = typeof PREF_KEYS[number];
type PrefValue = string | number | boolean;

export interface PrefEntry { value: PrefValue; confidence: number; seen: number; last: string; }
export type MemoryType = 'semantic' | 'episodic';
export interface MemoryEntry {
  name: string;
  description: string;
  type: MemoryType;
  content: string;
  createdAt: string;
  lastUsed: string;
  useCount: number;
  salience: number;      // importance 0-1
  confidence: number;
  sourceRuns: number[];  // provenance (which runs) — also basis for consolidation
  mergedFrom: string[];  // compaction lineage (names merged/superseded into this)
}
export interface UserProfile {
  userId: string;
  updated: string;
  prefs: Partial<Record<PrefKey, PrefEntry>>;   // facts
  memories: MemoryEntry[];                       // semantic + episodic
}

const DIR = 'data/profiles';
const today = () => new Date().toISOString().slice(0, 10);
const sanitize = (userId: string) => userId.replace(/[^a-zA-Z0-9_-]/g, '_');
const jsonPath = (userId: string) => join(DIR, `${sanitize(userId)}.json`);
const mdPath = (userId: string) => join(DIR, `${sanitize(userId)}.md`);

export function freshProfile(userId: string): UserProfile {
  return { userId, updated: today(), prefs: {}, memories: [] };
}

export function loadProfile(userId: string): UserProfile {
  const p = jsonPath(userId);
  if (!existsSync(p)) return freshProfile(userId);
  try {
    const raw = JSON.parse(readFileSync(p, 'utf8')) as UserProfile;
    raw.memories ??= [];   // tolerate older files
    return raw;
  } catch { return freshProfile(userId); }
}

export function saveProfile(profile: UserProfile): void {
  mkdirSync(DIR, { recursive: true });
  writeFileSync(jsonPath(profile.userId), JSON.stringify(profile, null, 2));
  writeFileSync(mdPath(profile.userId), renderMd(profile));
}

// ── Facts (structured, 0 LLM) ──────────────────────────────────────────────────
/** Immediate level: fold ONE turn's filter into fact slots. Same value reinforces;
 * different value erodes confidence, replacing a stale pref that keeps getting contradicted. */
export function learnFromFilter(profile: UserProfile, filter: SearchFilter): UserProfile {
  const day = today();
  for (const key of PREF_KEYS) {
    const v = filter[key] as PrefValue | undefined;
    if (v == null) continue;
    const e = profile.prefs[key];
    if (!e) profile.prefs[key] = { value: v, confidence: 0.3, seen: 1, last: day };
    else if (e.value === v) { e.seen += 1; e.confidence = Math.min(0.95, e.confidence + 0.15); e.last = day; }
    else { e.confidence -= 0.2; if (e.confidence <= 0.2) profile.prefs[key] = { value: v, confidence: 0.3, seen: 1, last: day }; }
  }
  profile.updated = day;
  return profile;
}

/** High-confidence facts as a partial filter — soft defaults to fill missing fields. */
export function preferredFilter(profile: UserProfile, threshold = 0.5): Partial<SearchFilter> {
  const out: Partial<SearchFilter> = {};
  for (const key of PREF_KEYS) {
    const e = profile.prefs[key];
    if (e && e.confidence >= threshold) (out as Record<string, PrefValue>)[key] = e.value;
  }
  return out;
}

// ── Semantic/episodic entries (name/desc + metadata) ───────────────────────────
export interface NewMemory {
  name: string; description: string; type: MemoryType; content: string;
  salience?: number; confidence?: number; sourceRuns?: number[]; mergedFrom?: string[];
}
/** Add or merge a classified memory (used by the periodic consolidation sub-agent). */
export function addMemory(profile: UserProfile, m: NewMemory): UserProfile {
  const day = today();
  const e = profile.memories.find((x) => x.name === m.name);
  if (e) {
    e.description = m.description || e.description;
    e.content = m.content || e.content;
    e.salience = Math.max(e.salience, m.salience ?? e.salience);
    e.confidence = Math.max(e.confidence, m.confidence ?? e.confidence);
    e.sourceRuns = [...new Set([...e.sourceRuns, ...(m.sourceRuns ?? [])])];
    e.mergedFrom = [...new Set([...e.mergedFrom, ...(m.mergedFrom ?? [])])];
    e.lastUsed = day;
  } else {
    profile.memories.push({
      name: m.name, description: m.description, type: m.type, content: m.content,
      createdAt: day, lastUsed: day, useCount: 0,
      salience: m.salience ?? 0.5, confidence: m.confidence ?? 0.5,
      sourceRuns: m.sourceRuns ?? [], mergedFrom: m.mergedFrom ?? [],
    });
  }
  profile.updated = day;
  return profile;
}

/** Record that a memory was loaded/hit (recency + frequency, for compaction). */
export function touchMemory(profile: UserProfile, name: string): void {
  const e = profile.memories.find((x) => x.name === name);
  if (e) { e.lastUsed = today(); e.useCount += 1; }
}

export const semanticMemories = (p: UserProfile) => p.memories.filter((m) => m.type === 'semantic');
export const episodicMemories = (p: UserProfile) => p.memories.filter((m) => m.type === 'episodic');
/** Lightweight index (name+desc+type) — what a model reads to decide what to load. */
export const memoryIndex = (p: UserProfile) => p.memories.map((m) => ({ name: m.name, description: m.description, type: m.type }));

function daysSince(iso: string): number {
  return Math.max(0, (Date.now() - new Date(iso).getTime()) / 86_400_000);
}
/** salience × recency — the cheap fallback ranking when no LLM. */
function rank(m: MemoryEntry): number {
  return m.salience * (1 / (1 + daysSince(m.lastUsed) / 30));
}

function jsonArray(text: string): number[] {
  const m = text.match(/\[[\s\S]*?\]/);
  if (!m) return [];
  try { const a = JSON.parse(m[0]); return Array.isArray(a) ? a.map(Number).filter(Number.isFinite) : []; }
  catch { return []; }
}

/** Selectively load episodic memories relevant to the current task. LLM reads the
 * name/description index and picks; falls back to salience×recency top-k without an LLM. */
export async function selectMemories(episodic: MemoryEntry[], context: string, llm?: LLMClient, k = 3): Promise<MemoryEntry[]> {
  if (!episodic.length) return [];
  const byRank = () => [...episodic].sort((a, b) => rank(b) - rank(a)).slice(0, k);
  if (!llm?.chatWithTools) return byRank();
  const index = episodic.map((e, i) => `${i}: [${e.name}] ${e.description}`).join('\n');
  const turn = await llm.chatWithTools([
    { role: 'system', content: 'You pick which past episodic memories are RELEVANT to the current task. '
      + 'Given a numbered list ("i: [name] description") and the task, return a JSON array of the relevant indices (e.g. [0,2]); [] if none.' } as ChatMessage,
    { role: 'user', content: `Memories:\n${index}\n\nCurrent task: ${context}` } as ChatMessage,
  ], []);
  const picked = jsonArray(turn.content).filter((i) => episodic[i]).map((i) => episodic[i]!);
  return (picked.length ? picked : byRank()).slice(0, k);
}

/** Delete a memory by name (used for consolidation: superseded/contradicted/redundant). */
export function forgetMemory(profile: UserProfile, name: string): boolean {
  const i = profile.memories.findIndex((m) => m.name === name);
  if (i < 0) return false;
  profile.memories.splice(i, 1);
  profile.updated = today();
  return true;
}

/** Combined compaction score: importance × recency × frequency. */
function compScore(m: MemoryEntry): number {
  const recency = 1 / (1 + daysSince(m.lastUsed) / 30);
  const frequency = 1 + Math.log1p(m.useCount) / 5;
  return m.salience * recency * frequency;
}

export interface CompactOptions { maxSemantic?: number; maxEpisodic?: number; minScore?: number; }
/**
 * Deterministic (0-LLM) compaction: drop very-low-score entries (decay) and, per type,
 * keep only the top-N by score (capacity). The LLM-level promotion/merge/conflict work is
 * done by the consolidation sub-agent; this is the code-level backstop that bounds size.
 */
export function compactMemories(profile: UserProfile, opts: CompactOptions = {}): { removed: string[] } {
  const maxSem = opts.maxSemantic ?? 30;
  const maxEp = opts.maxEpisodic ?? 40;
  const minScore = opts.minScore ?? 0.08;
  const removed: string[] = [];
  for (const [type, cap] of [['semantic', maxSem], ['episodic', maxEp]] as Array<[MemoryType, number]>) {
    const ranked = profile.memories.filter((m) => m.type === type).sort((a, b) => compScore(b) - compScore(a));
    ranked.forEach((m, i) => { if (i >= cap || compScore(m) < minScore) removed.push(m.name); });
  }
  if (removed.length) {
    profile.memories = profile.memories.filter((m) => !removed.includes(m.name));
    profile.updated = today();
  }
  return { removed };
}

// ── Injection helpers ──────────────────────────────────────────────────────────
/** Soft context for the auto agent's system prompt: facts + all semantic + given episodic. */
export function profileHint(profile: UserProfile, episodic: MemoryEntry[] = []): string {
  const parts: string[] = [];
  const pf = preferredFilter(profile);
  if (Object.keys(pf).length) parts.push(`likely preferences ${JSON.stringify(pf)}`);
  const mems = [...semanticMemories(profile), ...episodic];
  if (mems.length) parts.push(`what we know: ${mems.map((m) => m.content).join('; ')}`);
  if (!parts.length) return '';
  return `USER PROFILE (soft context — the current request always overrides): ${parts.join('; ')}.`;
}

/** Human-readable mirror. */
export function renderMd(p: UserProfile): string {
  const prefRows = PREF_KEYS.filter((k) => p.prefs[k]).map((k) => {
    const e = p.prefs[k]!; return `| ${k} | ${e.value} | ${e.confidence.toFixed(2)} | ${e.seen} | ${e.last} |`;
  });
  const memRows = (t: MemoryType) => p.memories.filter((m) => m.type === t)
    .map((m) => `| ${m.name} | ${m.description} | ${m.salience.toFixed(2)} | ${m.useCount} | ${m.lastUsed} |`);
  return [
    `# User profile — ${p.userId}`, '', `_updated ${p.updated}_`, '',
    '## Facts (structured — soft defaults)', '', '| field | value | conf | seen | last |', '|---|---|---|---|---|',
    ...(prefRows.length ? prefRows : ['| _(none)_ | | | | |']), '',
    '## Semantic (generalized preferences)', '', '| name | description | salience | used | last |', '|---|---|---|---|---|',
    ...(memRows('semantic').length ? memRows('semantic') : ['| _(none)_ | | | | |']), '',
    '## Episodic (events — selectively loaded)', '', '| name | description | salience | used | last |', '|---|---|---|---|---|',
    ...(memRows('episodic').length ? memRows('episodic') : ['| _(none)_ | | | | |']), '',
  ].join('\n');
}
