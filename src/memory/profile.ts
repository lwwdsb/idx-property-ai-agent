/**
 * Long-term user memory (cross-session) — shared by both modes. Classified memory:
 *   - facts:    structured slots (prefs). No name/desc; type is decided by schema, not
 *               content -> 0 LLM. Always full-injected as soft defaults.
 *   - semantic: generalized preferences ("likes bright old homes"). name/desc + metadata.
 *   - episodic: specific events. name/desc + metadata; SELECTIVELY loaded by description.
 *
 * Semantic/episodic entries carry three compaction signals: recency (lastUsed / updatedAt),
 * frequency (useCount), importance (salience) + provenance (sourceRuns/mergedFrom).
 * Consolidation/compaction itself is the periodic sub-agent's job; this file stores the
 * data + the cheap immediate level (facts) + load/select helpers.
 *
 * STORAGE — split by WRITER, so the chat agent and the consolidation sub-agent never write the
 * same file and cannot clobber each other (no locking needed; the conflict is gone by design):
 *   <uid>.facts.json     prefs                       <- written by the chat path only
 *   <uid>.memories.json  memories + watermark        <- written by the consolidation agent only
 *   <uid>.usage.json     per-memory useCount/lastUsed <- written by the chat path only
 *   <uid>.md             rendered mirror, derived, best-effort (never read back)
 *   <uid>.json           LEGACY single file — still read (and migrated) if present
 * Reading the other writer's file is fine: memories are soft context, a slightly stale read
 * changes nothing. Every write is atomic (tmp file + rename), because a torn JSON file would
 * hit loadProfile's catch and silently reset the whole profile.
 *
 * Preferences are SOFT DEFAULTS: only fill fields the user didn't give.
 */
import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from 'node:fs';
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
  /** Content last rewritten by the consolidation agent (its file). Distinct from lastUsed,
   *  which is a read-side counter owned by the chat path — different writers, different files. */
  updatedAt: string;
  lastUsed: string;
  useCount: number;
  salience: number;      // importance 0-1
  sourceRuns: number[];  // provenance (which runs) — also basis for consolidation
  mergedFrom: string[];  // compaction lineage (names merged/superseded into this)
}
export interface UserProfile {
  userId: string;
  updated: string;
  prefs: Partial<Record<PrefKey, PrefEntry>>;   // facts
  memories: MemoryEntry[];                       // semantic + episodic
  /**
   * Consumer cursor into this user's agent_runs: the highest run id already digested by
   * the consolidation sub-agent. Reading "the last N runs" is a sliding window — it re-reads
   * what it just read, and silently drops runs that scroll past the limit between passes.
   * A watermark turns that into "everything after the last pass": no re-reads, no gaps.
   * Advanced only on a clean finish, so a crashed pass re-reads rather than skips
   * (at-least-once; addMemory merges by name, so a repeat is harmless).
   */
  lastConsolidatedRunId?: number;
}

const DIR = 'data/profiles';
const today = () => new Date().toISOString().slice(0, 10);
const sanitize = (userId: string) => userId.replace(/[^a-zA-Z0-9_-]/g, '_');
const legacyPath = (userId: string) => join(DIR, `${sanitize(userId)}.json`);
const factsPath = (userId: string) => join(DIR, `${sanitize(userId)}.facts.json`);
const memoriesPath = (userId: string) => join(DIR, `${sanitize(userId)}.memories.json`);
const usagePath = (userId: string) => join(DIR, `${sanitize(userId)}.usage.json`);
const mdPath = (userId: string) => join(DIR, `${sanitize(userId)}.md`);

interface UsageEntry { useCount: number; lastUsed: string; }

/** Write via tmp+rename: readers see either the whole old file or the whole new one. */
function writeAtomic(path: string, text: string): void {
  mkdirSync(DIR, { recursive: true });
  const tmp = `${path}.tmp`;
  writeFileSync(tmp, text);
  renameSync(tmp, path);
}
function readJson<T>(path: string): T | null {
  if (!existsSync(path)) return null;
  try { return JSON.parse(readFileSync(path, 'utf8')) as T; } catch { return null; }
}

export function freshProfile(userId: string): UserProfile {
  return { userId, updated: today(), prefs: {}, memories: [], lastConsolidatedRunId: 0 };
}

/**
 * Merge the per-writer files back into one view. Usage counters live in their own file, so a
 * memory's useCount/lastUsed are filled in here rather than stored alongside its content.
 */
export function loadProfile(userId: string): UserProfile {
  const legacy = readJson<UserProfile>(legacyPath(userId));   // pre-split file, if any
  const facts = readJson<{ prefs?: UserProfile['prefs']; updated?: string }>(factsPath(userId));
  const mem = readJson<{ memories?: MemoryEntry[]; lastConsolidatedRunId?: number; updated?: string }>(memoriesPath(userId));
  const usage = readJson<Record<string, UsageEntry>>(usagePath(userId)) ?? {};

  const prefs = facts?.prefs ?? legacy?.prefs ?? {};
  const memories = (mem?.memories ?? legacy?.memories ?? []).map((raw) => {
    // `confidence` was defined but never consumed by compScore, and the add_memory tool never
    // even exposed it — dropped. Strip it off existing files instead of carrying it forever.
    const { confidence: _dead, ...m } = raw as MemoryEntry & { confidence?: number };
    const u = usage[m.name];
    return {
      ...m,
      updatedAt: m.updatedAt ?? m.createdAt,
      useCount: u?.useCount ?? m.useCount ?? 0,
      lastUsed: u?.lastUsed ?? m.lastUsed ?? m.createdAt,
    };
  });
  const updated = [facts?.updated, mem?.updated, legacy?.updated].filter(Boolean).sort().pop() ?? today();
  return {
    userId, updated, prefs, memories,
    lastConsolidatedRunId: mem?.lastConsolidatedRunId ?? legacy?.lastConsolidatedRunId ?? 0,
  };
}

/**
 * The three writes below are deliberately SEPARATE. Each is owned by exactly one writer, so
 * "load -> modify -> save" from the chat path and from the consolidation agent touch different
 * files and cannot overwrite each other. Call the one matching what you changed:
 *   learnFromFilter        -> saveFacts
 *   touchMemory            -> saveUsage
 *   addMemory/forgetMemory/compactMemories/watermark -> saveMemories
 */
export function saveFacts(profile: UserProfile): void {
  writeAtomic(factsPath(profile.userId), JSON.stringify({ prefs: profile.prefs, updated: profile.updated }, null, 2));
  refreshMd(profile);
}
export function saveMemories(profile: UserProfile): void {
  // Usage counters are the other writer's file — strip them so we never write them back here.
  const memories = profile.memories.map(({ useCount: _u, lastUsed: _l, ...rest }) => rest);
  writeAtomic(memoriesPath(profile.userId), JSON.stringify({
    memories, lastConsolidatedRunId: profile.lastConsolidatedRunId ?? 0, updated: profile.updated,
  }, null, 2));
  refreshMd(profile);
}
export function saveUsage(profile: UserProfile): void {
  const usage: Record<string, UsageEntry> = {};
  for (const m of profile.memories) usage[m.name] = { useCount: m.useCount, lastUsed: m.lastUsed };
  writeAtomic(usagePath(profile.userId), JSON.stringify(usage, null, 2));
  refreshMd(profile);
}
/** Derived, human-only mirror. Never read back, so a failure here must not break a save. */
function refreshMd(profile: UserProfile): void {
  try { writeAtomic(mdPath(profile.userId), renderMd(profile)); } catch { /* mirror only */ }
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
  salience?: number; sourceRuns?: number[]; mergedFrom?: string[];
}
/** Add or merge a classified memory (used by the periodic consolidation sub-agent). */
export function addMemory(profile: UserProfile, m: NewMemory): UserProfile {
  const day = today();
  const e = profile.memories.find((x) => x.name === m.name);
  if (e) {
    // same name = UPDATE of the same memory. If the content changed (e.g. an opposite
    // preference), the new value REPLACES the old (recency wins) — salience follows the new
    // content, NOT max (max would leave a stale high score on new content).
    // Reinforcement of "same content over time" comes from useCount/recency in compScore.
    const contentChanged = !!m.content && m.content !== e.content;
    e.description = m.description || e.description;
    e.content = m.content || e.content;
    if (m.salience !== undefined) e.salience = contentChanged ? m.salience : Math.max(e.salience, m.salience);
    e.sourceRuns = [...new Set([...e.sourceRuns, ...(m.sourceRuns ?? [])])];
    e.mergedFrom = [...new Set([...e.mergedFrom, ...(m.mergedFrom ?? [])])];
    e.updatedAt = day;   // NOT lastUsed: that one belongs to the chat path's usage file
  } else {
    profile.memories.push({
      name: m.name, description: m.description, type: m.type, content: m.content,
      createdAt: day, updatedAt: day, lastUsed: day, useCount: 0,
      salience: m.salience ?? 0.5,
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
/** Freshest of the two writers' timestamps: last read (usage file) vs last rewrite (memories file). */
const freshness = (m: MemoryEntry) => Math.min(daysSince(m.lastUsed), daysSince(m.updatedAt ?? m.createdAt));
/** salience × recency — the cheap fallback ranking when no LLM. */
function rank(m: MemoryEntry): number {
  return m.salience * (1 / (1 + freshness(m) / 30));
}

function jsonArray(text: string): number[] {
  const m = text.match(/\[[\s\S]*?\]/);
  if (!m) return [];
  try { const a = JSON.parse(m[0]); return Array.isArray(a) ? a.map(Number).filter(Number.isFinite) : []; }
  catch { return []; }
}

export interface SelectLimits { semantic?: number; episodic?: number; }
/**
 * Selectively load the memories relevant to the current task — BOTH types.
 *
 * Semantic used to be injected wholesale. That quietly broke its own scoring: every entry got
 * "used" on every turn, so useCount carried no signal (all of them rise together, relative
 * order never moves) and eviction fell back to salience + age alone. Selecting both types
 * gives semantic a real usage signal, and bounds the prompt as the set grows.
 *
 * The LLM reads one name/description index for both types and returns the relevant indices;
 * caps are applied PER TYPE afterwards, so a pile of episodics can't crowd out preferences.
 * Without an LLM it degrades to salience×recency, capped the same way.
 */
export async function selectMemories(
  memories: MemoryEntry[],
  context: string,
  llm?: LLMClient,
  limits: SelectLimits = {},
): Promise<MemoryEntry[]> {
  const maxSem = limits.semantic ?? 5;   // generalized prefs: few, usually relevant
  const maxEp = limits.episodic ?? 3;    // events: many, mostly irrelevant
  if (!memories.length) return [];
  const capPerType = (list: MemoryEntry[]) => {
    const out: MemoryEntry[] = [];
    let sem = 0, ep = 0;
    for (const m of list) {
      if (m.type === 'semantic') { if (sem < maxSem) { out.push(m); sem += 1; } }
      else if (ep < maxEp) { out.push(m); ep += 1; }
    }
    return out;
  };
  const byRank = () => capPerType([...memories].sort((a, b) => rank(b) - rank(a)));
  if (!llm?.chatWithTools) return byRank();
  const index = memories.map((m, i) => `${i}: [${m.type}] [${m.name}] ${m.description}`).join('\n');
  const turn = await llm.chatWithTools([
    { role: 'system', content: 'You pick which stored memories about this user are RELEVANT to the current task. '
      + 'Each line is "i: [semantic|episodic] [name] description" — semantic = a generalized preference, '
      + 'episodic = a past event. Return a JSON array of the relevant indices (e.g. [0,2]); [] if none.' } as ChatMessage,
    { role: 'user', content: `Memories:\n${index}\n\nCurrent task: ${context}` } as ChatMessage,
  ], []);
  const picked = jsonArray(turn.content).filter((i) => memories[i]).map((i) => memories[i]!);
  return picked.length ? capPerType(picked) : byRank();
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
  const recency = 1 / (1 + freshness(m) / 30);
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
/**
 * Soft context for the auto agent's system prompt: facts + the memories selectMemories picked.
 * Nothing is pulled in wholesale here — whatever is injected must have been SELECTED, so it is
 * also something we can honestly mark as used (see selectMemories).
 */
export function profileHint(profile: UserProfile, selected: MemoryEntry[] = []): string {
  const parts: string[] = [];
  const pf = preferredFilter(profile);
  if (Object.keys(pf).length) parts.push(`likely preferences ${JSON.stringify(pf)}`);
  if (selected.length) parts.push(`what we know: ${selected.map((m) => m.content).join('; ')}`);
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
