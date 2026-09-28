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
import { FACT_CONFIDENCE_THRESHOLD, SELECT_SEMANTIC, SELECT_EPISODIC } from '../tuning.js';

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
  /** The structured constraint this memory DECLARES, written by whoever created it.
   *  Declared rather than parsed — see memoryDerivedFilter for why prose cannot be trusted. */
  slots?: Partial<SearchFilter>;
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
export function preferredFilter(profile: UserProfile, threshold = FACT_CONFIDENCE_THRESHOLD): Partial<SearchFilter> {
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
  slots?: Partial<SearchFilter>;
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
    if (m.slots !== undefined) e.slots = m.slots;   // a reversed preference restates its slots
    e.updatedAt = day;   // NOT lastUsed: that one belongs to the chat path's usage file
  } else {
    profile.memories.push({
      name: m.name, description: m.description, type: m.type, content: m.content,
      createdAt: day, updatedAt: day, lastUsed: day, useCount: 0,
      salience: m.salience ?? 0.5,
      sourceRuns: m.sourceRuns ?? [], mergedFrom: m.mergedFrom ?? [],
      slots: m.slots,
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

/** What `list_memories` hands the consolidation agent: the index PLUS any detected contradiction.
 * The deterministic layer can only refuse to use a conflicted field; deciding which claim is
 * right needs the sessions, so the conflict is put in front of the one agent that reads them. */
export const memoryIndexWithConflicts = (p: UserProfile) => ({
  memories: memoryIndex(p),
  conflicts: slotConflicts(p.memories),
});

function daysSince(iso: string): number {
  return Math.max(0, (Date.now() - new Date(iso).getTime()) / 86_400_000);
}
/** Freshest of the two writers' timestamps: last read (usage file) vs last rewrite (memories file). */
const freshness = (m: MemoryEntry) => Math.min(daysSince(m.lastUsed), daysSince(m.updatedAt ?? m.createdAt));
/** salience × recency — the cheap fallback ranking when no LLM. */
function rank(m: MemoryEntry): number {
  return m.salience * (1 / (1 + freshness(m) / 30));
}

/**
 * Parse the selector's reply. Returns null when nothing could be parsed, and `[]` when the model
 * validly answered "none" — the two must NOT collapse.
 *
 * They used to: this returned `[]` for a missing array, a throwing JSON.parse AND a genuine `[]`,
 * and the caller then treated every empty result as a failure and fell back to ranking EVERY
 * memory. So on a greeting or an out-of-domain request — where the correct answer is to inject
 * nothing — the whole profile was injected instead. The prompt asks for "[] if none" and the code
 * discarded that answer. Measured on the two empty-gold probes: precision 0.00 before, 1.00 after.
 */
function jsonArray(text: string): number[] | null {
  const m = text.match(/\[[\s\S]*?\]/);
  if (!m) return null;
  try {
    const a = JSON.parse(m[0]);
    return Array.isArray(a) ? a.map(Number).filter(Number.isFinite) : null;
  } catch { return null; }
}

/**
 * The selector's instructions.
 *
 * The first version said only "pick the RELEVANT ones", and a conservative reading of "relevant"
 * cost recall: measured 0.729, missing 5 of 18 gold items across 10 cases — including the episodic
 * memory needed to resolve "这套", without which the reference cannot be resolved at all.
 *
 * The rules below are CATEGORIES, deliberately not hints about particular cases: what makes a
 * memory bear on a task, not which memory to pick. Rules 1-3 are the recall half, 4-5 the
 * precision half.
 *
 * Rule 4 was written twice, and the second version is the interesting one. Rule 3 originally read
 * "how the user wants things presented", and the selector duly attached "prefers short answers" to
 * every task that produced a reply — which is every task, so the model was reading the rule
 * correctly. Banning it outright then broke the one case whose entire gold answer IS that memory:
 * a bare domain question, where response style is the only thing any memory can contribute. So a
 * standing style preference is not irrelevant and not always relevant — it is DOMINATED: worth
 * nothing beside a substantive memory, and worth having when there is no substantive memory to
 * have. It reads as a priority ordering, not a relevance test, which is the tell that it does not
 * really belong in a per-task selector at all — see docs/todo.md.
 *
 * The final instruction leans toward inclusion, and that lean is an architectural consequence
 * rather than a preference: structured constraints no longer travel through here at all (see
 * memoryDerivedFilter), so this output only shapes the PROSE hint. Over-selecting now costs prompt
 * noise; under-selecting still loses something the user already said. The empty answer stays fully
 * available, and the two empty-gold probes hold it to that.
 */
const SELECT_PROMPT = [
  'You pick which stored memories about this user are RELEVANT to the current task.',
  'Each line is "i: [semantic|episodic] [name] description" — semantic = a generalized preference,',
  'episodic = a past event.',
  '',
  'A memory is relevant when any of these hold:',
  '1. The task will PRODUCE OR EVALUATE LISTINGS — including when they are emailed, summarised or',
  '   compared rather than shown — and the memory narrows or ranks listings (budget, beds, type,',
  '   features, area). Every such preference applies, not just the closest one.',
  '2. The task contains a REFERENCE that cannot be resolved without history — "这套", "那套",',
  '   "上次", "第一个", "#2", "the first one". The episodic memory that resolves it is REQUIRED;',
  '   without it the reference is unresolvable. Pick the ONE episodic that matches every',
  '   descriptor in the reference (what it was, who it was for, when): a near-miss episodic that',
  '   matches some descriptors and contradicts another is a distractor, not a second candidate.',
  '3. The memory says WHICH DIMENSION this user weighs when judging — resale vs living in, schools',
  '   vs commute — and the task asks for a judgement. This is about what to weigh, not about',
  '   response formatting.',
  '',
  'Two rules cut the other way:',
  '4. A global response-style preference (e.g. "prefers short answers") applies to any reply at',
  '   all, so it is the LOWEST-priority memory: it adds nothing once substantive, task-specific',
  '   memories are selected, and you should leave it out then. Include it only when NO substantive',
  '   memory applies and the task still needs a real answer — a plain domain question, say.',
  '5. A memory about a field the TASK ITSELF gives a value for is superseded by the task. This',
  '   holds even when the task phrases it as a change from the old value ("raise the budget to X",',
  '   "预算放宽到 X") — the new value replaces the memory outright, so do not select it. Drop only',
  '   the memory about that one field; every other stored preference still applies.',
  '',
  'Return a JSON array of the relevant indices (e.g. [0,2]).',
  'Return [] — and mean it — when the task is a greeting, small talk, or has nothing to do with',
  'property or with this user\'s history. Otherwise, when a memory plausibly bears on the task,',
  'include it: a missing preference is a worse outcome than an extra one.',
].join('\n');

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
  const maxSem = limits.semantic ?? SELECT_SEMANTIC;   // generalized prefs: few, usually relevant
  const maxEp = limits.episodic ?? SELECT_EPISODIC;    // events: many, mostly irrelevant
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
    { role: 'system', content: SELECT_PROMPT } as ChatMessage,
    { role: 'user', content: `Memories:\n${index}\n\nCurrent task: ${context}` } as ChatMessage,
  ], []);
  const idx = jsonArray(turn.content);
  if (idx === null) return byRank();          // unparseable -> degrade to the deterministic rank
  const picked = idx.filter((i) => memories[i]).map((i) => memories[i]!);
  return capPerType(picked);                  // an EMPTY selection is a valid answer, not a failure
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

/**
 * Slots a SEMANTIC memory may contribute to the seed filter.
 *
 * `city` and `proximity` are deliberately absent. The city is the highest-consequence slot —
 * pinning the wrong one silently changes every result — and the FACTS layer already learns it
 * behind a repetition gate. One remembered sentence should not do in a single step what that
 * layer deliberately makes slow.
 */
const MEMORY_SLOTS = ['maxPrice', 'minPrice', 'beds', 'baths', 'propertyType', 'pool', 'minSqft'] as const;

/**
 * Structured constraints a memory DECLARES, so a remembered preference that maps onto a slot
 * travels the deterministic channel instead of depending on the model noticing it in prose.
 *
 * DECLARED, NOT PARSED — and that is the second design here, replacing the first. The first
 * version ran the regex query parser over memory CONTENT. It worked on hand-written,
 * query-shaped content and broke on the first real consolidation output, because the sub-agent
 * writes descriptive prose and prose routinely negates the opposite of what it means:
 *
 *   "a swimming pool is a hard requirement — listings WITHOUT A POOL must be filtered out"
 *      parsed to  pool: false   — the exact inverse, which would filter out the only homes the
 *                                 user will accept
 *   "budget is 300万; anything UNDER 200万 is too cheap"
 *      parsed to  maxPrice: 2000000   — latching onto the REJECTED figure
 *
 * Two different failure modes in six probes, one of them an inversion. Picking a "safe subset"
 * of slots would just be a guess about which mis-parses are tolerable, so parsing is gone: the
 * writer knows the constraint and states it. Memories created before this schema simply do not
 * seed until they are re-consolidated, which is the safe direction to fail.
 *
 * SEMANTIC ONLY. An episodic memory records what HAPPENED — "看中过 Canterbury 那套 3 居 120 万的"
 * describes a past listing, not a current requirement.
 *
 * NOT gated on selection: a structured constraint is not a "maybe relevant fact", it is a filter
 * the user already stated, and it applies to every search the way a FACT does — which is why
 * facts are injected without consulting the selector either. Staleness is handled downstream:
 * executeTool merges the model's own args OVER these, so a value stated this turn always wins.
 */
export interface SlotConflict {
  type: MemoryType;
  field: string;
  /** memory name -> the value it declares. Always 2+ distinct values. */
  claims: Record<string, unknown>;
}

/**
 * The deterministic net for CROSS-NAME contradiction.
 *
 * Same-name contradiction has always been resolvable: the second write is the same memory being
 * updated, so the new content replaces the old. Two memories with DIFFERENT names claiming
 * opposite things had no net at all — `wants-pool` ("买家坚持要带泳池") and `no-pool-please`
 * ("买家明确说不要泳池") both survived, and naming discipline was load-bearing.
 *
 * WHY THIS COMPARES SLOTS AND NOT PROSE. Reading the contradiction out of the two descriptions is
 * the one approach already disproven here: prose routinely negates the opposite of what it means,
 * and parsing it produced an INVERSION on the very example above (see memoryDerivedFilter's note).
 * A declared slot is the writer's own structured statement, so `{pool: true}` vs `{pool: false}`
 * is a contradiction by construction — no language understanding needed, and no name involved.
 *
 * Scoped WITHIN a type on purpose. A semantic memory states a standing requirement; an episodic
 * records what happened. "看过带泳池那套" does not contradict "不要泳池" — the first is an event,
 * the second a preference, and cross-type pairs are not contradictions at all.
 *
 * What this DOESN'T catch, stated so the boundary is not mistaken for coverage: a contradiction
 * that exists only in prose, between memories that declared no slots. That one still needs the
 * consolidation sub-agent, which is why conflicts are also surfaced to it (see memoryIndex).
 */
export function slotConflicts(memories: MemoryEntry[]): SlotConflict[] {
  const out: SlotConflict[] = [];
  for (const type of ['semantic', 'episodic'] as MemoryType[]) {
    for (const field of MEMORY_SLOTS) {
      const claims: Record<string, unknown> = {};
      for (const m of memories) {
        if (m.type !== type || !m.slots) continue;
        const v = (m.slots as Record<string, unknown>)[field];
        if (v !== undefined && v !== null) claims[m.name] = v;
      }
      const distinct = new Set(Object.values(claims).map((v) => JSON.stringify(v)));
      if (distinct.size > 1) out.push({ type, field, claims });
    }
  }
  return out;
}

export function memoryDerivedFilter(memories: MemoryEntry[]): Partial<SearchFilter> {
  // A CONFLICTED FIELD GOES SILENT. Without this the loop below resolved a contradiction by
  // array order — whichever memory happened to be earlier won, which is not a decision, it is an
  // accident that changes when the file is rewritten. Silence follows the precedent already set
  // for facts: holding the old value is stale, jumping to the other one over-trusts a single
  // record, and there is no supersession link between two separately-named memories to justify
  // either. The user's own words this turn still reach the filter (executeTool merges the
  // model's args over the seed), so silence costs a default, never a stated constraint.
  const conflicted = new Set(
    slotConflicts(memories).filter((c) => c.type === 'semantic').map((c) => c.field),
  );
  const out: Record<string, unknown> = {};
  for (const m of memories) {
    if (m.type !== 'semantic' || !m.slots) continue;
    for (const k of MEMORY_SLOTS) {
      if (conflicted.has(k)) continue;
      const v = (m.slots as Record<string, unknown>)[k];
      if (out[k] === undefined && v !== undefined && v !== null) out[k] = v;
    }
  }
  return out as Partial<SearchFilter>;
}

export function seedFilterFor(profile: UserProfile, _selected: MemoryEntry[] = []): Partial<SearchFilter> {
  // memoryDerivedFilter takes ALL memories, not the selected ones — see its own note. `_selected`
  // is kept in the signature so call sites read symmetrically with profileHint(profile, selected).
  return { ...memoryDerivedFilter(profile.memories), ...preferredFilter(profile) };
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
