/**
 * Multi-turn search conversation (Week 4).
 *
 * One turn: load session -> parse (merging onto the session filter as a patch,
 * Q12) -> decide via structural signals (Q6): reset / clarify city / empty /
 * too-many / results -> persist updated session. Every reply shows the current
 * filter (Q12 transparency). No LLM key required (regex + structural signals).
 */
import { parseQuery } from '../search/parseQuery.js';
import { isKnownCity } from '../search/cityDictionary.js';
import { summarizeFilter, mergeFilter, type SearchFilter, type FilterPatch } from '../search/filters.js';
import { searchActiveListings, searchSignal } from '../search/searchListings.js';
import { formatListingCard, type ListingRow } from '../search/listingRow.js';
import { defaultSessionStore, freshSession, type SessionStore } from './session.js';
import type { LLMClient } from '../llm/client.js';

export type TurnKind = 'results' | 'clarify' | 'too_many' | 'empty' | 'reset';

export interface TurnResult {
  kind: TurnKind;
  reply: string;
  filter: SearchFilter;
  rows?: ListingRow[];
}

export interface TurnOptions {
  store?: SessionStore;
  llm?: LLMClient;
  pageSize?: number;
  /** Auto mode: use this already-extracted filter (LLM + memory) directly, skipping the
   * regex/LLM parse of the message. No fallback parse — what the LLM couldn't extract,
   * regex won't either. Deterministic mode leaves this undefined (parses as before). */
  filter?: SearchFilter;
  /** High-confidence long-term preferences — the LAST resort in the fill chain:
   *    this turn's parse  >  slot carried across turns  >  these  >  ask the user
   * The first two are already folded into the parse (session.filter is its base), so these
   * only ever touch fields still blank afterwards. */
  filterDefaults?: Partial<SearchFilter>;
}

const RESET_RE = /\b(start over|reset|new search|restart|clear)\b/i;
const RESET_ZH = /重新(找|搜|来)|重来|清空|换一(个|批)|从头/;

export function isReset(message: string): boolean {
  return RESET_RE.test(message) || RESET_ZH.test(message);
}

const filterLine = (f: SearchFilter) => `🔎 Current filter: ${summarizeFilter(f)}`;
/** Say out loud which fields we guessed, so a default is never a silent assumption. */
function seedLine(filter: SearchFilter, seeded: string[]): string {
  if (!seeded.length) return '';
  const picked = Object.fromEntries(seeded.map((k) => [k, (filter as Record<string, unknown>)[k]])) as SearchFilter;
  return `\nℹ️ Assumed ${summarizeFilter(picked)} from your usual searches — say otherwise to change it.`;
}

export async function handleSearchTurn(
  userId: string,
  message: string,
  opts: TurnOptions = {},
): Promise<TurnResult> {
  const store = opts.store ?? defaultSessionStore;
  const pageSize = opts.pageSize ?? 5;

  // explicit reset wipes accumulated state (Q12)
  if (isReset(message)) {
    await store.clear(userId);
    return { kind: 'reset', reply: "Starting fresh — what are you looking for?", filter: {} };
  }

  const session = (await store.get(userId)) ?? freshSession();

  // parse this turn as a patch onto the running filter — UNLESS auto already extracted it
  const parsed = opts.filter !== undefined
    ? { filter: mergeFilter(session.filter, opts.filter as FilterPatch), confidence: 'high' as const, clarification: '', rejectedCity: undefined }
    : await parseQuery(message, { base: session.filter, llm: opts.llm, isKnownCity });
  session.filter = parsed.filter;
  session.step += 1;

  // Long-term preferences fill what is STILL blank (see TurnOptions.filterDefaults for the
  // chain). Skipped when the user named a city we don't serve: that is an explicit intent,
  // not a blank — quietly swapping in their usual city would answer a question they didn't ask.
  const seeded: string[] = [];
  if (opts.filterDefaults && !parsed.rejectedCity) {
    const f = session.filter as Record<string, unknown>;
    for (const [k, v] of Object.entries(opts.filterDefaults)) {
      if (v != null && f[k] == null) { f[k] = v; seeded.push(k); }
    }
  }
  const seeds = seedLine(session.filter, seeded);

  // missing required slot -> ask, keep what we have. A seeded city counts as filled: we know
  // where they usually look, so we search there and say so instead of asking every time.
  if (parsed.confidence === 'low' && !session.filter.city) {
    await store.set(userId, session);
    const reply = `${parsed.clarification}\n${filterLine(session.filter)}`;
    return { kind: 'clarify', reply, filter: session.filter };
  }

  // searchable -> hit the DB
  const { rows, total } = await searchActiveListings(session.filter, 1, pageSize);
  session.lastResults = rows;
  await store.set(userId, session);

  const sig = searchSignal(total);
  if (sig?.signal === 'empty') {
    return { kind: 'empty', reply: `${sig.clarification}\n${filterLine(session.filter)}${seeds}`, filter: session.filter };
  }

  const header = sig?.signal === 'too_many'
    ? `${sig.clarification}\nHere are a few of the ${total}:`
    : `Found ${total} match${total === 1 ? '' : 'es'}. Top ${rows.length}:`;
  const cards = rows.map((r, i) => `${i + 1}. ${formatListingCard(r)}`).join('\n\n');
  const reply = `${filterLine(session.filter)}${seeds}\n${header}\n\n${cards}`;

  return { kind: sig?.signal === 'too_many' ? 'too_many' : 'results', reply, filter: session.filter, rows };
}
