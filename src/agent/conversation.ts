/**
 * Multi-turn search conversation (Week 4).
 *
 * One turn: load session -> parse (merging onto the session filter as a patch,
 * Q12) -> decide via structural signals (Q6): reset / clarify city / empty /
 * too-many / results -> persist updated session. Every reply shows the current
 * filter (Q12 transparency). No LLM key required (regex + structural signals).
 */
import { parseQuery } from '../search/parseQuery.js';
import { summarizeFilter, type SearchFilter } from '../search/filters.js';
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
}

const RESET_RE = /\b(start over|reset|new search|restart|clear)\b/i;
const RESET_ZH = /重新(找|搜|来)|重来|清空|换一(个|批)|从头/;

export function isReset(message: string): boolean {
  return RESET_RE.test(message) || RESET_ZH.test(message);
}

const filterLine = (f: SearchFilter) => `🔎 Current filter: ${summarizeFilter(f)}`;

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

  // parse this turn as a patch onto the running filter
  const parsed = await parseQuery(message, { base: session.filter, llm: opts.llm });
  session.filter = parsed.filter;
  session.step += 1;

  // missing required slot -> ask, keep what we have
  if (parsed.confidence === 'low') {
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
    return { kind: 'empty', reply: `${sig.clarification}\n${filterLine(session.filter)}`, filter: session.filter };
  }

  const header = sig?.signal === 'too_many'
    ? `${sig.clarification}\nHere are a few of the ${total}:`
    : `Found ${total} match${total === 1 ? '' : 'es'}. Top ${rows.length}:`;
  const cards = rows.map(formatListingCard).join('\n\n');
  const reply = `${filterLine(session.filter)}\n${header}\n\n${cards}`;

  return { kind: sig?.signal === 'too_many' ? 'too_many' : 'results', reply, filter: session.filter, rows };
}
